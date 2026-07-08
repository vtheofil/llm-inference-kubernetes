"""
locustfile.py — Load testing for the Travel Assistant backend.

Usage (from the project root):

  pip install locust

  # Run with web UI (open http://localhost:8089 in browser)
  locust -f locust/locustfile.py --host http://localhost:8000

  # Run headless (10 users, ramp 1/s, 2-minute run)
  locust -f locust/locustfile.py --host http://localhost:8000 \
         --headless -u 10 -r 1 -t 120s \
         --csv=results/load_test

  # vdcloud example (kubectl port-forward in another terminal first)
  #   kubectl port-forward -n vtheofil-priv svc/backend-service 8000:8000
  locust -f locust/locustfile.py --host http://localhost:8000 \
         --headless -u 15 -r 2 -t 10m --csv=results/vdcloud

Metrics collected:
  - Request latency (min / avg / median / p95 / max)
  - Throughput (requests/sec)
  - Error rate (% failures)
  - Concurrent user behaviour under increasing load

The high latency from phi3:mini (CPU-only) is intentional — it keeps CPU
utilisation high long enough for the HPA to trigger a scale-out event.

Important: all queries are English-only because the production prompt
enforces an "English-only response" rule (the corpus is English; smaller
models mix languages otherwise). Each query also mentions a destination
so the pre-LLM `_detect_destination` guard routes to retrieval + the LLM
rather than short-circuiting to the canned clarification reply — that
short-circuit returns in ~10 ms and would NOT push CPU above HPA's 70%
threshold, defeating the autoscaling demo.
"""

import random
from locust import HttpUser, task, between, events

# ── ENTERPRISE_QUERIES sent to POST /chat ─────────────────────────────────────────
# 
ENTERPRISE_QUERIES = [
    "Who manages Project Atlas?",
    "What is the budget of Project Atlas?",
    "Who is the tech lead of Project Atlas?",
    "Who leads the Engineering department?",
    "Which employees work on Project Atlas?",
    "What is the status of Project Atlas?",
    "How can an employee reset their VPN password?",
    "What is the annual leave policy?",
    "What are the Engineering priorities for March 2026?",
]


class EnterpriseUser(HttpUser):
    """
    Simulates a user querying the Travel Assistant.

    wait_time: random pause between 1–5 seconds between requests.
    This models realistic think-time while keeping enough concurrent
    load to push CPU utilisation above the 70% HPA threshold.
    """

    wait_time = between(1, 5)

    # Force a new TCP connection on every request so kube-proxy's random L4
    # selection re-rolls the pod for each call. Without this, every Locust
    # worker reuses the same persistent HTTP/1.1 connection and therefore
    # the same backend pod for its entire lifetime — defeating the
    # autoscaling demo (we observed this: 15 workers all stuck on one
    # pod while two other pods sat idle).
    _NO_KEEPALIVE = {"Connection": "close"}

    @task(10)
    def chat_blocking(self):
        """POST /chat — blocking response (primary load generator)."""
        query = random.choice(ENTERPRISE_QUERIES)
        with self.client.post(
            "/chat",
            json={"message": query},
            headers=self._NO_KEEPALIVE,
            catch_response=True,
            timeout=300,   # phi3 on CPU is much faster but cold pods need headroom
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if not data.get("response"):
                    response.failure("Empty response body")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")

    @task(3)
    def health_check(self):
        """GET /healthz — lightweight probe, verifies pod is alive."""
        with self.client.get(
            "/healthz",
            headers=self._NO_KEEPALIVE,
            catch_response=True,
            timeout=5,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def metrics_check(self):
        """GET /metrics — reads latency counters exposed by the backend."""
        with self.client.get(
            "/metrics",
            headers=self._NO_KEEPALIVE,
            catch_response=True,
            timeout=6,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics failed: {response.status_code}")


# ── Event hooks for extra console logging ─────────────────────────────────────

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    if exception:
        print(f"[ERROR] {request_type} {name} — {exception}")
    elif name == "/chat" and response_time > 60_000:
        # Log slow LLM responses (expected on CPU-only Mistral)
        print(f"[SLOW ] {request_type} {name} — {response_time/1000:.1f}s")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("=" * 60)
    print("Atlas Systems — Load Test Started")
    print(f"Target host : {environment.host}")
    print("Endpoints   : POST /chat (x10), GET /health (x3), GET /metrics (x1)")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats.total
    print("=" * 60)
    print("Load Test Finished — Summary")
    print(f"  Total requests : {stats.num_requests}")
    print(f"  Failures       : {stats.num_failures}")
    print(f"  Avg latency    : {stats.avg_response_time:.0f} ms")
    print(f"  Median latency : {stats.median_response_time:.0f} ms")
    print(f"  95th pct       : {stats.get_response_time_percentile(0.95):.0f} ms")
    print(f"  Throughput     : {stats.total_rps:.2f} req/s")
    print("=" * 60)
