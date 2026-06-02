"""
locustfile.py — Load testing for the Travel Assistant backend.

Usage (from the project root):

  pip install locust

  # Run with web UI (open http://localhost:8089 in browser)
  locust -f locust/locustfile.py --host http://<BACKEND_HOST>:8000

  # Run headless (10 users, ramp 1/s, 2-minute run)
  locust -f locust/locustfile.py --host http://<BACKEND_HOST>:8000 \
         --headless -u 10 -r 1 -t 120s \
         --csv=results/load_test

  # Minikube example (NodePort 30001)
  locust -f locust/locustfile.py --host http://$(minikube ip):30001 \
         --headless -u 20 -r 2 -t 180s --csv=results/minikube

  # vdcloud example (kubectl port-forward in another terminal first)
  #   kubectl port-forward -n vtheofil-priv svc/backend-service 8000:8000
  locust -f locust/locustfile.py --host http://localhost:8000 \
         --headless -u 20 -r 2 -t 180s --csv=results/vdcloud

Metrics collected:
  - Request latency (min / avg / median / p95 / max)
  - Throughput (requests/sec)
  - Error rate (% failures)
  - Concurrent user behaviour under increasing load

The high latency from Mistral 7B (CPU-only) is intentional — it keeps CPU
utilisation high long enough for the HPA to trigger a scale-out event.
"""

import random
from locust import HttpUser, task, between, events

# ── Travel queries sent to POST /chat ─────────────────────────────────────────
# Realistic prompts covering all five supported destinations.
TRAVEL_QUERIES = [
    "Θέλω να πάω Ρώμη 5 μέρες με 700€. Φτιάξε μου αναλυτικό πρόγραμμα.",
    "Ταξίδι στο Τόκιο 7 μέρες με budget 1500€. Τι προτείνεις;",
    "Παρίσι 3 μέρες με 500€. Ποια αξιοθέατα να δω και πόσο κοστίζουν;",
    "Barcelona 4 days with €600 budget. Give me a detailed itinerary.",
    "Θέλω Σαντορίνη και Αθήνα 6 μέρες με 900€. Πώς να οργανωθώ;",
    "Japan 10 days, budget 2000€. Best cities and daily schedule please.",
    "Rome 3 days with 400€, solo traveler. Budget accommodation tips?",
    "Ελλάδα 5 μέρες με 600€. Ποιες πόλεις να επισκεφτώ;",
    "France 7 days 1200€. Paris plus one region outside Paris.",
    "Spain 5 days 800€. Madrid and Barcelona, how to split the time?",
    "Ιαπωνία 14 μέρες με 3000€. Αναλυτικό ταξιδιωτικό πλάνο παρακαλώ.",
    "Italy 6 days 900€. Rome, Florence, Venice — is it doable?",
]


class TravelUser(HttpUser):
    """
    Simulates a user querying the Travel Assistant.

    wait_time: random pause between 1–5 seconds between requests.
    This models realistic think-time while keeping enough concurrent
    load to push CPU utilisation above the 70% HPA threshold.
    """

    wait_time = between(1, 5)

    @task(10)
    def chat_blocking(self):
        """POST /chat — blocking response (primary load generator)."""
        query = random.choice(TRAVEL_QUERIES)
        with self.client.post(
            "/chat",
            json={"message": query},
            catch_response=True,
            timeout=300,   # Mistral on CPU can take up to 5 min
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
        """GET /health — lightweight probe, verifies pod is alive."""
        with self.client.get("/health", catch_response=True, timeout=5) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def metrics_check(self):
        """GET /metrics — reads latency counters exposed by the backend."""
        with self.client.get("/metrics", catch_response=True, timeout=5) as response:
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
    print("Travel Assistant — Load Test Started")
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
