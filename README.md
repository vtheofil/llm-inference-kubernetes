# Scalable LLM Inference on Kubernetes — CPU-Only RAG System

> Bachelor thesis project — a production-grade **Retrieval-Augmented Generation (RAG)** system deployed on Kubernetes, engineered for **CPU-only inference** with a focus on autoscaling limits, cold-start mitigation, and empirical model comparison.

**Author:** Vasileios Theofilogiannakos
**Institution:** University of Thessaly — Department of Informatics & Telecommunications
**Supervisor:** Prof. Ioannis Konstantinou
**Defended:** July 2026

---

## 🎯 What this project is

An end-to-end RAG system that serves a synthetic enterprise knowledge base ("Atlas Systems", 447 markdown documents) over a Gradio UI. The system runs on Kubernetes with:

- **Sidecar Ollama container** for local LLM inference (no external API calls)
- **ChromaDB** for vector storage with `multilingual-e5-small` embeddings
- **HPA-driven autoscaling** on Backend replicas
- **DaemonSet-based image pre-warming** across cluster nodes
- **Locust-driven load testing** with mathematically-justified timeout budgets

The research contribution is not the RAG pipeline itself — it's the **empirical study** of *why* naive Kubernetes autoscaling fails for LLM workloads, and *what layered mitigation* actually works.

---

## 🔬 Key findings

Under a 10-concurrent-user Locust load, comparing **Phi-3 Mini (3.8B)** and **Mistral 7B**, both Q4_K_M-quantized:

| Metric | Phi-3 Mini | Mistral 7B |
|---|---|---|
| Successful requests | **89** | 19 |
| Median latency | **76 s** | 192 s |
| p95 latency | **263 s** | 831 s (~14 min) |
| Throughput (RPS) | **0.099** | 0.020 |
| Failure rate | 0 % | 0 % |

**Take-away:** For CPU-only interactive serving, the smaller model wins on every operational metric — 4.7× throughput, ~3× lower median latency — at negligible quality cost for factual retrieval.

---

## 🏗️ Architecture

```
┌──────────────────┐         ┌───────────────────────────────┐         ┌──────────────────┐
│  Gradio Pod      │  HTTP   │  Backend Pod                  │  HTTP   │  ChromaDB Pod    │
│  (UI, port 7860) │────────►│  ├── FastAPI (:8000)          │────────►│  (vector DB)     │
└──────────────────┘         │  └── Ollama sidecar (:11434)  │         │  + PVC 2Gi       │
                             │      via localhost (0 hops)   │         └──────────────────┘
                             └───────────────────────────────┘
                                       ▲
                                       │  HPA target: 70% CPU
                                       │  minReplicas: 1  →  maxReplicas: 5
                                       │
                             ┌───────────────────────────────┐
                             │  Ollama image preloader       │
                             │  DaemonSet (all nodes)        │
                             │  Pre-warms phi3 + mistral     │
                             └───────────────────────────────┘
```

**Design decisions and their rationale are documented inline** in `k8s/deployment.yaml` (memory tuning history, OOM analysis, KEEP_ALIVE trade-offs) and in the thesis PDF.

---

## 🧊 5-Layer Cold-Start Mitigation Strategy

The observation driving this design: HPA scales pods, but **new pods don't serve traffic for 1–2 minutes** after being scheduled. Naive HPA gives you replicas — not capacity.

| # | Layer | Purpose | File |
|---|---|---|---|
| 1 | **Pre-baked container image** | Model weights baked into the image → no runtime `ollama pull` | [`Dockerfile.ollama`](Dockerfile.ollama) |
| 2 | **DaemonSet image pre-warming** | Image cached on every node → kubelet skips the ~5 GB registry pull on scale-up | [`k8s/preload-daemonset.yaml`](k8s/preload-daemonset.yaml) |
| 3 | **Conditional data ingestion** | Init container skips re-ingest when the ChromaDB collection already has data → eliminates a race condition where a scaling pod would wipe live data mid-serve | [`app/ingest.py`](app/ingest.py) |
| 4 | **`postStart` warmup + custom readiness probe** | Dummy inference forces the model into RAM *before* the pod is admitted to the Service; readiness stays a shallow `ollama list \| grep` so it never queues behind real inference | [`k8s/deployment.yaml`](k8s/deployment.yaml) |
| 5 | **`OLLAMA_KEEP_ALIVE=24h`** | Model stays resident in RAM — trades memory footprint for a warm-pool guarantee | [`k8s/deployment.yaml`](k8s/deployment.yaml) |

Net effect: pod ready-time drops from **~4 minutes → ~60 seconds**.

---

## 🧰 Tech Stack

**Application:** Python 3.11, FastAPI, Gradio, Pydantic, `sentence-transformers` (`multilingual-e5-small`)
**Vector DB:** ChromaDB (HNSW, cosine similarity, 384-dim embeddings)
**LLM serving:** Ollama (llama.cpp under the hood), quantized `phi3:mini` and `mistral` in GGUF Q4_K_M
**Orchestration:** Kubernetes (Deployments, Services, HPA, DaemonSet, ConfigMap, Secret, PVC), Docker
**Build/deploy:** Makefile-driven CI (build → push → rolling deploy → rollback), Kaniko (in-cluster image build for slow-uplink scenarios)
**Load testing:** Locust (in-cluster deployment), with `Connection: close` per request to work around kube-proxy L4 sticky-connection behavior

---

## 📁 Repository layout

```
.
├── app/                      # FastAPI backend + RAG pipeline
│   ├── main.py               # /chat, /chat/stream, /sources, /healthz, /readyz
│   ├── rag.py                # Retrieval, prompt build, LLM call, follow-up detection
│   └── ingest.py             # Markdown → chunks → embeddings → ChromaDB (idempotent)
│
├── ui/
│   └── gradio_app.py         # Two-view chat UI (conversation + retrieved sources)
│
├── data/                     # 447 synthetic markdown documents ("Atlas Systems")
│   ├── employees/            # 150
│   ├── projects/             # 50
│   ├── departments/          # 10
│   ├── policies/             # 10
│   ├── it_faq/               # 10
│   ├── meetings/             # 150
│   ├── monthly_plans/        # 36
│   ├── job_postings/         # 20
│   └── about/                # 11
│
├── scripts/
│   └── generate_atlas_data.py    # Faker-based generator (seed=42, no LLM)
│
├── k8s/                      # Kubernetes manifests
│   ├── configmap.yaml
│   ├── deployment.yaml       # ChromaDB + Backend (init/main/sidecar) + Gradio
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── preload-daemonset.yaml    # Dual-model image pre-warmer
│   ├── locust-configmap.yaml
│   ├── locust-deployment.yaml
│   └── kaniko-build-mistral.yaml # Fallback in-cluster image build
│
├── locust/
│   └── locustfile.py         # Load-test scenario (weighted tasks, semantic checks)
│
├── Dockerfile                # Backend image
├── Dockerfile.gradio         # UI image
├── Dockerfile.ollama         # Pre-baked phi3:mini image (~5 GB)
├── Dockerfile.ollama.mistral # Pre-baked mistral image (~7 GB)
├── docker-compose.yml        # Local dev stack
├── Makefile                  # build / push / deploy / rollback / status
└── requirements.txt
```

---

## 🚀 Getting started

### Local development

```bash
# Bring up the stack (ChromaDB + FastAPI + Ollama + Gradio) with docker-compose
docker-compose up --build

# Or run the backend and UI directly (needs a running ChromaDB + Ollama):
pip install -r requirements.txt
python -m app.ingest                     # one-off, populates ChromaDB
uvicorn app.main:app --host 0.0.0.0 --port 8000
python ui/gradio_app.py                  # → http://localhost:7860
```

### Kubernetes deployment

```bash
# Set kubectl context to the target cluster / namespace, then:
make build push          # build + push backend and gradio images
make deploy              # applies configmap → daemonset → deployment → svc → hpa → locust
make status              # kubectl get pods,svc,hpa,daemonset
make rollback            # kubectl rollout undo backend-deployment
```

The Makefile substitutes image tags into the manifests at deploy time — the source YAML uses `__VERSION__`, `__OLLAMA_VERSION__`, `__MISTRAL_VERSION__` placeholders.

### Running the load test

```bash
# In-cluster (Locust runs as a pod inside the target namespace):
kubectl apply -f k8s/locust-configmap.yaml
kubectl apply -f k8s/locust-deployment.yaml
# Open http://<node-ip>:30089 for the Locust web UI

# Or local (needs kubectl port-forward svc/backend-service 8000:8000):
locust -f locust/locustfile.py --host http://localhost:8000
```

---

## 📊 Selected implementation notes

- **`OLLAMA_NUM_PARALLEL=1`** — Enforced serial inference per pod. Concurrent inferences each allocate a private KV-cache; without serialization the pod OOMs mid-request. Trade-off: lower per-pod throughput, absorbed by HPA horizontal scaling.
- **Locust `timeout=900 s`** — Not arbitrary: with 3 concurrent users and 5 pods, ~4% of requests suffer a triple kube-proxy collision (all three routed to the same pod). Serial inference × 3 × ~250 s ≈ 750 s worst case + 20 % margin. Documented inline.
- **`Connection: close` on every Locust request** — Without this, HTTP/1.1 keep-alive causes all workers to reuse the same TCP connection → the same backend pod → HPA autoscaling appears to fail. This is a subtle kube-proxy L4 selection issue, not a K8s bug.
- **Reduced `TOP_K` from 5 → 3 when switching to Phi-3 Mini** — The smaller model gets confused by too many chunks; fewer, more relevant retrievals produce more focused answers.

---

## 📚 Full thesis (Greek)

See [`thesis/Theofilogiannakos_Vasileios_Thesis.pdf`](thesis/Theofilogiannakos_Vasileios_Thesis.pdf) for the complete 100+ page document, including theoretical background, related work, detailed experimental protocol, and quality evaluation.

---

## 🙏 Acknowledgments

Special thanks to **Prof. Ioannis Konstantinou** for the supervision, guidance, and the vdcloud cluster access that made the experimental evaluation possible.

---

## 📄 License

MIT — see [LICENSE](LICENSE) if provided.
