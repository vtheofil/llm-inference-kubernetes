# Makefile — Atlas Systems / Ptyxiakh deployment workflow
# Follows professor's pattern: config/secret → storage → database → application
#
# Target environment: vdcloud cluster (namespace: vtheofil-priv).
#
# Image versioning policy:
#   - $(VERSION)         applies to backend + gradio images
#                        (bump on every app/data change)
#   - $(OLLAMA_VERSION)  applies ONLY to the pre-baked Ollama image
#                        (rarely changes — only when the LLM model itself
#                        or the Ollama runtime is updated; saves 5GB
#                        rebuilds/re-pulls on every app change)
#
# Usage:
#   make build         Build backend + gradio images
#   make push          Push backend + gradio images
#   make deploy        Deploy to vdcloud (namespace: vtheofil-priv)
#   make status        Show pod / svc / HPA / DaemonSet status
#   make rollback      Roll back the backend deployment
#   make clean         Remove locally built backend + gradio images
#   make build-ollama  Build only the pre-baked Ollama image (slow, ~5 GB)
#   make push-ollama   Push only the pre-baked Ollama image

DOCKER_USER     = vtheofil1

# v1.0.2: programmatic destination guard, history-slicing bug fix
# (HISTORY_RETRIEVAL_TURNS = 0 used to silently include the whole history
# due to Python's lst[-0:] semantics), neutral prompt suffix, hardened
# English-only language rule, Ollama memory limit raised 4→6→8 GiB after
# observed OOM, CPU 4 cores stays.
#
# v1.0.3: Sustained-load OOM mitigation. After two distinct OOM events
# under Locust load with 3 concurrent users:
#   - First (8 GiB) → pathological long-output request OOMed mid-gen.
#   - Second (12 GiB, ~16 min into run) → gradual memory growth from
#     unbounded num_predict producing a 2500+ token itinerary.
# Two-pronged fix:
#   - Cap `num_predict: 1024` in app/rag.py (bounds worst-case KV cache)
#   - Bump Ollama memory limit 12 → 16 GiB (safety headroom)
# OLLAMA_NUM_PARALLEL=1 and OLLAMA_KEEP_ALIVE=24h kept (warm-pool intent).
#
# v2.0.0: Domain pivot — Travel RAG → Atlas Systems Enterprise Knowledge.
# Greek-language exploration (phi3, phi3.5, Meltemi, Krikri) determined
# that small CPU-friendly models cannot reliably serve Greek RAG queries.
# We pivoted the application layer to an English-language synthetic
# enterprise dataset (~550 markdown docs about employees, projects,
# departments, policies, IT FAQs, meetings) that aligns with phi3:mini's
# strength in factual extraction. Infrastructure (HPA, DaemonSet, image
# pre-baking, postStart warmup) is unchanged — only the corpus, the
# system prompt, and the ChromaDB collection name change.
#
# v2.0.1: Fixed SourceRecord pydantic model in app/main.py — was still
# requiring legacy travel-domain fields (country, city, topic) which
# caused HTTP 500 on /chat after the Atlas pivot. New schema mirrors the
# enterprise metadata: type, title, name, department, project. Also
# Atlas-rebranded the Gradio UI (title, examples, placeholder).
#
# v2.0.3: Dataset + retrieval quality improvements:
#   - Enforced ONE leadership role per department in the data generator
#     (eliminates the "3 CMOs / 2 Engineering Directors" RAG-confusion bug).
#   - Added query-intent detection + ChromaDB metadata filtering in rag.py
#     (e.g. "Tell me about Project Cygnus" filters to type=project chunks,
#     preventing monthly_plan documents that merely mention Cygnus from
#     polluting the answer).
#
# v2.0.4: Conversational + UX fixes:
#   - Follow-up question detection in rag.py — short pronoun-style queries
#     ("tell me more", "what about it") re-use the previous user turn as
#     the retrieval anchor so the conversation can continue without losing
#     topic ("Tell me about Project Cygnus" → "give me more info about it"
#     now correctly retrieves Cygnus chunks instead of random documents).
#   - Two-step Gradio submit flow — the user's message + a "Thinking…"
#     placeholder are shown IMMEDIATELY when they press Send. The
#     placeholder is then replaced by the real LLM response when it
#     arrives. Eliminates the long blank period (30-60s) where the user
#     could not tell whether the message had been received.
# v2.0.5: Added DEBUG log on /chat (message + history_turns) to diagnose
# whether the two-step Gradio submit flow is actually forwarding chat
# history to the backend. Required to confirm/refute the suspicion that
# follow-up retrieval is not triggering because Gradio is sending an
# empty history.
VERSION         = v2.1.0

# Pre-baked Ollama image is pinned to v1.0.3 — the LLM (phi3:mini) and
# Ollama runtime have not changed since then, so we avoid 5 GB rebuilds
# and 5 GB re-pulls (DaemonSet pre-warm) on every backend release.
OLLAMA_VERSION  = v1.0.3

# Pre-baked Ollama (mistral) image — separate from the phi3 image to keep
# each pre-baked image small. Built locally via Dockerfile.ollama.mistral
# and pushed alongside the phi3 image.
MISTRAL_IMAGE_REPO = ptyxiakh-ollama-mistral
MISTRAL_VERSION    = v1.0.0

BACKEND_IMG  = $(DOCKER_USER)/ptyxiakh-backend:$(VERSION)
GRADIO_IMG   = $(DOCKER_USER)/ptyxiakh-gradio:$(VERSION)
OLLAMA_IMG   = $(DOCKER_USER)/ptyxiakh-ollama:$(OLLAMA_VERSION)
MISTRAL_IMG  = $(DOCKER_USER)/$(MISTRAL_IMAGE_REPO):$(MISTRAL_VERSION)

NAMESPACE    = vtheofil-priv

# ── Docker ───────────────────────────────────────────────────────────────────

.PHONY: build
build:
	@echo "==> Building backend image $(BACKEND_IMG)"
	docker build -t $(BACKEND_IMG) .
	@echo "==> Building gradio  image $(GRADIO_IMG)"
	docker build -t $(GRADIO_IMG) -f Dockerfile.gradio .

.PHONY: push
push: build
	@echo "==> Pushing $(BACKEND_IMG)"
	docker push $(BACKEND_IMG)
	@echo "==> Pushing $(GRADIO_IMG)"
	docker push $(GRADIO_IMG)

.PHONY: build-ollama
build-ollama:
	@echo "==> Building pre-baked Ollama image $(OLLAMA_IMG) (~5 GB, takes 2-3 min)"
	docker build -t $(OLLAMA_IMG) -f Dockerfile.ollama .

.PHONY: push-ollama
push-ollama: build-ollama
	@echo "==> Pushing $(OLLAMA_IMG) (~5 GB, takes 3-10 min depending on connection)"
	docker push $(OLLAMA_IMG)

.PHONY: clean
clean:
	docker rmi $(BACKEND_IMG) $(GRADIO_IMG) || true

# ── vdcloud deploy (namespace: vtheofil-priv) ───────────────────────────────

.PHONY: deploy
deploy:
	@echo "==> Deploying to vdcloud (namespace: $(NAMESPACE))"
	@echo "    Application version: $(VERSION)"
	@echo "    Ollama (phi3)    image version: $(OLLAMA_VERSION)"
	@echo "    Ollama (mistral) image version: $(MISTRAL_VERSION)"
	@echo "--- 1. Config / Secret"
	kubectl apply -f k8s/configmap.yaml -n $(NAMESPACE)
	kubectl apply -f k8s/secret.yaml    -n $(NAMESPACE)
	@echo "--- 2. Image preloader DaemonSet (pre-pulls BOTH phi3 and mistral images on every node)"
	sed 's|__VERSION__|$(VERSION)|g; s|__OLLAMA_VERSION__|$(OLLAMA_VERSION)|g; s|__MISTRAL_VERSION__|$(MISTRAL_VERSION)|g' \
	    k8s/preload-daemonset.yaml | kubectl apply -f -
	@echo "--- 3. Storage + Database + Application (deployment.yaml)"
	sed 's|__VERSION__|$(VERSION)|g; s|__OLLAMA_VERSION__|$(OLLAMA_VERSION)|g; s|__MISTRAL_VERSION__|$(MISTRAL_VERSION)|g' \
	    k8s/deployment.yaml | kubectl apply -f -
	@echo "--- 4. Services"
	kubectl apply -f k8s/service.yaml -n $(NAMESPACE)
	@echo "--- 5. HPA"
	kubectl apply -f k8s/hpa.yaml -n $(NAMESPACE)
	@echo "--- 6. Locust"
	kubectl apply -f k8s/locust-configmap.yaml -n $(NAMESPACE)
	kubectl apply -f k8s/locust-deployment.yaml -n $(NAMESPACE)
	@echo "==> Waiting for backend rollout …"
	kubectl rollout status deployment/backend-deployment  \
	    -n $(NAMESPACE) --timeout=15m
	kubectl rollout status deployment/gradio-deployment   \
	    -n $(NAMESPACE) --timeout=5m

.PHONY: status
status:
	kubectl get pods,svc,hpa,daemonset -n $(NAMESPACE)

.PHONY: rollback
rollback:
	kubectl rollout undo deployment/backend-deployment -n $(NAMESPACE)
	kubectl rollout status deployment/backend-deployment \
	    -n $(NAMESPACE) --timeout=15m

.PHONY: deploy-locust
deploy-locust:
	@echo "==> Deploying Locust"
	kubectl apply -f k8s/locust-configmap.yaml -n $(NAMESPACE)
	kubectl apply -f k8s/locust-deployment.yaml -n $(NAMESPACE)

.PHONY: delete-locust
delete-locust:
	kubectl delete -f k8s/locust-deployment.yaml -n $(NAMESPACE) || true
	kubectl delete -f k8s/locust-configmap.yaml -n $(NAMESPACE) || true
