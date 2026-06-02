# Makefile — Travel Assistant deployment workflow
# Follows professor's pattern: config/secret → storage → database → application
#
# Usage:
#   make build          Build both Docker images (backend + gradio)
#   make push           Push images to Docker Hub
#   make deploy-mk      Deploy to Minikube (GCP VM, namespace: travel-assistant)
#   make deploy-vd      Deploy to vdcloud  (namespace: vtheofil-priv)
#   make status-mk      Show pod/HPA status on Minikube
#   make status-vd      Show pod/HPA status on vdcloud
#   make rollback-mk    Roll back backend on Minikube
#   make rollback-vd    Roll back backend on vdcloud
#   make clean          Remove locally built images

DOCKER_USER  = vtheofil1
VERSION      = v1.0.0

BACKEND_IMG  = $(DOCKER_USER)/ptyxiakh-backend:$(VERSION)
GRADIO_IMG   = $(DOCKER_USER)/ptyxiakh-gradio:$(VERSION)

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

.PHONY: clean
clean:
	docker rmi $(BACKEND_IMG) $(GRADIO_IMG) || true

# ── Minikube (GCP VM, namespace: travel-assistant) ───────────────────────────

.PHONY: deploy-mk
deploy-mk:
	@echo "==> Deploying to Minikube (namespace: travel-assistant)"
	@echo "--- 1. Namespace"
	kubectl apply -f k8s-minikube/namespace.yaml
	@echo "--- 2. Config / Secret"
	kubectl apply -f k8s-minikube/configmap.yaml -n travel-assistant
	kubectl apply -f k8s-minikube/secret.yaml    -n travel-assistant
	@echo "--- 3. Storage + Database + Application (deployment.yaml)"
	sed 's|__VERSION__|$(VERSION)|g' k8s-minikube/deployment.yaml \
	    | kubectl apply -f -
	@echo "--- 4. Services"
	kubectl apply -f k8s-minikube/service.yaml -n travel-assistant
	@echo "--- 5. HPA"
	kubectl apply -f k8s-minikube/hpa.yaml -n travel-assistant
	@echo "==> Waiting for rollout …"
	kubectl rollout status deployment/backend-deployment  \
	    -n travel-assistant --timeout=5m
	kubectl rollout status deployment/gradio-deployment   \
	    -n travel-assistant --timeout=5m

.PHONY: status-mk
status-mk:
	kubectl get pods,svc,hpa -n travel-assistant

.PHONY: rollback-mk
rollback-mk:
	kubectl rollout undo deployment/backend-deployment -n travel-assistant
	kubectl rollout status deployment/backend-deployment \
	    -n travel-assistant --timeout=3m

# ── vdcloud (namespace: vtheofil-priv) ──────────────────────────────────────

.PHONY: deploy-vd
deploy-vd:
	@echo "==> Deploying to vdcloud (namespace: vtheofil-priv)"
	@echo "--- 1. Config / Secret"
	kubectl apply -f k8s/configmap.yaml -n vtheofil-priv
	kubectl apply -f k8s/secret.yaml    -n vtheofil-priv
	@echo "--- 2. Storage + Database + Application (deployment.yaml)"
	sed 's|__VERSION__|$(VERSION)|g' k8s/deployment.yaml \
	    | kubectl apply -f -
	@echo "--- 3. Services"
	kubectl apply -f k8s/service.yaml -n vtheofil-priv
	@echo "--- 4. HPA"
	kubectl apply -f k8s/hpa.yaml -n vtheofil-priv
	@echo "==> Waiting for rollout …"
	kubectl rollout status deployment/backend-deployment  \
	    -n vtheofil-priv --timeout=5m
	kubectl rollout status deployment/gradio-deployment   \
	    -n vtheofil-priv --timeout=5m

.PHONY: status-vd
status-vd:
	kubectl get pods,svc,hpa -n vtheofil-priv

.PHONY: rollback-vd
rollback-vd:
	kubectl rollout undo deployment/backend-deployment -n vtheofil-priv
	kubectl rollout status deployment/backend-deployment \
	    -n vtheofil-priv --timeout=3m
