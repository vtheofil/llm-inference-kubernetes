# Dockerfile — FastAPI backend (multi-stage build)
#
# Stage 1: install dependencies (cached layer, rebuilds only when requirements change)
# Stage 2: copy application code into a slim final image

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed by some Python packages (e.g. chromadb, sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~2 GB vs the default CUDA build)
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# curl is needed by entrypoint.sh to health-check ChromaDB and Ollama
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin           /usr/local/bin

# Copy application code
COPY app/    ./app/
COPY data/   ./data/
COPY entrypoint.sh ./

RUN chmod +x entrypoint.sh

EXPOSE 8000

# entrypoint.sh handles waiting for dependencies, ingestion, then starts uvicorn
ENTRYPOINT ["./entrypoint.sh"]
