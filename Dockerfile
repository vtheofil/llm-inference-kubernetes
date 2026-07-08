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

# curl is used by Kubernetes init container to health-check ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin           /usr/local/bin

# Pin HuggingFace / sentence-transformers cache inside /app so the
# non-root user can read it at runtime (no home-directory required).
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Pre-download the embedding model at build time (runs as root → no
# permission issues).  Bakes the ~470 MB weights into the image so
# startup is instant and the container never needs internet access.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small')"

# Copy application code
COPY app/    ./app/
COPY data/   ./data/

# Non-root user (professor's pattern #12)
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# In Kubernetes, an initContainer runs `python -m app.ingest` before this
# starts, and the readiness probe on ChromaDB/Ollama gates traffic. In
# docker-compose, depends_on with healthcheck handles dependency ordering.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
