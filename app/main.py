"""
main.py — FastAPI backend for the Travel Assistant.

Endpoints:
  POST /chat          — blocking chat; returns full response
  POST /chat/stream   — streaming chat; returns text/plain stream
  GET  /health        — Kubernetes liveness/readiness probe
  GET  /metrics       — basic request count and latency metrics
"""

import os
import time
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from app.rag import generate_response, generate_response_stream, _get_embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── In-memory metrics store ──────────────────────────────────────────────────
_metrics: dict = {
    "total_requests":  0,
    "total_latency_ms": 0.0,
    "errors":          0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the embedding model at startup so the first request is fast."""
    logger.info("Pre-loading embedding model …")
    import asyncio
    await asyncio.get_event_loop().run_in_executor(None, _get_embedder)
    logger.info("Travel Assistant API ready.")
    yield
    logger.info("Travel Assistant API shut down.")


app = FastAPI(
    title="Travel Assistant API",
    description="RAG-powered travel planning with Ollama + Mistral 7B",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────

class HistoryTurn(BaseModel):
    """One turn of conversation history (user message + assistant reply)."""
    user: str
    assistant: str = ""

class ChatRequest(BaseModel):
    message: str
    # Optional chat history for context-aware retrieval and generation.
    # Each entry is a previous turn: {"user": "...", "assistant": "..."}.
    history: list[HistoryTurn] = []

class ChatResponse(BaseModel):
    response: str
    latency_ms: float


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    """Legacy health endpoint (kept for backwards-compat). Use /healthz or /readyz."""
    return {"status": "ok"}


@app.get("/healthz", tags=["ops"])
def healthz():
    """
    Kubernetes liveness probe.
    Returns 200 immediately — just confirms the process is alive.
    Does NOT check dependencies (ChromaDB / Ollama).
    """
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
def readyz():
    """
    Kubernetes readiness probe.
    Checks that both ChromaDB and Ollama are reachable before the pod
    is allowed to receive traffic.  Returns 503 if either dependency is down.
    """
    errors = []

    # ── Check ChromaDB ───────────────────────────────────────────────────────
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = os.getenv("CHROMA_PORT", "8000")
    try:
        r = httpx.get(f"http://{chroma_host}:{chroma_port}/api/v2/heartbeat", timeout=3.0)
        r.raise_for_status()
    except Exception as exc:
        errors.append(f"chromadb: {exc}")

    # ── Check Ollama ─────────────────────────────────────────────────────────
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=3.0)
        r.raise_for_status()
    except Exception as exc:
        errors.append(f"ollama: {exc}")

    if errors:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "errors": errors},
        )
    return {"status": "ready"}


@app.get("/metrics", tags=["ops"])
def get_metrics():
    """Basic observability metrics for load-testing / HPA demonstration."""
    avg = (
        _metrics["total_latency_ms"] / _metrics["total_requests"]
        if _metrics["total_requests"] > 0
        else 0.0
    )
    return {
        "total_requests":  _metrics["total_requests"],
        "avg_latency_ms":  round(avg, 2),
        "errors":          _metrics["errors"],
    }


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest):
    """
    Blocking chat endpoint.
    Retrieves relevant context from ChromaDB and generates a full response
    from Ollama Mistral 7B before returning.
    """
    _metrics["total_requests"] += 1
    t0 = time.time()
    try:
        history = [h.model_dump() for h in req.history]
        response = generate_response(req.message, history)
        latency  = (time.time() - t0) * 1000
        _metrics["total_latency_ms"] += latency
        logger.info("Chat OK — %.0f ms", latency)
        return ChatResponse(response=response, latency_ms=round(latency, 1))
    except Exception as exc:
        _metrics["errors"] += 1
        logger.error("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream", tags=["chat"])
def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint.
    Tokens are pushed to the client as they are generated by Ollama.
    The Gradio UI connects here to display a live typing effect.
    """
    _metrics["total_requests"] += 1
    history = [h.model_dump() for h in req.history]

    def _generator():
        try:
            for token in generate_response_stream(req.message, history):
                yield token
        except Exception as exc:
            _metrics["errors"] += 1
            logger.error("Stream error: %s", exc)
            yield f"\n[Error: {exc}]"

    return StreamingResponse(_generator(), media_type="text/plain")
