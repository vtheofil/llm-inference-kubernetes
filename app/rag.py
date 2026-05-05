"""
rag.py — RAG (Retrieval-Augmented Generation) pipeline.

1. Embeds the user query with a local sentence-transformers model.
2. Retrieves the top-k most relevant chunks from ChromaDB.
3. Builds a structured travel-assistant prompt.
4. Sends the prompt to Ollama (Mistral 7B) and streams the response.
"""

import os
import json
import logging
import httpx
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Generator

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
CHROMA_HOST  = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT  = int(os.getenv("CHROMA_PORT", "8001"))
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION   = "travel_docs"
LLM_MODEL    = "mistral:7b"
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K        = 3      # number of context chunks to retrieve
TIMEOUT      = 180.0  # seconds to wait for Ollama response

# ── Lazy singletons (initialized on first use, not at import time) ────────────
_embedder = None
_chroma_client = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model '%s' …", EMBED_MODEL)
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_chroma_client() -> chromadb.HttpClient:
    global _chroma_client
    if _chroma_client is None:
        logger.info("Connecting to ChromaDB at %s:%s …", CHROMA_HOST, CHROMA_PORT)
        _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return _chroma_client


def _get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection (raises if not yet ingested)."""
    return _get_chroma_client().get_collection(COLLECTION)


def retrieve_context(query: str, n: int = TOP_K) -> str:
    """
    Embed *query* and return the top-n most similar document chunks
    from ChromaDB, concatenated with a separator.
    """
    query_vec = _get_embedder().encode(query).tolist()
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n,
        include=["documents", "metadatas"],
    )
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []

    # Annotate each chunk with its source country for transparency
    annotated = []
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "unknown").capitalize()
        annotated.append(f"[Source: {source}]\n{doc}")

    return "\n\n---\n\n".join(annotated)


# ── Prompt template ──────────────────────────────────────────────────────────
_SYSTEM = """\
You are an expert travel planner. Using ONLY the travel guide context provided \
below, create a detailed, practical travel itinerary for the user.

Your response MUST include:
- A day-by-day plan with specific activities
- Entry costs for attractions (in € or local currency)
- Accommodation options and costs per night (budget / mid-range / luxury)
- Transportation tips and costs
- Food budget per day with examples
- Total estimated cost breakdown for the whole trip
- Practical tips to save money

Be specific, concise, and friendly. If the user's destination or budget does \
not match the context, say so and suggest alternatives from the context."""


def _build_prompt(context: str, question: str) -> str:
    return (
        f"{_SYSTEM}\n\n"
        f"=== TRAVEL GUIDE CONTEXT ===\n{context}\n"
        f"=== END CONTEXT ===\n\n"
        f"User request: {question}\n\n"
        "Travel plan:"
    )


# ── Generation helpers ───────────────────────────────────────────────────────

def generate_response(query: str) -> str:
    """Return the full LLM response as a single string (blocking)."""
    context = retrieve_context(query)
    prompt  = _build_prompt(context, query)

    resp = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def generate_response_stream(query: str) -> Generator[str, None, None]:
    """
    Yield individual tokens from the LLM as they are produced.
    Suitable for Server-Sent Events or StreamingResponse in FastAPI.
    """
    context = retrieve_context(query)
    prompt  = _build_prompt(context, query)

    with httpx.stream(
        "POST",
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data  = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                continue
