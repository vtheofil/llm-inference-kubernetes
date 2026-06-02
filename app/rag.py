"""
rag.py — RAG (Retrieval-Augmented Generation) pipeline.

1. Embeds the user query with multilingual-e5-small (E5 requires "query:" prefix).
   For multi-turn chats, recent user messages are combined with the current query
   so that retrieval is context-aware (history-aware retrieval).
2. Retrieves the top-k most relevant chunks from ChromaDB.
3. Builds a structured travel-assistant prompt (with current date + chat history).
4. Sends the prompt to Ollama (Mistral 7B) and streams the response.

E5 prefix convention:
  - Queries   → "query: <text>"
  - Documents → "passage: <text>"  (applied at ingest time in ingest.py)
"""

import os
import json
import logging
import httpx
import chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import Generator

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
CHROMA_HOST  = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT  = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION   = "travel_docs"
LLM_MODEL    = "mistral:7b"
# multilingual-e5-small: trained for retrieval, cross-lingual (Greek ↔ English),
# requires "query: " / "passage: " prefixes for best results (~470 MB, CPU-fast)
EMBED_MODEL  = "intfloat/multilingual-e5-small"
TOP_K        = 5      # number of context chunks to retrieve
TIMEOUT      = 180.0  # seconds to wait for Ollama response
# Max previous user turns to include in the combined retrieval query
HISTORY_RETRIEVAL_TURNS = 2
# Max previous turns to show in the generation prompt (truncated to save tokens)
HISTORY_PROMPT_TURNS    = 2
HISTORY_ASSISTANT_CHARS = 150  # truncate long assistant replies in the prompt

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


def _combined_retrieval_query(message: str, history: list) -> str:
    """
    Combine the current user message with recent user turns from history.
    This makes retrieval context-aware: if a user asks "what about food costs?"
    after asking about Rome, the retrieval query becomes "Rome 5 days 700€ food costs"
    instead of just "what about food costs?".

    history format: [{"user": "...", "assistant": "..."}, ...]
    Zero CPU overhead — only affects the embedding call (sentence-transformers, fast).
    """
    if not history:
        return message
    recent = [
        turn.get("user", "")
        for turn in history[-HISTORY_RETRIEVAL_TURNS:]
        if turn.get("user")
    ]
    combined = " ".join(recent + [message])
    logger.debug("Combined retrieval query: %s", combined[:200])
    return combined


def retrieve_context(message: str, history: list | None = None, n: int = TOP_K) -> str:
    """
    Embed the (history-aware) query with E5 "query: " prefix and return the
    top-n most similar document chunks from ChromaDB.
    """
    combined = _combined_retrieval_query(message, history or [])
    # E5 requires the "query: " prefix at retrieval time
    query_vec = _get_embedder().encode(f"query: {combined}").tolist()
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n,
        include=["documents", "metadatas"],
    )
    docs  = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0]  if results["metadatas"]  else []

    # Annotate each chunk with its source country for transparency
    annotated = []
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "unknown").capitalize()
        annotated.append(f"[Source: {source}]\n{doc}")

    return "\n\n---\n\n".join(annotated)


# ── Prompt template ──────────────────────────────────────────────────────────
_SYSTEM = """\
You are an expert travel planner. Using ONLY the travel guide context \
provided below, create a practical travel itinerary.

RULES:
1. Respond in the SAME LANGUAGE as the user's message.
2. STRICTLY respect the user's budget — never suggest options that exceed it.
3. Keep your response under 600 words — be specific, not verbose.
4. If the destination or budget is not covered by the context, say so clearly \
   and suggest the closest available alternative from the context.

Your response MUST include:
- Day-by-day plan (Day 1, Day 2, ...) with specific activities and entry costs (€)
- Accommodation: budget option with price per night
- Transport: how to get there + local transport costs
- Food budget per day (realistic estimate)
- Total cost breakdown at the end
- 2-3 money-saving tips

Use ONLY information from the context. Do not invent prices or places."""


def _build_history_block(history: list) -> str:
    """
    Build a short conversation-history block for the generation prompt.
    Only the last HISTORY_PROMPT_TURNS turns are included; assistant replies
    are truncated to HISTORY_ASSISTANT_CHARS to keep the prompt short (CPU-friendly).
    Returns an empty string when there is no history.
    """
    if not history:
        return ""
    turns = []
    for turn in history[-HISTORY_PROMPT_TURNS:]:
        user_msg = turn.get("user", "").strip()
        asst_msg = turn.get("assistant", "").strip()
        if user_msg:
            turns.append(f"User: {user_msg}")
        if asst_msg:
            snippet = asst_msg[:HISTORY_ASSISTANT_CHARS]
            if len(asst_msg) > HISTORY_ASSISTANT_CHARS:
                snippet += "…"
            turns.append(f"Assistant: {snippet}")
    if not turns:
        return ""
    return "=== CONVERSATION HISTORY ===\n" + "\n".join(turns) + "\n=== END HISTORY ===\n\n"


def _build_prompt(context: str, question: str, history: list | None = None) -> str:
    today = datetime.now().strftime("%B %Y")          # e.g. "June 2026"
    history_block = _build_history_block(history or [])
    return (
        f"{_SYSTEM}\n\n"
        f"Current date: {today}\n\n"
        f"{history_block}"
        f"=== TRAVEL GUIDE CONTEXT ===\n{context}\n"
        f"=== END CONTEXT ===\n\n"
        f"User request: {question}\n\n"
        "Travel plan:"
    )


# ── Generation helpers ───────────────────────────────────────────────────────

def generate_response(query: str, history: list | None = None) -> str:
    """Return the full LLM response as a single string (blocking)."""
    context = retrieve_context(query, history)
    prompt  = _build_prompt(context, query, history)

    resp = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def generate_response_stream(query: str, history: list | None = None) -> Generator[str, None, None]:
    """
    Yield individual tokens from the LLM as they are produced.
    Suitable for Server-Sent Events or StreamingResponse in FastAPI.
    """
    context = retrieve_context(query, history)
    prompt  = _build_prompt(context, query, history)

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
