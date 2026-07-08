"""
rag.py — RAG (Retrieval-Augmented Generation) pipeline for Atlas Systems.

1. Embeds the user query with multilingual-e5-small (E5 requires "query:" prefix).
   For multi-turn chats, recent user messages may be combined with the current
   query so retrieval is context-aware (disabled by default — see
   HISTORY_RETRIEVAL_TURNS).
2. Detects query intent (project / department / employee / monthly_plan / faq /
   policy) from keywords and applies a ChromaDB `where={"type": ...}` filter
   so that, for example, a question about a specific project does not get
   polluted with monthly_plan chunks that merely mention the project name.
3. Retrieves the top-k most relevant chunks from ChromaDB along with their
   metadata (type, title, name, department, project, source_path).
4. Builds an enterprise knowledge-assistant prompt and sends it to Ollama
   (currently phi3:mini) and returns the response.

E5 prefix convention:
  - Queries   → "query: <text>"
  - Documents → "passage: <text>"  (applied at ingest time in ingest.py)

Bounded knowledge:
  The retriever is restricted to the Atlas Systems internal corpus
  (employees, projects, departments, policies, IT FAQs, meetings, monthly
  plans, job postings, and about pages). The system prompt instructs the
  LLM to refuse out-of-scope (general world-knowledge) questions.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Generator

import chromadb
import httpx
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
CHROMA_HOST  = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT  = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION   = "enterprise_docs"
LLM_MODEL    = "phi3:mini"
EMBED_MODEL  = "intfloat/multilingual-e5-small" 
TOP_K        = 3      # number of context chunks to retrieve
                       # Lowered from 5 → 3 when we switched to phi3:mini.
                       # The smaller model is easier to confuse with chunks
                       # from multiple countries — fewer, more relevant chunks
                       # yield more focused responses.
TIMEOUT      = 1200.0  # seconds to wait for Ollama response. Generous to
                       # accommodate cold-pod routing scenarios where Ollama
                       # must mmap phi3 into RAM (~60-90s) before generation.
# Max previous user turns to include in the combined retrieval query.
# Set to 0 to avoid prior-turn topics polluting the embedding for the
# current question. History is still shown to the LLM in the prompt
# (HISTORY_PROMPT_TURNS) so conversational continuity is preserved.
HISTORY_RETRIEVAL_TURNS = 1
# Max previous turns to show in the generation prompt (truncated to save tokens).
HISTORY_PROMPT_TURNS    = 2
HISTORY_ASSISTANT_CHARS = 100  # truncate long assistant replies in the prompt

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


# ── Citation formatting ─────────────────────────────────────────────────────

def _format_citation(meta: dict) -> str:
    """
    Build a human-readable citation label for an Atlas Systems document.

    Examples:
      type=employee,  name=Maria K.            → "Employee / Maria K."
      type=project,   name=Atlas               → "Project / Atlas"
      type=policy,    title=Annual Leave...    → "Policy / Annual Leave..."
      type=faq,       title=How do I reset...  → "FAQ / How do I reset..."
      type=department, department=Engineering  → "Department / Engineering"
      no type                                  → "Unknown source"
    """
    doc_type = (meta.get("type") or "").strip()
    title    = (meta.get("title") or "").strip()
    name     = (meta.get("name") or "").strip()
    dept     = (meta.get("department") or "").strip()
    project  = (meta.get("project") or "").strip()

    if not doc_type:
        return title or "Unknown source"

    label = title or name or project or dept or doc_type
    pretty_type = doc_type.replace("_", " ").title()
    return f"{pretty_type} / {label}"


# ── Retrieval ───────────────────────────────────────────────────────────────

# Follow-up question heuristics. When the user writes a short pronoun-style
# message ("tell me more", "what about it", "who works on it"), the
# embedding of THAT message alone lacks topical anchoring and the retriever
# returns irrelevant chunks. We detect these patterns and prepend the
# previous user message so the embedding regains its topic.
#
# We deliberately do NOT always combine with history (the default
# HISTORY_RETRIEVAL_TURNS = 0), because that re-introduces the opposite
# failure mode (topic drift on genuine topic switches).
_FOLLOWUP_PATTERNS = re.compile(
    r"^\s*("
    r"it|that|this|they|those|these|"
    r"tell me more|more (info|information|details)|"
    r"what about|how about|and (what|who|how)|"
    r"who works on|what is the (budget|status|deadline|team)|"
    r"who (manages|leads|is) (it|that|them)"
    r")\b",
    re.IGNORECASE,
)


def _is_followup(message: str) -> bool:
    """Heuristic: does this look like a follow-up to a previous question?"""
    if not message:
        return False
    m = message.strip()
    # Very short messages with a pronoun are almost always follow-ups.
    if len(m.split()) <= 4 and any(p in m.lower() for p in ("it", "that", "this", "more")):
        return True
    return bool(_FOLLOWUP_PATTERNS.match(m))


def _combined_retrieval_query(message: str, history: list) -> str:
    """
    Build the retrieval query.

    Default: just use the current message (HISTORY_RETRIEVAL_TURNS = 0).
    Override: if the current message looks like a follow-up
    ("tell me more", "what about it", …) prepend the previous user turn so
    the embedding regains topical anchoring.
    """
    if not history:
        return message

    if _is_followup(message):
        last_user = ""
        for turn in reversed(history):
            if turn.get("user"):
                last_user = turn["user"]
                break
        if last_user:
            combined = f"{last_user} {message}"
            # DEBUG (v2.0.6): upgraded to WARNING for diagnosis of the
            # Gradio→backend history-passing flow. Remove or downgrade once
            # the follow-up retrieval is confirmed to fire in production.
            logger.warning("FOLLOWUP DETECTED")
            logger.warning("LAST_USER=%s", last_user)
            logger.warning("COMBINED=%s", combined)
            return combined

    if HISTORY_RETRIEVAL_TURNS <= 0:
        return message

    recent = [
        turn.get("user", "")
        for turn in history[-HISTORY_RETRIEVAL_TURNS:]
        if turn.get("user")
    ]
    combined = " ".join(recent + [message])
    logger.debug("Combined retrieval query: %s", combined[:200])
    return combined


# ── Query intent classification (for metadata-filtered retrieval) ───────────

# Keyword cues per document type. If a query matches one of these patterns,
# we apply ChromaDB's `where={"type": "<doc_type>"}` filter so that, for
# example, a question about "Project Cygnus" doesn't get polluted with
# monthly_plan chunks that happen to mention Cygnus.
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "project":      ["project ", "project."],
    "department":   ["department"],
    "monthly_plan": ["monthly plan", "plan for", "plan in", "priorities for",
                     "priorities in", "focus projects"],
    "policy":       ["policy", "policies", "leave", "expense", "reimburs",
                     "conduct", "training budget", "remote work"],
    "faq":          ["how do i", "how do you", "how can i", "reset", "vpn",
                     "printer", "password", "install"],
}

# Months mentioned alongside a year → strong signal for monthly_plan intent.
_MONTHS = ["january", "february", "march", "april", "may", "june",
           "july", "august", "september", "october", "november", "december"]


def _detect_intent(message: str) -> str | None:
    """Return a document `type` (per ingest metadata) when the query strongly
    points at one category, else None for unfiltered semantic search.

    Examples:
      "Tell me about Project Cygnus"            → "project"
      "Tell me about the Marketing Department"  → "department"
      "What were the priorities in March 2026?" → "monthly_plan"
      "How do I reset my VPN password?"         → "faq"
      "What is the annual leave policy?"        → "policy"
      "Who is the CEO of Atlas Systems?"        → None  (no filter, semantic)
    """
    m = (message or "").lower()

    # Strongest cue: month name + year → monthly_plan
    if any(month in m for month in _MONTHS) and any(year in m for year in ("202", "2023", "2024", "2025", "2026")):
        return "monthly_plan"

    # Generic keyword scan, in priority order.
    for doc_type in ("project", "department", "policy", "faq", "monthly_plan"):
        for kw in _INTENT_KEYWORDS[doc_type]:
            if kw in m:
                return doc_type
    return None


def retrieve_with_sources(
    message: str,
    history: list | None = None,
    n: int = TOP_K,
) -> tuple[str, list[dict]]:
    """
    Run a (history-aware) similarity search and return:
      - context_text: chunks concatenated and annotated with citation labels,
                      ready to be injected into the LLM prompt
      - sources:      list of structured records suitable for the UI's
                      "Show Sources" panel — one entry per retrieved chunk

    Each `sources` record contains:
      { "citation": "Project / Cygnus",
        "type":     "project",
        "title":    "Project Cygnus",
        "department": "",
        "project":   "Cygnus",
        "source_path": "projects/proj_cygnus.md",
        "snippet":  "# Project Cygnus..." }

    Intent-aware filtering:
      For queries that strongly point to a single document type (e.g.
      "Tell me about Project Cygnus" → projects), we apply ChromaDB's
      `where={"type": ...}` filter to keep monthly_plan / meeting / job_posting
      chunks from polluting the answer. If no relevant chunks come back under
      the filter (rare), we fall back to a plain semantic search.
    """
    combined = _combined_retrieval_query(message, history or [])
    # DEBUG (v2.0.6): diagnose whether the Gradio UI is actually forwarding
    # the chat history to the backend (without history, follow-up retrieval
    # cannot fire). Remove these once the flow is confirmed.
    logger.warning("QUERY=%s", message)
    logger.warning("HISTORY=%s", history)
    logger.warning("FINAL_RETRIEVAL_QUERY=%s", combined)
    # E5 requires the "query: " prefix at retrieval time
    query_vec = _get_embedder().encode(f"query: {combined}").tolist()
    collection = _get_collection()

    intent = _detect_intent(combined)
    where_filter = {"type": intent} if intent else None
    if where_filter:
        logger.info("retrieval: intent=%s → where=%s", intent, where_filter)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n,
        include=["documents", "metadatas", "distances"],
        where=where_filter,
    )
    # Fallback: if the filter returned nothing useful, run an unfiltered query.
    if where_filter and (not results["documents"] or not results["documents"][0]):
        logger.warning("retrieval: filter returned 0 results, falling back to unfiltered")
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
    docs   = results["documents"][0]  if results["documents"]  else []
    metas  = results["metadatas"][0]  if results["metadatas"]  else []
    dists  = results["distances"][0]  if results.get("distances") else [None] * len(docs)

    annotated_blocks: list[str] = []
    sources: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        citation = _format_citation(meta)
        annotated_blocks.append(f"[Source: {citation}]\n{doc}")
        sources.append({
            "citation":    citation,
            "type":        meta.get("type", ""),
            "name":        meta.get("name", ""),
            "department":  meta.get("department", ""),
            "project":     meta.get("project", ""),
            "title":       meta.get("title", ""),
            "source_path": meta.get("source_path", ""),
            "distance":    dist,   # cosine distance — lower = more similar
            "snippet":     doc,
        })

    context_text = "\n\n---\n\n".join(annotated_blocks)
    return context_text, sources


def retrieve_context(message: str, history: list | None = None, n: int = TOP_K) -> str:
    """Backwards-compatible wrapper: returns only the annotated context string."""
    context, _ = retrieve_with_sources(message, history, n)
    return context


# ── Prompt template ──────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an internal knowledge assistant for Atlas Systems, a tech "
    "consulting company. You answer questions about employees, projects, "
    "departments, policies, meetings, and IT support — using ONLY the "
    "company knowledge base context provided below.\n"
    "\n"
    "LANGUAGE RULE:\n"
    "Respond ENTIRELY in English. The corpus is English.\n"
    "\n"
    "CORE RULES:\n"
    "1. Use ONLY information from the CONTEXT block below. Do NOT invent "
    "names, projects, budgets, dates, roles, policies, or procedures.\n"
    "2. If the answer is not in the CONTEXT, say clearly: 'I could not find "
    "that information in the knowledge base.'\n"
    "3. Be FACTUAL and CONCISE. Answer the question asked — do not pad with "
    "unrelated facts from other entities.\n"
    "4. When the CONTEXT contains multiple entities (e.g. multiple employees), "
    "use ONLY the one(s) directly relevant to the question.\n"
    "\n"
    "RESPONSE SHAPE:\n"
    "\n"
    "• Single-fact questions (Who is the PM of X? What is the budget of Y? "
    "How many leave days?): answer in **1 complete sentence** that restates "
    "the subject and gives the specific fact. Optionally add a brief second "
    "sentence with one supporting detail from the context. No preamble.\n"
    "  Examples:\n"
    "    Q: Who is the CEO of Atlas Systems?\n"
    "    A: The CEO of Atlas Systems is Maria Voulgari, who founded the "
    "company in 2014.\n"
    "    Q: What is the budget of Project Phoenix?\n"
    "    A: Project Phoenix has a total budget of €1,250,000.\n"
    "\n"
    "• List questions (Which projects does X work on? Who knows Python?): "
    "open with a brief intro sentence, then a short bullet list.\n"
    "\n"
    "• Procedural questions (How do I reset my VPN?): give the steps in "
    "order, briefly.\n"
    "\n"
    "• Out-of-scope (general world knowledge, unrelated topics): say briefly "
    "that this assistant only answers Atlas Systems internal questions.\n"
    "\n"
    "Keep total response under 200 words."
)


def _build_history_block(history: list) -> str:
    """Short conversation-history block for the generation prompt."""
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
    # Neutral "Answer:" suffix — the system prompt's RESPONSE SHAPE rules
    # decide whether the answer should be a single sentence, a bullet list,
    # or a numbered procedure, without a built-in bias from the suffix.
    return (
        f"{_SYSTEM}\n\n"
        f"Current date: {today}\n\n"
        f"{history_block}"
        f"=== ATLAS SYSTEMS KNOWLEDGE CONTEXT ===\n{context}\n"
        f"=== END CONTEXT ===\n\n"
        f"User question: {question}\n\n"
        "Answer:"
    )


# ── Generation helpers ───────────────────────────────────────────────────────

def generate_response(query: str, history: list | None = None) -> dict:
    """
    Blocking LLM call. Returns a dict:
        {"response": "<full text>", "sources": [...]}
    so the UI can render sources alongside the answer without re-running retrieval.
    """
    # DEBUG (v2.0.6): log the message + history at the very top of
    # generate_response so we can see exactly what the backend received from
    # the FastAPI handler (which in turn received it from the Gradio UI).
    # Remove once the follow-up flow is confirmed end-to-end.
    logger.warning("GENERATE_RESPONSE")
    logger.warning("QUERY=%s", query)
    logger.warning("HISTORY=%s", history)
    # Atlas Systems: no destination guard — every query goes through RAG.
    # The retrieval layer is itself the relevance filter.
    context, sources = retrieve_with_sources(query, history)
    prompt = _build_prompt(context, query, history)

    resp = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
              # num_predict caps the maximum number of tokens generated.
              # Without this, phi3 can run for 2500+ tokens on a "Plan 14 days"
              # request, and the corresponding KV-cache growth was traced to
              # the OOM observed at 12 GiB memory limit (sustained-load run
              # of ~16 minutes per pod). 1024 tokens is enough for a
              # 7-day-itinerary-with-budget answer and bounds worst-case
              # Ollama memory at ~10 GiB.
              "options": {"num_ctx": 4096, "num_predict": 512,
                          "temperature": 0.2}},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return {
        "response": resp.json().get("response", ""),
        "sources": sources,
    }


def generate_response_stream(
    query: str,
    history: list | None = None,
) -> Generator[str, None, None]:
    """
    Streaming LLM call. Yields tokens as they are produced.
    For the UI to know about sources, callers should first call
    `retrieve_with_sources()` (or use a non-stream endpoint that returns them).

    Same destination-guard logic as `generate_response()`: short-circuit
    to a fixed clarification message if the user has not named a
    supported destination.
    """
    context = retrieve_context(query, history)
    prompt = _build_prompt(context, query, history)

    with httpx.stream(
        "POST",
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": True,
              # See generate_response() above for the num_predict rationale.
              "options": {"num_ctx": 4096, "num_predict": 512,
                          "temperature": 0.2}},
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
