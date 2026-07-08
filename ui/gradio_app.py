"""
gradio_app.py — Gradio chat UI for the Atlas Systems Knowledge Assistant.

Two-view design:
  • Chat view (default): conversation with the assistant
  • Sources view: shows which knowledge-base chunks were retrieved for the
    LAST question (citation + similarity distance + snippet)

The user toggles between the two with a button — the source panel exposes
the underlying RAG transparency so reviewers can see exactly which
enterprise documents informed each answer.

Backend endpoint used:
  POST /chat  → returns {"response": "...", "sources": [...]} so we get
                the assistant text AND the retrieved chunks in one round-trip
                (no streaming, slightly higher perceived latency, but the
                sources panel is the whole point of this UI iteration).

Run locally:
    python ui/gradio_app.py
"""

from __future__ import annotations

import os
import httpx

# ── Aggressive monkey-patch for gradio_client schema bugs ─────────────────────
# Gradio 4.44.1 + gradio_client 1.3.0 both crash during launch() while
# generating the OpenAPI schema for the API endpoint. Different versions
# fail at different lines with either:
#   TypeError: argument of type 'bool' is not iterable
#   gradio_client.utils.APIInfoParseError: Cannot parse schema True
# both rooted in newer JSON-Schema specs emitting boolean values where
# gradio_client expects dict (e.g. `additionalProperties: True`).
#
# The narrow patch (overriding only `get_type`) was insufficient because
# the parser also calls `_json_schema_to_python_type` recursively and
# encounters bools at deeper levels. Aggressive fix: short-circuit the
# top-level parser to return "Any" for bool inputs, leaving dict schemas
# untouched.
#
# Cost of this patch: the auto-generated API documentation (/docs)
# becomes less precise on types — completely fine for us because we set
# show_api=False on launch() and do not expose the API to external
# clients (Gradio is a UI in front of our own FastAPI backend).
import sys
print("[gradio_app] Applying gradio_client schema patches…", flush=True)
try:
    import gradio_client.utils as _gc_utils

    _orig_get_type = _gc_utils.get_type
    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _orig_get_type(schema)
    _gc_utils.get_type = _patched_get_type

    _orig_json_to_py = _gc_utils._json_schema_to_python_type
    def _patched_json_to_py(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        try:
            return _orig_json_to_py(schema, defs)
        except Exception as exc:
            print(f"[gradio_app] schema parse error swallowed: {exc!r}", flush=True)
            return "Any"
    _gc_utils._json_schema_to_python_type = _patched_json_to_py

    print("[gradio_app] gradio_client patches applied successfully", flush=True)
except Exception as exc:
    print(f"[gradio_app] WARNING: could not apply gradio_client patches: {exc!r}", flush=True)

import gradio as gr

# Backend URL — override via environment variable
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CHAT_URL    = f"{BACKEND_URL}/chat"
HEALTH_URL  = f"{BACKEND_URL}/healthz"


# ── Backend helpers ─────────────────────────────────────────────────────────

def _messages_to_backend_history(history: list) -> list[dict]:
    """
    Convert Gradio's "messages"-style history (list of {role, content}) into
    the [{user, assistant}, ...] turn-pair shape the backend expects.
    """
    formatted: list[dict] = []
    pending_user: str | None = None
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            formatted.append({"user": pending_user, "assistant": content})
            pending_user = None
    return formatted


def ask_backend(message: str, history: list) -> tuple[str, list[dict]]:
    """
    Send the user's message + history to the backend's blocking /chat
    endpoint. Returns (assistant_reply_text, list_of_source_records).
    """
    timeout = httpx.Timeout(connect=10.0, read=1200.0, write=10.0, pool=10.0)
    try:
        r = httpx.post(
            CHAT_URL,
            json={
                "message": message,
                "history": _messages_to_backend_history(history),
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        # The backend's main.py wraps the dict in a ChatResponse model that
        # exposes `.response` and `.latency_ms`; sources live on the raw RAG
        # call. Tolerate both shapes for forwards compatibility.
        if isinstance(data, dict):
            reply = data.get("response") or data.get("answer") or "(empty response)"
            sources = data.get("sources") or []
        else:
            reply, sources = str(data), []
        return reply, sources
    except httpx.ConnectError:
        return (
            f"❌ Cannot connect to backend at {BACKEND_URL}. "
            "Is the FastAPI server running?"
        ), []
    except Exception as exc:
        return f"❌ Error ({type(exc).__name__}): {exc}", []


# ── Source-panel formatting ─────────────────────────────────────────────────

def format_sources(sources: list[dict]) -> str:
    """Render the retrieved-chunks list as Markdown for the Sources view."""
    if not sources:
        return (
            "*No sources yet.* Ask a question first and the retrieved chunks "
            "will appear here."
        )
    parts: list[str] = ["## Retrieved Context\n"]
    parts.append("_Top-k semantically-relevant chunks from the knowledge base._\n")
    for i, s in enumerate(sources, start=1):
        citation = s.get("citation", "Unknown")
        source_path = s.get("source_path", "")
        distance = s.get("distance")
        snippet = (s.get("snippet") or "").strip()
        if isinstance(distance, (int, float)):
            similarity_line = f"_cosine distance: {distance:.4f}_"
        else:
            similarity_line = ""
        parts.append(f"### {i} · {citation}")
        if source_path:
            parts.append(f"`data/{source_path}`")
        if similarity_line:
            parts.append(similarity_line)
        parts.append("\n```\n" + snippet + "\n```\n")
    return "\n".join(parts)


# ── Examples shown under the input box ──────────────────────────────────────

EXAMPLES = [
    "Who is the CEO of Atlas Systems?",
    "When was Atlas Systems founded?",
    "How many days of annual leave does an employee get after 5 years of service?",
    "How do I reset my VPN password?",
    "Who leads the Engineering department?",
    "What is the annual training budget per employee?",
    "Where is the Atlas Systems Athens HQ located?",
    "What cloud providers does Atlas Systems partner with?",
]


# ── Build Gradio UI ─────────────────────────────────────────────────────────

with gr.Blocks(
    title="Atlas Systems · AI Knowledge Assistant",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"),
    fill_width=True,
    css="""
        .gradio-container { max-width: 100% !important; padding: 0 24px !important; }
        #chatbot { min-height: 580px !important; }
        .header-box { text-align: center; padding: 16px 0 4px; }
        footer { display: none !important; }
    """,
) as demo:
    # Holds the source records returned by the most recent /chat call.
    sources_state = gr.State(value=[])

    gr.HTML(
        """
        <div class="header-box">
            <h1><span class="brand-mark">Atlas Systems</span>
                &nbsp;·&nbsp; AI Knowledge Assistant</h1>
            <p>Retrieval-augmented answers grounded in your enterprise knowledge base</p>
        </div>
        """
    )

    # ── Chat view (visible by default) ──────────────────────────────────────
    with gr.Column(visible=True) as chat_view:
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            type="messages",
            height=540,
            show_label=False,
            avatar_images=(None, None),
            bubble_full_width=False,
        )
        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Ask anything about Atlas Systems…",
                scale=8,
                container=False,
                show_label=False,
            )
            send_btn = gr.Button("Send  ➤", scale=1, variant="primary")
        with gr.Row():
            show_sources_btn = gr.Button("View Sources", scale=1, variant="secondary")
            clear_btn = gr.Button("Clear", scale=1, variant="secondary")

        gr.Examples(
            examples=[[e] for e in EXAMPLES],
            inputs=[msg_box],
            label="Try an example",
        )

    # ── Sources view (hidden by default) ────────────────────────────────────
    with gr.Column(visible=False) as sources_view:
        back_btn = gr.Button("← Back to Chat", variant="primary")
        sources_md = gr.Markdown(
            value="*No sources yet.* Ask a question first and the retrieved "
                  "chunks will appear here."
        )

    # ── Event handlers ──────────────────────────────────────────────────────

    # The chat round-trip is split into TWO steps (à la Ed Donner / week5)
    # so the user's message appears in the chat IMMEDIATELY when they hit
    # Send, instead of only after the LLM finishes generating.
    #
    # Step 1 (`on_user_submit`)   — append the user's message and clear the
    #                               textbox. Gradio's chatbot then shows the
    #                               default "in progress" indicator on its
    #                               own; we don't need a manual placeholder.
    # Step 2 (`on_assistant_reply`) — chained via `.then()`, sends the chat
    #                               (including the new user turn) to the
    #                               backend and appends the assistant reply.
    #
    # KEY POINT — we pass the chatbot history in Gradio's native
    # {"role": ..., "content": ...} format directly to `ask_backend`,
    # which calls `_messages_to_backend_history` ONCE to convert it for the
    # backend. A previous version did the conversion in both functions and
    # ended up sending an empty history (root cause of the broken
    # follow-up retrieval).

    def on_user_submit(message: str, history: list, _sources: list):
        """Step 1 — append the user message and clear the textbox."""
        message = (message or "").strip()
        if not message:
            return "", history, _sources or []
        return "", (history or []) + [
            {"role": "user", "content": message},
        ], _sources or []

    def on_assistant_reply(history: list, _sources: list):
        """Step 2 — send the latest user message + the prior history to the
        backend, then append the assistant reply."""
        if not history:
            return history, _sources or []
        # The latest user message is the LAST entry (just appended in step 1).
        last_message = history[-1].get("content", "")
        prior = history[:-1]   # Gradio messages format; conversion happens inside ask_backend
        reply, sources = ask_backend(last_message, prior)
        new_history = history + [{"role": "assistant", "content": reply}]
        return new_history, sources

    def on_show_sources(_sources: list):
        """Toggle to the sources view, rendered from the latest state."""
        return (
            gr.update(visible=False),       # chat_view
            gr.update(visible=True),        # sources_view
            format_sources(_sources or []),  # sources_md value
        )

    def on_back_to_chat():
        return (
            gr.update(visible=True),   # chat_view
            gr.update(visible=False),  # sources_view
        )

    def on_clear():
        # Resets both the chat transcript and the retrieved-sources buffer.
        return [], []

    # Wire up the events — two-step submit so the user message shows up
    # before the LLM finishes.
    msg_box.submit(
        on_user_submit,
        inputs=[msg_box, chatbot, sources_state],
        outputs=[msg_box, chatbot, sources_state],
    ).then(
        on_assistant_reply,
        inputs=[chatbot, sources_state],
        outputs=[chatbot, sources_state],
    )
    send_btn.click(
        on_user_submit,
        inputs=[msg_box, chatbot, sources_state],
        outputs=[msg_box, chatbot, sources_state],
    ).then(
        on_assistant_reply,
        inputs=[chatbot, sources_state],
        outputs=[chatbot, sources_state],
    )
    show_sources_btn.click(
        on_show_sources,
        inputs=[sources_state],
        outputs=[chat_view, sources_view, sources_md],
    )
    back_btn.click(
        on_back_to_chat,
        inputs=None,
        outputs=[chat_view, sources_view],
    )
    clear_btn.click(
        on_clear,
        inputs=None,
        outputs=[chatbot, sources_state],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False,
        show_api=False,
    )
