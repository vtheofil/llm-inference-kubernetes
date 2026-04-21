"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const BRAND = "Thesis Chat";
const SUB = "Streaming LLM τώρα • RAG σε λίγο";

export default function Page() {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Γεια! Είμαι το demo chat για την πτυχιακή σου.\nΓράψε κάτι και θα απαντήσω (streaming).",
    },
  ]);

  const listRef = useRef(null);
  const esRef = useRef(null);

  const canSend = useMemo(() => input.trim().length > 0 && !busy, [input, busy]);

  useEffect(() => {
    return () => {
      try {
        esRef.current?.close();
      } catch {}
    };
  }, []);

  function scrollToBottom() {
    const el = listRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }

  function clearChat() {
    try {
      esRef.current?.close();
    } catch {}
    setBusy(false);
    setMessages([
      {
        role: "assistant",
        content:
          "Το chat καθάρισε ✅\nΓράψε κάτι για να ξεκινήσουμε πάλι.",
      },
    ]);
  }

  async function send() {
    if (!canSend) return;

    const userMsg = input.trim();
    setInput("");
    setBusy(true);

    try {
      esRef.current?.close();
    } catch {}

    const base = [...messages, { role: "user", content: userMsg }];
    const assistantIndex = base.length;
    const next = [...base, { role: "assistant", content: "", streaming: true }];

    setMessages(next);
    queueMicrotask(scrollToBottom);

    const historyPayload = base.map((m) => ({ role: m.role, content: m.content }));

    const qs = new URLSearchParams();
    qs.set("message", userMsg);
    qs.set("history", JSON.stringify(historyPayload));

    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
    const es = new EventSource(`${baseUrl}/chat/stream?${qs.toString()}`);
    esRef.current = es;

    let gotAny = false;

    es.addEventListener("delta", (ev) => {
      gotAny = true;
      const data = JSON.parse(ev.data || "{}");
      const delta = data.delta || "";
      if (!delta) return;

      setMessages((prev) => {
        const copy = [...prev];
        const cur = copy[assistantIndex] || { role: "assistant", content: "" };
        copy[assistantIndex] = {
          ...cur,
          content: (cur.content || "") + delta,
          streaming: true,
        };
        return copy;
      });

      queueMicrotask(scrollToBottom);
    });

    es.addEventListener("done", () => {
      setMessages((prev) => {
        const copy = [...prev];
        const cur = copy[assistantIndex] || { role: "assistant", content: "" };
        copy[assistantIndex] = { ...cur, streaming: false };
        return copy;
      });
      setBusy(false);
      try {
        es.close();
      } catch {}
      queueMicrotask(scrollToBottom);
    });

    es.addEventListener("error", () => {
      setMessages((prev) => {
        const copy = [...prev];
        const msg = gotAny
          ? "Το stream σταμάτησε απρόσμενα. Δοκίμασε ξανά."
          : "Σφάλμα στο stream. Δες το terminal του API για logs.";
        copy[assistantIndex] = { role: "assistant", content: msg, streaming: false };
        return copy;
      });
      setBusy(false);
      try {
        es.close();
      } catch {}
    });
  }

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 20,
          borderBottom: "1px solid var(--border)",
          background: "linear-gradient(180deg, rgba(0,0,0,0.65), rgba(0,0,0,0.25))",
          backdropFilter: "blur(14px)",
        }}
      >
        <div
          style={{
            maxWidth: 1100,
            margin: "0 auto",
            padding: "16px 16px",
            display: "flex",
            alignItems: "center",
            gap: 14,
          }}
        >
          <div
            style={{
              width: 38,
              height: 38,
              borderRadius: 14,
              border: "1px solid var(--border)",
              background: "rgba(255,255,255,0.06)",
              display: "grid",
              placeItems: "center",
              boxShadow: "0 10px 35px rgba(0,0,0,0.35)",
            }}
            title="LLM"
          >
            <span style={{ fontWeight: 900, letterSpacing: 0.2 }}>AI</span>
          </div>

          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
              <div style={{ fontWeight: 900, letterSpacing: 0.2 }}>{BRAND}</div>
              <span
                style={{
                  fontSize: 12,
                  color: "var(--muted)",
                  border: "1px solid var(--border)",
                  background: "rgba(255,255,255,0.04)",
                  padding: "4px 8px",
                  borderRadius: 999,
                }}
              >
                {busy ? "γράφει…" : "έτοιμο"}
              </span>
            </div>
            <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {SUB}
            </div>
          </div>

          <button
            onClick={clearChat}
            disabled={busy}
            style={{
              border: "1px solid var(--border)",
              background: "rgba(255,255,255,0.05)",
              color: "var(--text)",
              padding: "10px 12px",
              borderRadius: 14,
              cursor: busy ? "not-allowed" : "pointer",
              opacity: busy ? 0.6 : 1,
              fontWeight: 700,
            }}
            title="Clear chat"
          >
            Clear
          </button>
        </div>
      </header>

      {/* Main */}
      <main style={{ flex: 1, padding: 16 }}>
        <div
          style={{
            maxWidth: 1100,
            margin: "0 auto",
            display: "grid",
            gridTemplateColumns: "1fr",
            gap: 14,
          }}
        >
          {/* Chat panel */}
          <div
            style={{
              border: "1px solid var(--border)",
              background: "var(--panel)",
              borderRadius: 22,
              boxShadow: "var(--shadow)",
              overflow: "hidden",
            }}
          >
            <div
              ref={listRef}
              style={{
                height: "calc(100vh - 240px)",
                overflow: "auto",
                padding: 18,
              }}
            >
              {messages.map((m, idx) => (
                <MessageBubble key={idx} role={m.role} content={m.content} streaming={!!m.streaming} />
              ))}
            </div>

            {/* Composer */}
            <div
              style={{
                borderTop: "1px solid var(--border)",
                padding: 12,
                background: "rgba(0,0,0,0.25)",
              }}
            >
              <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
                <div style={{ flex: 1 }}>
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        send();
                      }
                    }}
                    rows={2}
                    placeholder="Γράψε μήνυμα… (Enter για send • Shift+Enter για νέα γραμμή)"
                    style={{
                      width: "100%",
                      resize: "none",
                      padding: "12px 12px",
                      borderRadius: 16,
                      border: "1px solid var(--border)",
                      background: "rgba(0,0,0,0.35)",
                      color: "var(--text)",
                      outline: "none",
                      lineHeight: 1.35,
                    }}
                  />
                  <div style={{ fontSize: 12, color: "var(--muted2)", marginTop: 8 }}>
                    Tip: δοκίμασε “γράψε μου πλάνο πτυχιακής με RAG + autoscaling”.
                  </div>
                </div>

                <button
                  onClick={send}
                  disabled={!canSend}
                  style={{
                    border: "1px solid var(--border)",
                    background: canSend ? "var(--green)" : "rgba(255,255,255,0.04)",
                    color: "var(--text)",
                    padding: "12px 16px",
                    borderRadius: 16,
                    cursor: canSend ? "pointer" : "not-allowed",
                    fontWeight: 900,
                    minWidth: 110,
                  }}
                >
                  {busy ? "…" : "Send"}
                </button>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div style={{ textAlign: "center", color: "var(--muted2)", fontSize: 12 }}>
            Local dev • Next.js UI • Express API • Streaming • (RAG next)
          </div>
        </div>
      </main>

      <style jsx global>{`
        @keyframes blink {
          0% { opacity: 1; }
          50% { opacity: 0; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}

function MessageBubble({ role, content, streaming }) {
  const isUser = role === "user";

  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        padding: "8px 0",
      }}
    >
      <div
        style={{
          maxWidth: "70%",
          minWidth: isUser ? 120 : 220,
          borderRadius: 18,
          border: "1px solid var(--border)",
          background: isUser ? "var(--blue)" : "var(--panel-strong)",
          padding: "12px 14px",
          whiteSpace: "pre-wrap",
          lineHeight: 1.5,
          boxShadow: isUser
            ? "0 14px 50px rgba(59,130,246,0.12)"
            : "0 14px 50px rgba(0,0,0,0.28)",
        }}
      >
        <div
          style={{
            fontSize: 12,
            color: "var(--muted)",
            marginBottom: 6,
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span>{isUser ? "You" : "Assistant"}</span>
          {isUser ? null : (
            <span
              style={{
                fontSize: 11,
                padding: "3px 8px",
                borderRadius: 999,
                border: "1px solid var(--border)",
                background: "rgba(255,255,255,0.04)",
                color: "var(--muted2)",
              }}
            >
              streamed
            </span>
          )}
        </div>

        <div>
          {content}
          {streaming ? (
            <span
              style={{
                display: "inline-block",
                width: 8,
                height: 14,
                marginLeft: 6,
                borderRadius: 4,
                background: "rgba(255,255,255,0.6)",
                animation: "blink 1s steps(2, start) infinite",
                verticalAlign: "text-bottom",
              }}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}