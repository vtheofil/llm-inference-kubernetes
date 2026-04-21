import "dotenv/config";
import express from "express";
import OpenAI from "openai";
import { initDB, insertChunk, searchSimilar, clearDocuments } from "./db.js";
import { embed } from "./embed.js";

const app = express();

// CORS για local dev (web:3000 -> api:8080)
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", process.env.CORS_ORIGIN || "http://localhost:3000");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Allow-Credentials", "true");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});
app.use(express.json({ limit: "2mb" }));

const {
  OPENAI_MODEL = "gpt-4o-mini",
  PORT         = "3001",
} = process.env;

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Ξεκίνα DB όταν ξεκινά ο server
initDB().catch((err) => {
  console.error("DB init failed:", err.message);
  process.exit(1);
});

// ─── Health ──────────────────────────────────────────────────────────────────
app.get("/health", (_req, res) => res.json({ ok: true }));

// ─── Ingest ──────────────────────────────────────────────────────────────────
// POST /ingest  { title: string, text: string, clear?: boolean }
// Κόβει το κείμενο σε chunks, παράγει embeddings, αποθηκεύει στο pgvector.
app.post("/ingest", async (req, res) => {
  const { title, text, clear = false } = req.body || {};
  if (!title || !text) {
    return res.status(400).json({ error: "title and text are required" });
  }

  try {
    if (clear) await clearDocuments();

    const chunks = chunkText(text, 400, 60);
    let stored = 0;

    for (const chunk of chunks) {
      const vector = await embed(chunk);
      await insertChunk(chunk, title, vector);
      stored++;
    }

    return res.json({ ok: true, source: title, chunks: stored });
  } catch (err) {
    console.error("Ingest error:", err);
    return res.status(500).json({ error: "ingest_failed", details: String(err) });
  }
});

// ─── Chat (streaming SSE) ─────────────────────────────────────────────────────
// GET /chat/stream?message=...&history=[...]
app.get("/chat/stream", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();

  const message    = (req.query.message  ?? "").toString();
  const historyRaw = (req.query.history  ?? "[]").toString();

  if (!message) {
    res.write(`event: error\ndata: ${JSON.stringify({ error: "message required" })}\n\n`);
    return res.end();
  }

  let history = [];
  try {
    history = JSON.parse(historyRaw);
    if (!Array.isArray(history)) history = [];
  } catch { history = []; }

  const safeHistory = history.slice(-12).map((m) => ({
    role:    m.role === "assistant" ? "assistant" : "user",
    content: String(m.content ?? ""),
  }));

  // ── RAG: βρες σχετικά chunks ──
  let context = "";
  try {
    const queryVec = await embed(message);
    const chunks   = await searchSimilar(queryVec, 4);

    if (chunks.length > 0) {
      context = chunks
        .map((c) => `[${c.source}]\n${c.content}`)
        .join("\n\n---\n\n");
    }
  } catch (err) {
    // Αν το RAG αποτύχει, συνεχίζουμε χωρίς context (graceful degradation)
    console.warn("RAG retrieval failed, falling back to plain LLM:", err.message);
  }

  // ── Φτιάξε το system prompt ──
  const systemPrompt = context
    ? `You are an expert travel planner assistant. A user wants help planning a trip.
Use the travel information provided below to create a detailed, personalised travel plan.

When the user mentions a destination, number of days, and budget, respond with:
1. A day-by-day itinerary
2. Estimated costs per day (accommodation, food, transport, attractions)
3. Total estimated cost
4. Practical tips for that destination

If the budget is tight, suggest ways to save money. Always be specific with prices.

TRAVEL REFERENCE DATA:
${context}

Base your answer on the data above. If exact data is missing, give a reasonable estimate and note it.`
    : `You are an expert travel planner assistant. Help users plan trips by giving detailed itineraries, cost breakdowns, and practical tips. Ask the user for their destination, number of days, and budget if not provided.`;

  const messages = [
    { role: "system", content: systemPrompt },
    ...safeHistory,
    { role: "user", content: message },
  ];

  // Keep-alive ping κάθε 15s
  const ping = setInterval(() => {
    res.write(`event: ping\ndata: {}\n\n`);
  }, 15000);

  try {
    const stream = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages,
      stream: true,
    });

    req.on("close", () => {
      clearInterval(ping);
      try { stream.controller?.abort?.(); } catch {}
    });

    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content || "";
      if (delta) {
        res.write(`event: delta\ndata: ${JSON.stringify({ delta })}\n\n`);
      }
    }

    clearInterval(ping);
    res.write(`event: done\ndata: {}\n\n`);
    res.end();
  } catch (err) {
    clearInterval(ping);
    res.write(`event: error\ndata: ${JSON.stringify({ error: String(err) })}\n\n`);
    res.end();
  }
});

// ─── Chat (non-streaming, για load tests) ────────────────────────────────────
app.post("/chat", async (req, res) => {
  try {
    const { message, history = [] } = req.body || {};
    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "message (string) required" });
    }

    const safeHistory = Array.isArray(history) ? history.slice(-12) : [];

    // RAG
    let context = "";
    try {
      const queryVec = await embed(message);
      const chunks   = await searchSimilar(queryVec, 4);
      if (chunks.length > 0) {
        context = chunks.map((c) => `[${c.source}]\n${c.content}`).join("\n\n---\n\n");
      }
    } catch { /* graceful degradation */ }

    const systemPrompt = context
      ? `You are an expert travel planner assistant. Use the travel information below to create detailed, personalised travel plans with day-by-day itineraries and cost breakdowns.\n\nTRAVEL REFERENCE DATA:\n${context}`
      : `You are an expert travel planner assistant. Help users plan trips with detailed itineraries, cost breakdowns, and practical tips.`;

    const messages = [
      { role: "system", content: systemPrompt },
      ...safeHistory.map((m) => ({
        role:    m.role === "assistant" ? "assistant" : "user",
        content: String(m.content ?? ""),
      })),
      { role: "user", content: message },
    ];

    const response = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages,
    });

    return res.json({ answer: response.choices[0]?.message?.content ?? "" });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "chat_failed", details: String(err) });
  }
});

// ─── Helpers ──────────────────────────────────────────────────────────────────
function chunkText(text, size = 400, overlap = 60) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    chunks.push(text.slice(start, start + size));
    start += size - overlap;
  }
  return chunks;
}

app.listen(Number(PORT), () => {
  console.log(`API listening on http://localhost:${PORT}`);
});
