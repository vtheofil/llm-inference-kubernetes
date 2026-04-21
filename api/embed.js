// Παράγει embeddings μέσω Ollama (nomic-embed-text, 768 διαστάσεις)

const OLLAMA_BASE = process.env.OLLAMA_BASE_URL?.replace("/v1", "") || "http://localhost:11434";
const EMBED_MODEL = "nomic-embed-text";

/**
 * Επιστρέφει το embedding vector (float[]) για ένα κείμενο.
 * @param {string} text
 * @returns {Promise<number[]>}
 */
export async function embed(text) {
  const res = await fetch(`${OLLAMA_BASE}/api/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: EMBED_MODEL, prompt: text }),
  });

  if (!res.ok) {
    throw new Error(`Ollama embed error: ${res.status} ${await res.text()}`);
  }

  const data = await res.json();
  return data.embedding; // number[]
}
