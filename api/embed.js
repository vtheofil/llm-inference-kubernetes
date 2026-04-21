import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/**
 * Παράγει embeddings μέσω OpenAI (text-embedding-3-small, 1536 dimensions)
 * @param {string} text
 * @returns {Promise<number[]>}
 */
export async function embed(text) {
  const res = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return res.data[0].embedding;
}
