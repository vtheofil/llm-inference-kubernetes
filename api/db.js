import pg from "pg";

const { Pool } = pg;

const pool = new Pool({
  host:     process.env.PG_HOST     || "localhost",
  port:     Number(process.env.PG_PORT || 5433),
  database: process.env.PG_DB       || "ragdb",
  user:     process.env.PG_USER     || "raguser",
  password: process.env.PG_PASSWORD || "ragpass",
});

// Δημιουργία πίνακα + index αν δεν υπάρχουν
export async function initDB() {
  await pool.query(`CREATE EXTENSION IF NOT EXISTS vector`);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS documents (
      id        SERIAL PRIMARY KEY,
      content   TEXT    NOT NULL,
      source    TEXT,
      embedding vector(768)
    )
  `);
  // IVFFlat index για γρήγορη cosine-similarity αναζήτηση
  await pool.query(`
    CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50)
  `);
  console.log("DB ready (pgvector)");
}

// Αποθήκευση ενός chunk
export async function insertChunk(content, source, embedding) {
  await pool.query(
    `INSERT INTO documents (content, source, embedding) VALUES ($1, $2, $3)`,
    [content, source, JSON.stringify(embedding)]
  );
}

// Εύρεση των k πιο σχετικών chunks για ένα query embedding
export async function searchSimilar(embedding, k = 4) {
  const result = await pool.query(
    `SELECT content, source, 1 - (embedding <=> $1) AS similarity
     FROM documents
     ORDER BY embedding <=> $1
     LIMIT $2`,
    [JSON.stringify(embedding), k]
  );
  return result.rows; // [{ content, source, similarity }]
}

// Διαγραφή όλων (για re-ingest)
export async function clearDocuments() {
  await pool.query(`DELETE FROM documents`);
}

export { pool };
