/**
 * scripts/ingest.js
 * Φορτώνει όλα τα .txt αρχεία από το corpus/ στο pgvector.
 *
 * Χρήση:
 *   node scripts/ingest.js            # φορτώνει όλα
 *   node scripts/ingest.js --clear    # διαγράφει πρώτα παλιά chunks
 */

import { readFileSync, readdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT      = join(__dirname, "..");

// Env vars φορτώνονται με --env-file=api/.env (βλ. package.json script)

const { initDB, insertChunk, clearDocuments } = await import("../api/db.js");
const { embed } = await import("../api/embed.js");

const CORPUS_DIR  = join(ROOT, "corpus");
const CHUNK_SIZE  = 400;
const OVERLAP     = 60;
const shouldClear = process.argv.includes("--clear");

function chunkText(text, size = CHUNK_SIZE, overlap = OVERLAP) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    chunks.push(text.slice(start, start + size));
    start += size - overlap;
  }
  return chunks;
}

async function ingest() {
  console.log("Connecting to pgvector...");
  await initDB();

  if (shouldClear) {
    console.log("Clearing existing documents...");
    await clearDocuments();
  }

  const files = readdirSync(CORPUS_DIR).filter((f) => f.endsWith(".txt"));
  console.log(`Found ${files.length} corpus files: ${files.join(", ")}\n`);

  let totalChunks = 0;

  for (const file of files) {
    const source = file.replace(".txt", "");
    const text   = readFileSync(join(CORPUS_DIR, file), "utf-8");
    const chunks = chunkText(text);

    console.log(`[${source}] ${chunks.length} chunks...`);

    for (let i = 0; i < chunks.length; i++) {
      process.stdout.write(`  chunk ${i + 1}/${chunks.length}\r`);
      const vector = await embed(chunks[i]);
      await insertChunk(chunks[i], source, vector);
      totalChunks++;
    }

    console.log(`  [${source}] done (${chunks.length} chunks stored)`);
  }

  console.log(`\nIngest complete! Total chunks stored: ${totalChunks}`);
  process.exit(0);
}

ingest().catch((err) => {
  console.error("Ingest failed:", err);
  process.exit(1);
});
