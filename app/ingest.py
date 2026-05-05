"""
ingest.py — Load travel documents into ChromaDB.

Reads all .txt files from the data/ directory, splits them into overlapping
chunks, embeds them with a local sentence-transformers model, and stores
everything in ChromaDB under the "travel_docs" collection.

Run once before starting the backend:
    python -m app.ingest
"""

import os
import glob
import time
import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration (override via environment variables) ──────────────────────
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
DATA_DIR    = os.getenv("DATA_DIR", "./data")
COLLECTION  = "travel_docs"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # multilingual, ~420 MB
CHUNK_SIZE  = 500   # characters per chunk
CHUNK_OVERLAP = 50  # overlap between consecutive chunks


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of CHUNK_SIZE characters."""
    chunks = []
    start  = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c]  # drop empty chunks


def collection_has_data(client: chromadb.HttpClient) -> bool:
    """Return True if the travel_docs collection already has documents."""
    try:
        col = client.get_collection(COLLECTION)
        return col.count() > 0
    except Exception:
        return False


def ingest(force: bool = False) -> None:
    """Main ingestion routine. Skips if data already exists (unless force=True)."""
    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT} …")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    if not force and collection_has_data(client):
        count = client.get_collection(COLLECTION).count()
        print(f"✓ Collection '{COLLECTION}' already has {count} chunks — skipping ingestion.")
        return

    # (Re)create the collection
    try:
        client.delete_collection(COLLECTION)
        print(f"  Deleted existing collection '{COLLECTION}'.")
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for embeddings
    )

    # Load embedding model
    print(f"Loading embedding model '{EMBED_MODEL}' …")
    embedder = SentenceTransformer(EMBED_MODEL)

    # Read and chunk all .txt files
    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{DATA_DIR}'.")

    ids, documents, embeddings, metadatas = [], [], [], []

    for filepath in txt_files:
        country = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"  {country}: {len(chunks)} chunks from {len(text):,} chars")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{country}_{i:04d}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({"source": country, "chunk_index": i})

    # Compute embeddings in one batch (faster than one-by-one)
    print(f"Embedding {len(documents)} chunks (this may take a minute) …")
    t0 = time.time()
    vecs = embedder.encode(documents, show_progress_bar=True, batch_size=32)
    embeddings = vecs.tolist()
    print(f"  Embedding done in {time.time() - t0:.1f}s")

    # Upload to ChromaDB in batches of 200
    BATCH = 200
    for start in range(0, len(ids), BATCH):
        sl = slice(start, start + BATCH)
        collection.add(
            ids=ids[sl],
            documents=documents[sl],
            embeddings=embeddings[sl],
            metadatas=metadatas[sl],
        )
    print(f"[OK] Ingested {len(ids)} chunks into collection '{COLLECTION}'.")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    ingest(force=force)
