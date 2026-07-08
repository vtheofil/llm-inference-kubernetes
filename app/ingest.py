"""
ingest.py — Load Atlas Systems enterprise documents into ChromaDB.

Walks the data/ tree, parses each markdown file's YAML frontmatter,
chunks the body semantically, embeds each chunk with multilingual-e5-small,
and stores everything in ChromaDB under the "enterprise_docs" collection.
"""

from __future__ import annotations

import glob
import os
import re
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
DATA_DIR = os.getenv("DATA_DIR", "./data")
COLLECTION = "enterprise_docs"
EMBED_MODEL = "intfloat/multilingual-e5-small"

MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 800

_FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    text = text.replace("\r\n", "\n")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    raw_meta, body = match.group(1), match.group(2)
    meta: dict[str, str] = {}

    for line in raw_meta.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")

    return meta, body


def chunk_markdown(body: str) -> list[str]:
    body = body.replace("\r\n", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    chunks: list[str] = []
    buffer = ""

    for p in paragraphs:
        if len(p) > MAX_CHUNK_CHARS:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""

            for piece in re.split(r"(?<=[.!?])\s+|\n", p):
                piece = piece.strip()
                if not piece:
                    continue

                if buffer and len(buffer) + len(piece) > MAX_CHUNK_CHARS:
                    chunks.append(buffer.strip())
                    buffer = piece
                else:
                    buffer = (buffer + " " + piece).strip()
            continue

        if not buffer:
            buffer = p
        elif len(buffer) + len(p) + 1 <= MAX_CHUNK_CHARS:
            buffer = buffer + "\n" + p
        else:
            chunks.append(buffer.strip())
            buffer = p

    if buffer:
        chunks.append(buffer.strip())

    merged: list[str] = []
    for c in chunks:
        if merged and len(merged[-1]) < MIN_CHUNK_CHARS:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)

    return merged


def collection_has_data(client: chromadb.HttpClient) -> bool:
    try:
        col = client.get_collection(COLLECTION)
        return col.count() > 0
    except Exception:
        return False


def ingest(force: bool = False) -> None:
    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT} …")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    if not force and collection_has_data(client):
        count = client.get_collection(COLLECTION).count()
        print(f"✓ Collection '{COLLECTION}' already has {count} chunks — skipping ingestion.")
        return

    try:
        client.delete_collection(COLLECTION)
        print(f"Deleted existing collection '{COLLECTION}'.")
    except Exception:
        pass

    client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    collection = client.get_collection(COLLECTION)

    md_files = sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.md")))
    if not md_files:
        raise FileNotFoundError(
            f"No markdown files found under '{DATA_DIR}/*/'."
        )

    print(f"Loading embedding model '{EMBED_MODEL}' …")
    embedder = SentenceTransformer(EMBED_MODEL)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    per_type_counts: dict[str, int] = {}

    for filepath in md_files:
        rel = os.path.relpath(filepath, DATA_DIR).replace(os.sep, "/")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        meta, body = parse_frontmatter(text)

        doc_type = meta.get("type") or Path(filepath).parent.name
        title = meta.get("title", Path(filepath).stem)
        name = meta.get("name", "")
        department = meta.get("department", "")

        if doc_type == "project":
            project = meta.get("name", title)
        else:
            project = meta.get("project", "")

        role = meta.get("role", "")
        manager = meta.get("manager", "")
        employee_id = meta.get("employee_id", "")
        project_id = meta.get("project_id", "")
        policy_id = meta.get("policy_id", "")
        faq_id = meta.get("faq_id", "")
        meeting_id = meta.get("meeting_id", "")
        month = meta.get("month", "")

        chunks = chunk_markdown(body)
        per_type_counts[doc_type] = per_type_counts.get(doc_type, 0) + len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_type}__{Path(filepath).stem}__{i:03d}"

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "type": doc_type,
                "title": title,
                "name": name,
                "department": department,
                "project": project,
                "role": role,
                "manager": manager,
                "employee_id": employee_id,
                "project_id": project_id,
                "policy_id": policy_id,
                "faq_id": faq_id,
                "meeting_id": meeting_id,
                "month": month,
                "source_path": rel,
                "chunk_index": i,
            })

    print(f"\nChunking summary ({len(ids)} chunks across {len(md_files)} files):")
    for doc_type, n in sorted(per_type_counts.items()):
        print(f"  • {doc_type:15s} → {n:4d} chunks")

    passages = [f"passage: {doc}" for doc in documents]

    print(f"\nEmbedding {len(passages)} chunks …")
    t0 = time.time()
    vecs = embedder.encode(passages, show_progress_bar=True, batch_size=32)
    embeddings = vecs.tolist()
    print(f"Embedding done in {time.time() - t0:.1f}s")

    batch_size = 200
    for start in range(0, len(ids), batch_size):
        sl = slice(start, start + batch_size)
        collection.add(
            ids=ids[sl],
            documents=documents[sl],
            embeddings=embeddings[sl],
            metadatas=metadatas[sl],
        )

    print(f"\n[OK] Ingested {len(ids)} chunks into collection '{COLLECTION}'.")


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    ingest(force=force)