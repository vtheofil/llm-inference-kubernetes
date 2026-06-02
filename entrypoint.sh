#!/bin/bash
# entrypoint.sh — Backend container startup script.
#
# 1. Waits until ChromaDB is healthy.
# 2. Waits until Ollama is reachable (may be a sidecar or separate container).
# 3. Runs ingest.py if the ChromaDB collection is empty.
# 4. Starts the FastAPI server with uvicorn.

set -euo pipefail

CHROMA_HOST="${CHROMA_HOST:-localhost}"
CHROMA_PORT="${CHROMA_PORT:-8000}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

echo "=== Travel Assistant Backend ==="

# ── Wait for ChromaDB ────────────────────────────────────────────────────────
echo "Waiting for ChromaDB at ${CHROMA_HOST}:${CHROMA_PORT} …"
until curl -sf "http://${CHROMA_HOST}:${CHROMA_PORT}/api/v2/heartbeat" > /dev/null; do
    echo "  ChromaDB not ready, retrying in 3s …"
    sleep 3
done
echo "✓ ChromaDB is up."

# ── Wait for Ollama ──────────────────────────────────────────────────────────
echo "Waiting for Ollama at ${OLLAMA_URL} …"
until curl -sf "${OLLAMA_URL}/api/tags" > /dev/null; do
    echo "  Ollama not ready, retrying in 5s …"
    sleep 5
done
echo "✓ Ollama is up."

# ── Ingest documents if collection is empty ──────────────────────────────────
echo "Checking ChromaDB collection …"
python -c "
import chromadb, os, sys
host = os.getenv('CHROMA_HOST', 'localhost')
port = int(os.getenv('CHROMA_PORT', '8000'))
client = chromadb.HttpClient(host=host, port=port)
try:
    col = client.get_collection('travel_docs')
    count = col.count()
    if count > 0:
        print(f'Collection already has {count} chunks — skipping ingestion.')
        sys.exit(0)
except Exception:
    pass
print('Collection empty or missing — running ingest …')
sys.exit(1)
" && NEED_INGEST=0 || NEED_INGEST=1

if [ "$NEED_INGEST" -eq 1 ]; then
    echo "Running document ingestion …"
    python -m app.ingest
    echo "✓ Ingestion complete."
fi

# ── Start FastAPI ────────────────────────────────────────────────────────────
echo "Starting FastAPI server on port 8000 …"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
