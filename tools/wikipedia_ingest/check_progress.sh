#!/bin/bash
# Quick status check for Wikipedia ingestion

echo "=== Wikipedia Ingestion Status ==="
echo

# Check if dump downloaded
if [ -f "data/simplewiki-latest-pages-articles.xml.bz2" ]; then
    SIZE=$(ls -lh data/simplewiki-latest-pages-articles.xml.bz2 | awk '{print $5}')
    echo "✓ Wikipedia dump downloaded ($SIZE)"
else
    echo "⏳ Downloading Wikipedia dump..."
fi

# Check if model cached
if [ -d "$HOME/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2" ]; then
    echo "✓ Embedding model cached"
else
    echo "⏳ Downloading embedding model (first run only)..."
fi

# Check if checkpoints exist
if [ -d "checkpoints" ] && [ -f "checkpoints/latest.json" ]; then
    INGESTED=$(jq '.stats.ingested' checkpoints/latest.json 2>/dev/null || echo "0")
    echo "✓ Progress: $INGESTED articles ingested"
fi

# Check Engram memory count
MEMORIES=$(curl -s http://localhost:7432/api/v1/memories 2>/dev/null | jq '.memories | length' 2>/dev/null || echo "0")
echo "✓ Engram database: $MEMORIES memories"

echo
echo "Run 'tail -f logs/ingest.log' for detailed progress (once logging starts)"
