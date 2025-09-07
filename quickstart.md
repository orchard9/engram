# Engram Quickstart: 60 Seconds to First Memory

Store and recall your first memory in under 60 seconds. No prior knowledge required.

## Install (15 seconds)

```bash
# Clone and build
git clone https://github.com/orchard9/engram.git
cd engram
cargo build --release
```

✅ **Success indicator**: Build completes with "Finished release" message

## Start Server (10 seconds)

```bash
# Start Engram on default port
./target/release/engram start
```

✅ **Success indicator**: See "Memory system ready at http://localhost:7432"

## Store Your First Memory (15 seconds)

Open a new terminal and run:

```bash
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The mitochondria is the powerhouse of the cell",
    "confidence": 0.95
  }' | jq '.'
```

✅ **Success indicator**: Returns JSON with `memory_id` like `"mem_abc123"`

Expected output:
```json
{
  "memory_id": "mem_1234abcd",
  "status": "stored",
  "activation": 1.0,
  "message": "Memory successfully encoded"
}
```

## Recall Your Memory (15 seconds)

```bash
curl -X GET "http://localhost:7432/api/v1/memories/recall?query=mitochondria" | jq '.'
```

✅ **Success indicator**: Returns your stored memory with confidence score

Expected output:
```json
{
  "memories": [
    {
      "id": "mem_1234abcd",
      "content": "The mitochondria is the powerhouse of the cell",
      "confidence": 0.95,
      "activation": 0.89,
      "relevance": 0.92
    }
  ],
  "total": 1,
  "query_analysis": {
    "understood_as": "searching for: mitochondria",
    "search_strategy": "semantic"
  }
}
```

## What Just Happened? (5 seconds to understand)

1. **Started** a cognitive memory system (not a database!)
2. **Stored** a memory with high confidence (0.95 = "very sure")
3. **Recalled** it using partial information (just "mitochondria")
4. **Activation** shows how "awake" the memory is (0.89 = highly active)

## Next Steps

### Try Episodic Memory (stores context)
```bash
curl -X POST http://localhost:7432/api/v1/episodes/remember \
  -H "Content-Type: application/json" \
  -d '{
    "what": "Learned about cellular biology",
    "when": "2024-01-15T10:30:00Z",
    "where": "Biology 101 lecture hall",
    "who": ["Professor Smith"],
    "why": "Preparing for exam",
    "how": "Taking detailed notes",
    "confidence": 0.85
  }' | jq '.'
```

### Explore the API
- Interactive docs: http://localhost:7432/docs
- OpenAPI spec: http://localhost:7432/api-docs/openapi.json

### Learn Memory Concepts
- **Confidence**: How certain the memory is (0.0-1.0)
- **Activation**: How readily accessible (decays over time)
- **Spreading**: Related memories activate each other
- **Consolidation**: Memories strengthen through replay

## Quick Troubleshooting

### "Connection refused" error
```bash
# Check if server is running
./target/release/engram status

# If not running, start it
./target/release/engram start
```

### "Command not found: jq"
```bash
# Install jq for pretty JSON (optional)
# macOS: brew install jq
# Linux: apt-get install jq
# Or just omit "| jq '.'" from commands
```

### Build fails
```bash
# Ensure Rust is installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Port already in use
```bash
# Use a different port
./target/release/engram start --port 8080

# Then update URLs to http://localhost:8080
```

### Memory not found when recalling
- Check spelling in your query
- Try broader search terms
- Memories need time to "settle" (wait 1-2 seconds after storing)

## Alternative: Docker (Zero Install)

```bash
# Coming soon - track issue #42
docker run -p 7432:7432 engram/engram:latest
```

## Get Help

- Full documentation: [usage.md](usage.md)
- API reference: http://localhost:7432/docs
- GitHub issues: https://github.com/orchard9/engram/issues
- Memory concepts: [vision.md](vision.md)

---

🎉 **Congratulations!** You've successfully stored and recalled your first memory in Engram. The system is now learning from your interactions and building connections between memories.