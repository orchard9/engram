# Getting Started

## Quick Start

Get Engram running in under 5 minutes:

### 1. Build and Start

```bash
# Build the project
cargo build

# Start the Engram server
./target/debug/engram start
```

### 2. Verify It's Working

```bash
# Check server status
./target/debug/engram status
```

You should see:
```
🏥 Engram Server Health Check
═══════════════════════════════════════
📋 Process ID: 12345
🌐 HTTP Port: 7432
✅ Process Status: Running
✅ HTTP Health: Responding (1.2ms)

🔌 API Endpoints:
  ✅ System Health API: 200 OK
  ✅ Memory Recall API: 200 OK
```

### 3. Run Tests

```bash
# Verify everything works
cargo test
```

### 4. Stop the Server

```bash
./target/debug/engram stop
```

## Your First Memory

Once Engram is running, try storing and recalling a memory:

```bash
# Store a memory
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Rust is a systems programming language", "confidence": 0.9}'

# Recall memories
curl "http://localhost:7432/api/v1/memories/recall?query=programming"
```

## Next Steps

- **[API Reference](/api/)**: Learn about memory operations and system endpoints
- **[Memory Operations](/api/memory)**: Deep dive into storing and recalling memories
- **[System Health](/api/system)**: Monitor your Engram instance

## Need Help?

- Check the server logs if something isn't working
- All API endpoints are documented at `http://localhost:7432/docs/` when the server is running
- File issues on [GitHub](https://github.com/orchard9/engram)