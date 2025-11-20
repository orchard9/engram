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
Engram Server Health Check
═══════════════════════════════════════
Process ID: 12345
HTTP Port: 7432
Process Status: Running
HTTP Health: Responding (1.2ms)

API Endpoints:
  System Health API: 200 OK
  Memory Recall API: 200 OK

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

## Memory Spaces (Multi-Tenancy)

Engram supports isolated memory spaces for multi-tenant deployments. Each space maintains separate storage, persistence, and metrics.

### Quick Start with Spaces

```bash
# List all memory spaces (starts with 'default')
./target/debug/engram space list

# Create a new memory space
./target/debug/engram space create research

# Store a memory in the 'research' space
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -H "X-Memory-Space: research" \
  -d '{"content": "CRISPR enables precise gene editing", "confidence": 0.95}'

# Recall from the 'research' space
curl -H "X-Memory-Space: research" \
  "http://localhost:7432/api/v1/memories/recall?query=gene%20editing"

# Check per-space health metrics
./target/debug/engram status --space research

```

### Space Routing Options

You can specify memory spaces in three ways:

1. **HTTP Header** (recommended):
   ```bash
   curl -H "X-Memory-Space: production" \
     http://localhost:7432/api/v1/memories/recall?query=data
   ```

2. **Query Parameter**:
   ```bash
   curl "http://localhost:7432/api/v1/memories/recall?space=production&query=data"
   ```

3. **Environment Variable** (CLI):
   ```bash
   export ENGRAM_MEMORY_SPACE=production
   ./target/debug/engram memory list
   ```

### Default Space Behavior

When no space is specified, all operations use the `default` space. This ensures backward compatibility with existing single-tenant deployments.

For production multi-tenant deployments, see the [Memory Space Migration Guide](../operations/memory-space-migration.md).

## Pattern Completion (Beta)

Pattern completion reconstructs missing memory details using hippocampal CA3 attractor dynamics:

```bash
# Store a complete memory
curl -X POST http://localhost:7432/api/v1/episodes/remember \
  -H "Content-Type: application/json" \
  -d '{
    "what": "Einstein published theory of relativity in 1915",
    "when": "2024-01-05T10:00:00Z",
    "where": "Physics lecture",
    "confidence": 0.90
  }'

# Complete a partial memory
curl -X POST http://localhost:7432/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "partial": {
      "what": "Einstein published theory",
      "when": null,
      "where": "Physics lecture"
    }
  }'

```

The response includes:

- `completed` - Reconstructed episode with filled-in details

- `source` - How completion was achieved (Recalled, Reconstructed, Imagined, Consolidated)

- `completion_confidence` - Multi-factor confidence score (convergence speed, energy reduction, field consensus)

- `alternatives` - Alternative hypotheses from System 2 reasoning

For production parameter tuning, see [Pattern Completion Parameter Tuning Guide](../tuning/completion_parameters.md).

## Next Steps

- **[API Reference](/api/)**: Learn about memory operations and system endpoints

- **[Memory Operations](/api/memory)**: Deep dive into storing and recalling memories

- **[System Health](/api/system)**: Monitor your Engram instance

## Need Help?

- Check the server logs if something isn't working

- All API endpoints are documented at `http://localhost:7432/docs/` when the server is running

- File issues on [GitHub](https://github.com/orchard9/engram)
