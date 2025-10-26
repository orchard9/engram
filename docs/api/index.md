# API Reference

Engram provides a REST API for interacting with your cognitive memory system. All endpoints return JSON and support CORS for web applications.

## Base URL

When running locally: `http://localhost:7432`

## Quick Reference

### Memory Operations

- `POST /api/v1/memories/remember` - Store a memory
- `GET /api/v1/memories/recall` - Search and recall memories
- `POST /api/v1/memories/recognize` - Pattern recognition
- `POST /api/v1/complete` - Pattern completion (Beta) - Reconstruct missing details from partial memories

### System Endpoints

- `GET /health` - Simple health check
- `GET /api/v1/system/health` - Detailed system health
- `GET /api/v1/system/introspect` - System statistics

### Interactive Documentation

For full interactive API documentation with examples, visit:
`http://localhost:7432/docs/` (when server is running)

## Authentication

Currently no authentication is required for local development. Production deployments should implement proper authentication.

## Multi-Tenancy (Memory Spaces)

Engram supports isolated memory spaces for multi-tenant deployments. Each space maintains separate storage, persistence, and metrics.

### X-Memory-Space Header

Use the `X-Memory-Space` header to route requests to specific memory spaces:

```bash
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -H "X-Memory-Space: production" \
  -d '{"content": "Important data", "confidence": 0.95}'
```

### Space Routing Precedence

When multiple sources specify a memory space, the following precedence applies:

1. **Header**: `X-Memory-Space` header (highest priority)
2. **Query Parameter**: `?space=<space_id>` in URL
3. **Request Body**: `"memory_space"` field in JSON payload

If no space is specified, requests default to the `default` space for backward compatibility.

### Examples

Using header (recommended):

```bash
curl -H "X-Memory-Space: research" \
  http://localhost:7432/api/v1/memories/recall?query=CRISPR
```

Using query parameter:

```bash
curl http://localhost:7432/api/v1/memories/recall?space=research&query=CRISPR
```

Using request body:

```bash
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Data", "confidence": 0.9, "memory_space": "research"}'
```

### Space Management

List available spaces:

```bash
curl http://localhost:7432/api/v1/spaces
```

Get per-space health metrics:

```bash
curl http://localhost:7432/api/v1/system/health
```

Response includes metrics for all spaces:

```json
{
  "spaces": [
    {
      "space": "default",
      "memories": 150,
      "pressure": 0.0,
      "wal_lag_ms": 0.0,
      "consolidation_rate": 0.0
    },
    {
      "space": "production",
      "memories": 450,
      "pressure": 0.0,
      "wal_lag_ms": 0.0,
      "consolidation_rate": 0.0
    }
  ]
}
```

## Error Handling

All errors return JSON with a consistent structure:

```json
{
  "error": "description of the error",
  "code": "ERROR_CODE",
  "details": {}
}
```

## Response Format

Successful responses include cognitive context and system messages:

```json
{
  "data": {},
  "system_message": "Human-readable explanation",
  "confidence": {
    "value": 0.8,
    "category": "High",
    "reasoning": "Why this confidence level"
  }
}
```
