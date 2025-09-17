# API Reference

Engram provides a REST API for interacting with your cognitive memory system. All endpoints return JSON and support CORS for web applications.

## Base URL

When running locally: `http://localhost:7432`

## Quick Reference

### Memory Operations
- `POST /api/v1/memories/remember` - Store a memory
- `GET /api/v1/memories/recall` - Search and recall memories
- `POST /api/v1/memories/recognize` - Pattern recognition

### System Endpoints
- `GET /health` - Simple health check
- `GET /api/v1/system/health` - Detailed system health
- `GET /api/v1/system/introspect` - System statistics

### Interactive Documentation

For full interactive API documentation with examples, visit:
`http://localhost:7432/docs/` (when server is running)

## Authentication

Currently no authentication is required for local development. Production deployments should implement proper authentication.

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