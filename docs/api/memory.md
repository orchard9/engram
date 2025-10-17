# Memory Operations

## Store a Memory

Store new information in Engram's cognitive memory system.

**Endpoint:** `POST /api/v1/memories/remember`

### Request

```json
{
  "content": "The capital of France is Paris",
  "confidence": 0.9
}
```

### Response

```json
{
  "memory_id": "mem_a1b2c3d4",
  "storage_confidence": {
    "value": 0.9,
    "category": "High",
    "reasoning": "Strong factual content with clear semantic meaning"
  },
  "consolidation_state": "Recent",
  "auto_links": [],
  "system_message": "Memory successfully stored with high confidence"
}
```

## Recall Memories

Search and retrieve memories using natural language queries.

**Endpoint:** `GET /api/v1/memories/recall?query={search_terms}`

### Parameters

- `query` (required) - Search terms or question
- `confidence_threshold` (optional) - Minimum confidence (0.0-1.0)
- `max_results` (optional) - Maximum number of results (default: 10)

### Example

```bash
curl "http://localhost:7432/api/v1/memories/recall?query=capital+France"
```

### Response

```json
{
  "memories": {
    "vivid": [
      {
        "id": "mem_a1b2c3d4",
        "content": "The capital of France is Paris",
        "confidence": {
          "value": 0.9,
          "category": "High",
          "reasoning": "Direct factual match"
        },
        "activation_level": 0.95,
        "similarity_score": 0.92,
        "retrieval_path": "Direct similarity match",
        "last_access": "2025-09-17T01:30:00Z"
      }
    ],
    "associated": [],
    "reconstructed": []
  },
  "recall_confidence": {
    "value": 0.9,
    "category": "High",
    "reasoning": "Strong query-memory alignment"
  }
}
```

## Pattern Recognition

Recognize patterns in provided content.

**Endpoint:** `POST /api/v1/memories/recognize`

### Request

```json
{
  "content": "Paris is a beautiful city in Europe",
  "context": "geographical"
}
```

### Response

```json
{
  "recognized_patterns": [
    {
      "pattern_type": "Location",
      "confidence": 0.85,
      "description": "Geographic location reference"
    }
  ],
  "system_message": "Pattern recognition completed successfully"
}
```
