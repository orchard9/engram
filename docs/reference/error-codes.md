# Error Codes Reference

Comprehensive error catalog for Engram with remediation guidance. Every error includes educational context to help you understand what went wrong and how to fix it.

## Table of Contents

- [Error Structure](#error-structure)

- [Storage Errors (ERR-1xxx)](#storage-errors-err-1xxx)

- [Retrieval Errors (ERR-2xxx)](#retrieval-errors-err-2xxx)

- [Consolidation Errors (ERR-3xxx)](#consolidation-errors-err-3xxx)

- [Validation Errors (ERR-4xxx)](#validation-errors-err-4xxx)

- [System Errors (ERR-5xxx)](#system-errors-err-5xxx)

- [Quick Reference Table](#quick-reference-table)

- [Diagnostic Commands](#diagnostic-commands)

## Error Structure

All Engram errors follow this educational format:

```json
{
  "error": {
    "code": "ERR-XXXX",
    "summary": "One-line error description",
    "context": "What was expected vs what actually happened",
    "suggestion": "How to fix this error",
    "example": {
      "current": "Your current code",
      "recommended": "Fixed version"
    },
    "error_confidence": {
      "value": 0.95,
      "reasoning": "Why we're confident about this diagnosis"
    },
    "similar_errors": ["ERR-YYYY", "ERR-ZZZZ"]
  }
}

```

Think of errors as teaching moments - they explain cognitive concepts while helping you debug.

## Storage Errors (ERR-1xxx)

Errors related to memory storage and memory space operations.

---

### ERR-1001: Embedding Dimension Mismatch

**Category:** Storage
**HTTP Status:** 400 Bad Request
**gRPC Status:** INVALID_ARGUMENT

#### Description

The embedding vector dimension doesn't match the configured dimension (default: 768).

Think of this like trying to fit a square peg in a round hole - Engram's vector index is optimized for a specific dimension, and mixing dimensions would break similarity search.

#### Common Causes

- Using embeddings from different models (e.g., BERT-base-768 vs GPT-small-1536)

- Accidentally truncating or padding vectors

- Copy-paste errors in embedding arrays

#### Resolution Steps

1. **Verify embedding model matches server configuration**

   ```bash
   # Check server embedding dimension
   curl http://localhost:8080/api/v1/introspect | jq '.metrics.embedding_dimension'
   # Should return: 768
   ```

2. **Ensure your embedding model produces correct dimensions**

   ```python
   # Python example with sentence-transformers
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('all-MiniLM-L6-v2')  # Produces 384-dim
   # WRONG for default Engram (expects 768)

   model = SentenceTransformer('all-mpnet-base-v2')  # Produces 768-dim
   # CORRECT for default Engram
   ```

3. **If you need different dimensions, reconfigure server**

   ```bash
   # Start Engram with custom dimension
   engram start --embedding-dimension 384
   ```

#### Example Request (Triggers Error)

```bash
curl -X POST http://localhost:8080/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "memory": {
      "content": "Test memory",
      "embedding": [0.1, 0.2, 0.3],  # Only 3 dimensions!
      "confidence": {"value": 0.8}
    }
  }'

```

#### Example Response

```json
{
  "error": {
    "code": "ERR-1001",
    "summary": "Embedding dimension mismatch",
    "context": "Expected 768 dimensions, received 3 dimensions. Engram requires consistent embedding dimensions for HNSW index integrity.",
    "suggestion": "Use an embedding model that produces 768-dimensional vectors (e.g., all-mpnet-base-v2, instructor-large), or reconfigure Engram with --embedding-dimension flag to match your model.",
    "example": {
      "current": "embedding: [0.1, 0.2, 0.3]  # 3 dimensions",
      "recommended": "embedding: [0.1, 0.2, ..., 0.768]  # 768 dimensions"
    },
    "error_confidence": {
      "value": 1.0,
      "reasoning": "Dimension count validation is deterministic"
    },
    "similar_errors": ["ERR-4003"]
  }
}

```

#### Related Errors

- [ERR-4003: Malformed Embedding Vector](#err-4003-malformed-embedding-vector)

---

### ERR-1002: Invalid Confidence Value

**Category:** Storage
**HTTP Status:** 400 Bad Request
**gRPC Status:** INVALID_ARGUMENT

#### Description

Confidence value must be in range [0.0, 1.0]. Values outside this range don't have cognitive meaning.

In Engram, confidence represents uncertainty like probability - 0.0 means "completely uncertain" and 1.0 means "absolutely certain". Values like 1.5 or -0.3 are meaningless.

#### Common Causes

- Confusing confidence with other scales (e.g., 0-100 percentage)

- Negative values from unchecked calculations

- Using confidence as a priority score

#### Resolution Steps

1. **Normalize confidence to [0.0, 1.0]**

   ```python
   # If you have percentage (0-100)
   confidence = percentage / 100.0

   # If you have unnormalized score
   confidence = min(max(score / max_score, 0.0), 1.0)
   ```

2. **Use confidence categories if uncertain about numeric values**

   ```python
   # Map intuitive categories to numbers
   CONFIDENCE_MAP = {
       'certain': 0.95,
       'high': 0.80,
       'medium': 0.60,
       'low': 0.40,
       'uncertain': 0.20
   }

   confidence = CONFIDENCE_MAP['high']  # 0.80
   ```

3. **Include reasoning to justify confidence**

   ```json
   {
     "confidence": {
       "value": 0.95,
       "reasoning": "Verified from authoritative source"
     }
   }
   ```

#### Example Response

```json
{
  "error": {
    "code": "ERR-1002",
    "summary": "Invalid confidence value",
    "context": "Confidence must be in range [0.0, 1.0], received: 1.5. Confidence represents probabilistic certainty, where 0.0 = completely uncertain and 1.0 = absolutely certain.",
    "suggestion": "If you're using percentages (0-100), divide by 100. If you're using priorities, map them to [0.0, 1.0] range. See cognitive confidence guidelines.",
    "example": {
      "current": "confidence: { value: 1.5 }",
      "recommended": "confidence: { value: 0.95, reasoning: 'Very confident from reliable source' }"
    },
    "error_confidence": {
      "value": 1.0,
      "reasoning": "Range validation is deterministic"
    }
  }
}

```

---

### ERR-1003: Memory Space Not Found

**Category:** Storage
**HTTP Status:** 404 Not Found
**gRPC Status:** NOT_FOUND

#### Description

The specified memory space doesn't exist. Memory spaces provide tenant isolation - like databases in PostgreSQL or keyspaces in Cassandra.

#### Common Causes

- Typo in memory_space_id

- Memory space not created yet

- Using wrong credentials for different tenant

#### Resolution Steps

1. **List available memory spaces**

   ```bash
   engram space list
   ```

2. **Create memory space if needed**

   ```bash
   engram space create --id "my_space" --description "My memory space"
   ```

3. **Verify authentication scope**

   ```bash
   # Check which memory space your API key can access
   curl -H "Authorization: Bearer ${API_KEY}" \
     http://localhost:8080/api/v1/introspect | jq '.metrics.memory_space_id'
   ```

#### Example Response

```json
{
  "error": {
    "code": "ERR-1003",
    "summary": "Memory space not found",
    "context": "Memory space 'tenant_42_memories' does not exist. Available spaces: ['default', 'tenant_1_memories', 'tenant_3_memories']",
    "suggestion": "Create the memory space with 'engram space create --id tenant_42_memories', or check for typos in memory_space_id. For multi-tenant deployments, verify your API key has access to this space.",
    "example": {
      "current": "memory_space_id: 'tenant_42_memories'",
      "recommended": "memory_space_id: 'default'  # or create space first"
    }
  }
}

```

---

### ERR-1004: Duplicate Memory ID

**Category:** Storage
**HTTP Status:** 409 Conflict
**gRPC Status:** ALREADY_EXISTS

#### Description

Memory ID already exists. Unlike traditional databases, Engram requires unique memory IDs for graph integrity.

#### Resolution Steps

1. **Use server-generated IDs (recommended)**

   ```python
   # Don't specify ID - let Engram generate it
   memory = Memory(
       content="...",
       embedding=[...],
       # No 'id' field - server generates unique ID
   )
   ```

2. **Check if memory exists first (deduplication pattern)**

   ```python
   # Recognize before remember
   response = client.recognize(content="...")

   if response.recognized and response.recognition_confidence.value > 0.9:
       print(f"Memory already exists: {response.similar_memories[0].id}")
   else:
       # Store new memory
       client.remember(memory)
   ```

3. **Use content-addressable IDs**

   ```python
   import hashlib

   # Generate deterministic ID from content
   content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
   memory_id = f"mem_{content_hash}"
   ```

#### Example Response

```json
{
  "error": {
    "code": "ERR-1004",
    "summary": "Duplicate memory ID",
    "context": "Memory with ID 'mem_1698765432_a1b2c3d4' already exists. Creating duplicates would break graph integrity and spreading activation paths.",
    "suggestion": "Let Engram generate IDs automatically (recommended), or use the recognize endpoint to check for duplicates before storing. For intentional updates, use forget + remember pattern.",
    "example": {
      "current": "memory: { id: 'mem_1698765432_a1b2c3d4', ... }",
      "recommended": "memory: { content: '...', embedding: [...] }  # No ID"
    }
  }
}

```

## Retrieval Errors (ERR-2xxx)

Errors related to memory recall and retrieval operations.

---

### ERR-2001: No Memories Above Threshold

**Category:** Retrieval
**HTTP Status:** 200 OK (not an error!)
**gRPC Status:** OK

#### Description

This is actually a **successful response**, not an error. No memories matched the confidence threshold, but the query succeeded.

This is a key cognitive difference: unlike database queries that fail when no results are found, Engram acknowledges "I searched but didn't find anything confident enough" is a valid outcome.

#### Common Causes

- Threshold too high for available memories

- Query too specific or novel

- Natural state for new memory spaces

#### Resolution Steps

1. **Lower confidence threshold**

   ```python
   # Too strict
   response = client.recall(cue, threshold=0.9)  # Might return nothing

   # More permissive
   response = client.recall(cue, threshold=0.6)  # Returns uncertain matches
   ```

2. **Check returned confidence**

   ```python
   response = client.recall(cue)

   if response.recall_confidence.value < 0.5:
       print(f"Low confidence recall: {response.recall_confidence.reasoning}")
       print("Consider: broader query, lower threshold, or query is genuinely novel")
   ```

3. **Use pattern completion for partial matches**

   ```python
   # If recall returns nothing, try pattern completion
   if len(response.memories) == 0:
       completion = client.complete(partial_pattern=cue)
   ```

#### Example Response

```json
{
  "memories": [],
  "recall_confidence": {
    "value": 0.35,
    "category": "CONFIDENCE_CATEGORY_LOW",
    "reasoning": "Searched 45,678 memories, none exceeded threshold 0.9. Highest similarity: 0.72 (below threshold). Consider lowering threshold or query is genuinely novel."
  },
  "metadata": {
    "total_activated": 45678,
    "above_threshold": 0,
    "avg_activation": 0.23,
    "recall_time_ms": 67
  }
}

```

---

### ERR-2002: Cue Type Not Supported

**Category:** Retrieval
**HTTP Status:** 400 Bad Request
**gRPC Status:** INVALID_ARGUMENT

#### Description

The cue type is not recognized or not supported in this Engram version.

#### Supported Cue Types

- `embedding`: Direct vector similarity

- `semantic`: Natural language query

- `context`: Episodic context (time/location/people)

- `temporal`: Temporal pattern matching

- `pattern`: Partial memory completion

#### Resolution Steps

```python
# Correct cue types
cues = {
    'embedding': Cue(embedding=EmbeddingCue(vector=[...])),
    'semantic': Cue(semantic=SemanticCue(query="...")),
    'context': Cue(context=ContextCue(location="...", time_start=...)),
    'temporal': Cue(temporal=TemporalCue(...)),
    'pattern': Cue(pattern=PatternCue(...))
}

```

---

### ERR-2003: Activation Spreading Timeout

**Category:** Retrieval
**HTTP Status:** 408 Request Timeout
**gRPC Status:** DEADLINE_EXCEEDED

#### Description

Spreading activation did not converge within timeout. The activation spread to too many nodes before settling.

Think of this like a wildfire spreading faster than you can track it - the query was too broad and activated too much of the memory graph.

#### Common Causes

- Unlimited hops with low decay (activation spreads indefinitely)

- Very general query in large memory space

- Highly interconnected memory graph

#### Resolution Steps

1. **Limit maximum hops**

   ```python
   # Unbounded spreading (can timeout)
   cue = Cue(
       semantic=SemanticCue(query="knowledge"),
       spread_activation=True
   )

   # Bounded spreading (safer)
   cue = Cue(
       semantic=SemanticCue(query="knowledge"),
       spread_activation=True,
       max_hops=3,  # Stop after 3 hops
       activation_decay=0.3  # Decay faster
   )
   ```

2. **Increase activation decay**

   ```python
   # Slow decay (spreads far)
   cue.activation_decay = 0.1

   # Fast decay (stays local)
   cue.activation_decay = 0.5
   ```

3. **Use pattern completion for broad queries**

   ```python
   # Instead of spreading activation
   response = client.recall(cue, spread_activation=True)  # Might timeout

   # Use pattern completion
   response = client.complete(partial_pattern)  # Faster, bounded
   ```

4. **Increase timeout for complex queries**

   ```python
   # Python
   response = client.recall(cue, timeout=10.0)  # 10 seconds

   # Go
   ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
   defer cancel()
   response, err := client.Recall(ctx, request)
   ```

#### Example Response

```json
{
  "error": {
    "code": "ERR-2003",
    "summary": "Activation spreading timeout",
    "context": "Spreading activation did not converge within 5000ms. Started from 1 seed memory, reached 45,123 nodes before timeout. Activation pattern showed divergence (no convergence plateau).",
    "suggestion": "Reduce max_hops (currently: unlimited) or increase activation_decay (currently: 0.1). For broad queries, consider using pattern completion instead of full spreading activation. See activation tuning guide.",
    "example": {
      "current": "recall(cue, spread_activation=true)",
      "recommended": "recall(cue, spread_activation=true, max_hops=3, activation_decay=0.3)"
    },
    "error_confidence": {
      "value": 0.95,
      "reasoning": "Timeout threshold exceeded, activation log shows clear divergence pattern"
    },
    "similar_errors": ["ERR-2004", "ERR-5001"]
  }
}

```

---

### ERR-2004: Invalid Similarity Threshold

**Category:** Retrieval
**HTTP Status:** 400 Bad Request
**gRPC Status:** INVALID_ARGUMENT

#### Description

Similarity threshold must be in range [0.0, 1.0]. Like confidence, similarity is a normalized score.

#### Resolution Steps

```python
# Correct threshold range
cue = EmbeddingCue(
    vector=[...],
    similarity_threshold=0.75  # Valid: 0.0 to 1.0
)

# Typical threshold values
thresholds = {
    'strict': 0.9,     # Very similar memories only
    'normal': 0.7,     # Moderately similar
    'permissive': 0.5  # Loosely similar
}

```

## Consolidation Errors (ERR-3xxx)

Errors related to memory consolidation operations.

---

### ERR-3001: Consolidation In Progress

**Category:** Consolidation
**HTTP Status:** 409 Conflict
**gRPC Status:** FAILED_PRECONDITION

#### Description

A consolidation process is already running. Engram prevents concurrent consolidation to maintain graph consistency.

Think of this like sleep - you can't sleep twice at the same time. Each consolidation cycle needs to complete before starting another.

#### Resolution Steps

1. **Wait for current consolidation to complete**

   ```bash
   # Check consolidation status
   curl http://localhost:8080/api/v1/introspect | jq '.active_processes'
   # Look for "consolidation_daemon"
   ```

2. **Monitor consolidation progress**

   ```python
   # Stream consolidation progress
   async for event in client.dream(replay_cycles=10):
       if event['type'] == 'progress':
           print(f"Progress: {event['connections']} new connections")
   ```

3. **Cancel running consolidation (if needed)**

   ```bash
   # Emergency cancel (loses progress)
   engram consolidate cancel --force
   ```

#### Example Response

```json
{
  "error": {
    "code": "ERR-3001",
    "summary": "Consolidation in progress",
    "context": "Consolidation started at 2024-10-27T10:15:00Z, currently 35% complete (342 of 980 memories processed). Concurrent consolidation would break graph consistency.",
    "suggestion": "Wait for current consolidation to complete (estimated 3 minutes remaining), or cancel it with 'engram consolidate cancel'. Check progress via /introspect endpoint.",
    "estimated_completion": "2024-10-27T10:18:00Z"
  }
}

```

---

### ERR-3002: Insufficient Memories to Consolidate

**Category:** Consolidation
**HTTP Status:** 400 Bad Request
**gRPC Status:** FAILED_PRECONDITION

#### Description

Not enough memories meet consolidation criteria. Consolidation needs a minimum number of memories to find patterns.

#### Resolution Steps

1. **Check pending memories count**

   ```bash
   curl http://localhost:8080/api/v1/introspect | \
     jq '.metrics.consolidation_pending'
   ```

2. **Lower importance threshold**

   ```python
   # Too strict (might skip memories)
   response = client.consolidate(
       criteria=ConsolidationCriteria(importance_threshold=0.9)
   )

   # More inclusive
   response = client.consolidate(
       criteria=ConsolidationCriteria(importance_threshold=0.5)
   )
   ```

3. **Wait for more memories to accumulate**

   ```python
   # Consolidation works best with batches
   MIN_BATCH_SIZE = 100

   metrics = client.introspect()
   if metrics.consolidation_pending >= MIN_BATCH_SIZE:
       client.consolidate()
   ```

---

### ERR-3003: Memory Space Locked

**Category:** Consolidation
**HTTP Status:** 423 Locked
**gRPC Status:** FAILED_PRECONDITION

#### Description

Memory space is locked for maintenance (backup, migration, or administrative operation).

#### Resolution Steps

```bash
# Check lock status
engram space info --id "my_space" | jq '.locked'

# Wait for unlock
engram space wait-unlock --id "my_space" --timeout 300

# Force unlock (admin only, dangerous)
engram space unlock --id "my_space" --force

```

## Validation Errors (ERR-4xxx)

Errors related to request validation and authentication.

---

### ERR-4001: Missing Required Field

**Category:** Validation
**HTTP Status:** 400 Bad Request
**gRPC Status:** INVALID_ARGUMENT

#### Description

A required field is missing from the request.

#### Common Missing Fields

```python
# Missing embedding
memory = Memory(
    content="Python is great",
    # ERROR: Missing embedding field
    confidence=Confidence(value=0.9)
)

# Fixed
memory = Memory(
    content="Python is great",
    embedding=[...],  # Required
    confidence=Confidence(value=0.9)
)

# Missing confidence
memory = Memory(
    content="Python is great",
    embedding=[...],
    # ERROR: Missing confidence field
)

# Fixed - confidence is ALWAYS required
memory = Memory(
    content="Python is great",
    embedding=[...],
    confidence=Confidence(value=0.8, reasoning="Estimated")
)

```

---

### ERR-4002: Invalid Timestamp Format

**Category:** Validation
**HTTP Status:** 400 Bad Request
**gRPC Status:** INVALID_ARGUMENT

#### Description

Timestamp must be in RFC3339 format or use protobuf Timestamp.

#### Resolution Steps

```python
# Correct formats
from google.protobuf.timestamp_pb2 import Timestamp
from datetime import datetime

# Proto timestamp (gRPC)
timestamp = Timestamp()
timestamp.FromDatetime(datetime.now())

# RFC3339 string (REST)
timestamp_str = "2024-10-27T10:30:00Z"
timestamp_str = datetime.now().isoformat() + "Z"

```

---

### ERR-4003: Malformed Embedding Vector

**Category:** Validation
**HTTP Status:** 400 Bad Request
**gRPC Status:** INVALID_ARGUMENT

#### Description

Embedding vector contains invalid values (NaN, Infinity, or non-numeric).

#### Resolution Steps

```python
import numpy as np

# Check for invalid values
embedding = np.array([0.1, 0.2, np.nan, 0.4])  # Contains NaN

if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
    # Replace NaN/Inf with zeros or re-generate embedding
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)

# Normalize if needed
embedding = embedding / np.linalg.norm(embedding)

```

---

### ERR-4004: Authentication Failed

**Category:** Validation
**HTTP Status:** 401 Unauthorized
**gRPC Status:** UNAUTHENTICATED

#### Description

Invalid or missing API key, or token expired.

#### Resolution Steps

1. **Check API key format**

   ```bash
   # Valid format: ek_live_<32_hex_chars>
   echo $ENGRAM_API_KEY
   # Should look like: ek_live_1234567890abcdef...
   ```

2. **Verify API key is active**

   ```bash
   engram auth verify --key "${ENGRAM_API_KEY}"
   ```

3. **Check token expiration (JWT)**

   ```python
   import jwt

   token = "eyJ..."
   decoded = jwt.decode(token, options={"verify_signature": False})
   exp = decoded['exp']

   if datetime.fromtimestamp(exp) < datetime.now():
       print("Token expired, need to refresh")
   ```

4. **Generate new API key**

   ```bash
   engram auth create-key --name "new-key"
   ```

## System Errors (ERR-5xxx)

Errors related to system resources and availability.

---

### ERR-5001: Storage Tier Unavailable

**Category:** System
**HTTP Status:** 503 Service Unavailable
**gRPC Status:** UNAVAILABLE

#### Description

One or more storage tiers (hot/warm/cold) are unavailable.

#### Resolution Steps

1. **Check tier health**

   ```bash
   ./scripts/diagnose_health.sh
   ```

2. **Check tier configuration**

   ```bash
   engram introspect | jq '.health.components.storage'
   ```

3. **Restart unhealthy tier**

   ```bash
   # If RocksDB (cold tier) is down
   engram storage restart --tier cold
   ```

---

### ERR-5002: GPU Acceleration Failed

**Category:** System
**HTTP Status:** 500 Internal Server Error
**gRPC Status:** INTERNAL

#### Description

GPU acceleration failed, falling back to CPU. Performance degraded but operations continue.

#### Resolution Steps

```bash
# Check GPU availability
nvidia-smi

# Check CUDA version compatibility
engram version --verbose | grep -i cuda

# Restart with CPU-only mode
engram start --no-gpu

```

---

### ERR-5003: Rate Limit Exceeded

**Category:** System
**HTTP Status:** 429 Too Many Requests
**gRPC Status:** RESOURCE_EXHAUSTED

#### Description

Request rate exceeded configured limits.

#### Resolution Steps

1. **Implement exponential backoff**

   ```python
   import time

   def retry_with_backoff(fn, max_retries=5):
       for attempt in range(max_retries):
           try:
               return fn()
           except RateLimitError as e:
               if attempt == max_retries - 1:
                   raise

               backoff = min(2 ** attempt, 60)  # Max 60s
               print(f"Rate limited, retrying in {backoff}s...")
               time.sleep(backoff)
   ```

2. **Use batching/streaming**

   ```python
   # Instead of 100 individual requests
   for memory in memories:
       client.remember(memory)  # Rate limited!

   # Use streaming (higher throughput)
   client.batch_remember(memories)  # Single request
   ```

3. **Check rate limit headers**

   ```bash
   curl -i http://localhost:8080/api/v1/memories/recall
   # X-RateLimit-Limit: 100
   # X-RateLimit-Remaining: 0
   # X-RateLimit-Reset: 1698765432
   ```

#### Example Response

```json
{
  "error": {
    "code": "ERR-5003",
    "summary": "Rate limit exceeded",
    "context": "100 requests in last second (limit: 100/sec). Current usage: 100/100 requests, 0 remaining.",
    "suggestion": "Implement exponential backoff or use gRPC streaming for batch operations (4.9x higher throughput). See rate limiting guide.",
    "retry_after_seconds": 1,
    "rate_limit": {
      "limit": 100,
      "remaining": 0,
      "reset": "2024-10-27T10:30:01Z"
    }
  }
}

```

---

### ERR-5004: Service Unavailable

**Category:** System
**HTTP Status:** 503 Service Unavailable
**gRPC Status:** UNAVAILABLE

#### Description

Engram service is temporarily unavailable (startup, shutdown, or overloaded).

#### Resolution Steps

1. **Check service status**

   ```bash
   systemctl status engram
   # or
   engram status
   ```

2. **Check logs**

   ```bash
   journalctl -u engram --since "5 minutes ago"
   ```

3. **Wait and retry with backoff**

   ```python
   import time

   max_retries = 3
   for attempt in range(max_retries):
       try:
           response = client.remember(memory)
           break
       except ServiceUnavailable:
           if attempt == max_retries - 1:
               raise
           time.sleep(2 ** attempt)  # Exponential backoff
   ```

## Quick Reference Table

| Error Code | HTTP | gRPC | Category | Severity | Retriable |
|------------|------|------|----------|----------|-----------|
| ERR-1001 | 400 | INVALID_ARGUMENT | Storage | Error | No |
| ERR-1002 | 400 | INVALID_ARGUMENT | Storage | Error | No |
| ERR-1003 | 404 | NOT_FOUND | Storage | Error | No |
| ERR-1004 | 409 | ALREADY_EXISTS | Storage | Error | No |
| ERR-2001 | 200 | OK | Retrieval | Info | N/A |
| ERR-2002 | 400 | INVALID_ARGUMENT | Retrieval | Error | No |
| ERR-2003 | 408 | DEADLINE_EXCEEDED | Retrieval | Error | Yes |
| ERR-2004 | 400 | INVALID_ARGUMENT | Retrieval | Error | No |
| ERR-3001 | 409 | FAILED_PRECONDITION | Consolidation | Warning | Yes |
| ERR-3002 | 400 | FAILED_PRECONDITION | Consolidation | Warning | No |
| ERR-3003 | 423 | FAILED_PRECONDITION | Consolidation | Warning | Yes |
| ERR-4001 | 400 | INVALID_ARGUMENT | Validation | Error | No |
| ERR-4002 | 400 | INVALID_ARGUMENT | Validation | Error | No |
| ERR-4003 | 400 | INVALID_ARGUMENT | Validation | Error | No |
| ERR-4004 | 401 | UNAUTHENTICATED | Validation | Error | No |
| ERR-5001 | 503 | UNAVAILABLE | System | Critical | Yes |
| ERR-5002 | 500 | INTERNAL | System | Warning | No |
| ERR-5003 | 429 | RESOURCE_EXHAUSTED | System | Warning | Yes |
| ERR-5004 | 503 | UNAVAILABLE | System | Critical | Yes |

## Diagnostic Commands

When errors occur, use these commands to diagnose:

### Health Check

```bash
# Overall system health
./scripts/diagnose_health.sh

# Component-specific health
curl http://localhost:8080/api/v1/introspect | jq '.health'

```

### Storage Diagnostics

```bash
# Check tier distribution
curl http://localhost:8080/api/v1/introspect | \
  jq '.metrics.storage_tier_distribution'

# Check memory statistics
curl http://localhost:8080/api/v1/introspect | \
  jq '.statistics'

```

### Performance Diagnostics

```bash
# Check recent operation latencies
engram metrics --metric recall_latency_p99 --last 5m

# Check activation spreading stats
engram metrics --metric activation_spread_time --last 1h

```

### Log Analysis

```bash
# Search for errors in last hour
journalctl -u engram --since "1 hour ago" | grep ERROR

# Find specific error code
journalctl -u engram | grep "ERR-2003"

# Export diagnostics bundle
engram support export-diagnostics --output diagnostics.tar.gz

```

## Getting Help

If error persists after following resolution steps:

1. **Check documentation**
   - [REST API Reference](/reference/rest-api.md)
   - [gRPC API Reference](/reference/grpc-api.md)
   - [Operations Guide](/operations/)

2. **Run diagnostics**

   ```bash
   ./scripts/diagnose_health.sh > health_report.txt
   engram support export-diagnostics --output diagnostics.tar.gz
   ```

3. **Search existing issues**
   - GitHub Issues: https://github.com/engram/engram/issues
   - Search for error code: "ERR-2003"

4. **File bug report**
   - Include error code and full error response
   - Attach diagnostics bundle
   - Describe what you were trying to do

## Error Handling Best Practices

### Graceful Degradation

```python
def robust_recall(cue, max_retries=3):
    """Recall with progressive fallback."""
    thresholds = [0.9, 0.7, 0.5]

    for threshold in thresholds:
        try:
            response = client.recall(
                cue=cue,
                threshold=threshold,
                spread_activation=True,
                max_hops=3
            )

            if response.recall_confidence.value > 0.5:
                return response

        except ActivationTimeout:
            # Reduce complexity and retry
            cue.max_hops = max(1, cue.max_hops - 1)
            cue.activation_decay += 0.1
            continue

    # Fallback to pattern completion
    return client.complete(partial_pattern=cue)

```

### Educational Logging

```python
import logging

def handle_error(error):
    """Log errors educationally."""
    logger.error(f"Error {error.code}: {error.summary}")
    logger.info(f"Context: {error.context}")
    logger.info(f"Suggestion: {error.suggestion}")

    if error.example:
        logger.info(f"Fix: {error.example['recommended']}")

    # Track for metrics
    metrics.increment(f"errors.{error.code}")

```

### Retry Strategy

```python
from tenacity import retry, stop_after_attempt, wait_exponential

RETRIABLE_ERRORS = {'ERR-2003', 'ERR-3001', 'ERR-5001', 'ERR-5003', 'ERR-5004'}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=lambda e: getattr(e, 'code', None) in RETRIABLE_ERRORS
)
def retriable_operation():
    return client.recall(cue)

```

## Next Steps

- **API References**: [REST](/reference/rest-api.md) | [gRPC](/reference/grpc-api.md)

- **Operations**: [Troubleshooting Guide](/operations/troubleshooting.md)

- **Monitoring**: [Health Checks](/operations/monitoring.md)

- **Performance**: [Tuning Guide](/operations/performance-tuning.md)
