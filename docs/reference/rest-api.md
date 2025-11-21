# REST API Reference

Complete REST API reference for Engram's cognitive memory operations. This guide provides precise endpoint specifications while explaining the cognitive principles that make Engram different from traditional databases.

## Table of Contents

- [Getting Started](#getting-started)

- [Authentication](#authentication)

- [Core Memory Operations](#core-memory-operations)

- [Episodic Operations](#episodic-operations)

- [Consolidation Operations](#consolidation-operations)

- [Pattern Operations](#pattern-operations)

- [Monitoring Operations](#monitoring-operations)

- [Error Handling](#error-handling)

- [Rate Limiting](#rate-limiting)

- [Performance Characteristics](#performance-characteristics)

## Getting Started

Base URL: `http://localhost:8080/api/v1`

All REST endpoints use JSON for request and response bodies. The API follows RESTful conventions with cognitive-friendly method names that reflect how memory actually works.

### Why REST for Memory Operations?

While gRPC offers better performance for high-throughput scenarios, REST provides:

- Browser compatibility for web frontends

- Human-readable debugging with curl

- Simple integration for occasional API calls

- Familiar HTTP semantics

For high-throughput operations (>100/sec) or streaming consolidation, use the [gRPC API](/reference/grpc-api.md) instead.

## Authentication

Authentication is optional and controlled by the security configuration. When enabled, API requests to protected endpoints require authentication via Bearer token.

### Authentication Status

By default, authentication is disabled for backward compatibility. Check the `/health` endpoint to verify authentication status:

```bash
curl http://localhost:8080/health
# Response indicates if auth is enabled
```

### Using API Keys (When Enabled)

When authentication is enabled, include the API key in the Authorization header using Bearer format:

```bash
curl -H "Authorization: Bearer engram_key_{id}_{secret}" \
     -H "Content-Type: application/json" \
     http://localhost:8080/api/v1/memories/recall
```

The API key format is: `engram_key_{id}_{secret}` where:
- `{id}` is the key identifier
- `{secret}` is the cryptographic secret

### Memory Space Access Control

For multi-tenant deployments, specify the memory space using the `X-Memory-Space-Id` header:

```bash
curl -H "Authorization: Bearer engram_key_{id}_{secret}" \
     -H "X-Memory-Space-Id: tenant_42_memories" \
     -H "Content-Type: application/json" \
     http://localhost:8080/api/v1/memories/recall
```

The server validates that the authenticated API key has access to the requested memory space. If the header is omitted, the default space associated with the API key is used.

### Obtaining an API Key

See [Security Configuration](/operations/security.md) for details on generating and managing API keys.

For multi-tenant deployments, each API key is scoped to one or more memory spaces. See [Multi-Tenant Isolation](#multi-tenant-isolation) for details.

### Protected vs Public Endpoints

When authentication is enabled, endpoints are categorized as protected or public:

#### Protected Endpoints (Require Authentication)

These endpoints require valid authentication when auth is enabled:

- `/api/v1/memories/remember` - Store new memories
- `/api/v1/memories/recall` - Retrieve memories
- `/api/v1/memories/forget` - Remove memories
- `/api/v1/memories/recognize` - Recognition checks
- `/api/v1/memories/search` - Search memories
- `/api/v1/episodes/remember` - Store episodes
- `/api/v1/episodes/replay` - Replay episodes
- `/api/v1/spaces` - Memory space management (GET/POST)
- `/api/v1/maintenance/compact` - Maintenance operations
- `/cluster/migrate` - Cluster migration
- `/cluster/rebalance` - Cluster rebalancing
- `/shutdown` - Server shutdown

#### Public Endpoints (Always Accessible)

These endpoints remain accessible without authentication:

- `/health`, `/health/alive`, `/health/spreading` - Health checks
- `/api/v1/system/health` - System health
- `/metrics`, `/metrics/prometheus` - Metrics endpoints
- `/cluster/health`, `/cluster/nodes` - Cluster status
- `/api/v1/system/spreading/config` - Spreading configuration
- `/api/v1/system/introspect` - System introspection
- `/api/v1/stream/*` - Streaming endpoints
- `/api/v1/monitoring/*` - Monitoring endpoints

Public endpoints are designed for load balancers, monitoring systems, and operational visibility.

### Authentication Error Responses

When authentication fails, the API returns standardized error responses:

#### 401 Unauthorized

Returned when authentication is required but missing or invalid:

```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Missing Authorization header",
    "context": "Authentication is required for this endpoint",
    "suggestion": "Include 'Authorization: Bearer engram_key_{id}_{secret}' header"
  }
}
```

Common 401 scenarios:
- Missing Authorization header
- Invalid API key format
- Expired API key
- Unknown API key
- Revoked API key

#### 403 Forbidden

Returned when authenticated but lacking required permissions:

```json
{
  "error": {
    "code": "FORBIDDEN",
    "message": "Access denied to memory space: tenant_42_memories",
    "context": "Your API key does not have access to this memory space",
    "suggestion": "Request access to the memory space or use a different API key"
  }
}
```

Common 403 scenarios:
- Accessing a memory space not in the API key's allowed list
- Missing required permission for an operation
- Attempting privileged operations without admin permissions

### Security Headers

All API responses include security headers for defense-in-depth:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
```

These headers provide protection against common web vulnerabilities even when accessing the API through browsers or web-based clients.

## Core Memory Operations

### POST /api/v1/memories/remember

Store a new memory with confidence scoring.

Think of this as "teaching" Engram something new. Unlike database INSERT operations that assume perfect accuracy, `remember` acknowledges that memories have varying levels of certainty.

#### Request

```bash
curl -X POST http://localhost:8080/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "memory_space_id": "default",
    "memory": {
      "content": "Python was created by Guido van Rossum in 1991",
      "embedding": [0.12, -0.45, 0.78, ...],
      "confidence": {
        "value": 0.95,
        "reasoning": "Verified from authoritative source"
      }
    },
    "auto_link": true,
    "link_threshold": 0.7
  }'

```

#### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `memory_space_id` | string | No | Memory space for multi-tenant isolation. Defaults to authenticated user's space. |
| `memory.content` | string | No | Human-readable memory content |
| `memory.embedding` | float[] | Yes | 768-dimensional semantic embedding vector |
| `memory.confidence.value` | float | Yes | Confidence score [0.0, 1.0] |
| `memory.confidence.reasoning` | string | No | Why this confidence level |
| `auto_link` | boolean | No | Automatically link to related memories (default: true) |
| `link_threshold` | float | No | Similarity threshold for auto-linking (default: 0.7) |

#### Response (201 Created)

```json
{
  "memory_id": "mem_1698765432_a1b2c3d4",
  "storage_confidence": {
    "value": 0.98,
    "category": "CONFIDENCE_CATEGORY_CERTAIN",
    "reasoning": "Successfully stored in hot tier with 3 auto-linked memories"
  },
  "linked_memories": [
    "mem_1698765001_x1y2z3",
    "mem_1698764890_p4q5r6",
    "mem_1698764123_m7n8o9"
  ],
  "initial_state": "CONSOLIDATION_STATE_RECENT"
}

```

#### What Just Happened?

Engram stored your memory and automatically found three related memories based on semantic similarity. These connections enable spreading activation during recall - when you retrieve one memory, related memories become more accessible, just like how your brain works.

The memory starts in "recent" state and will be consolidated during the next dream cycle, potentially forming stronger connections or merging with similar memories.

#### Neo4j Equivalent

```cypher
// Traditional graph database approach
CREATE (m:Memory {
  content: "Python was created by Guido van Rossum in 1991",
  embedding: [0.12, -0.45, 0.78, ...],
  created_at: timestamp()
})
RETURN m

// Must manually create relationships
MATCH (m1:Memory), (m2:Memory)
WHERE id(m1) = $new_id
  AND vectorSimilarity(m1.embedding, m2.embedding) > 0.7
CREATE (m1)-[:RELATED_TO]->(m2)

```

Engram handles relationship creation automatically and includes confidence scoring on both storage and linking operations.

#### Performance

- P50 latency: 12ms

- P99 latency: 45ms

- Throughput: ~80 requests/sec (single instance)

For batch operations, use gRPC streaming to achieve 4.9x better performance.

### POST /api/v1/memories/recall

Retrieve memories using various cue types with spreading activation.

This is where Engram differs most from traditional databases. Instead of exact matching or simple similarity search, recall uses cognitive retrieval patterns with confidence-weighted results.

#### Request

```bash
curl -X POST http://localhost:8080/api/v1/memories/recall \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "memory_space_id": "default",
    "cue": {
      "semantic": {
        "query": "programming language history",
        "fuzzy_threshold": 0.6
      }
    },
    "max_results": 10,
    "include_metadata": true,
    "trace_activation": true
  }'

```

#### Cue Types

Engram supports multiple retrieval cue types, each modeling different cognitive access patterns:

##### Embedding Cue (Direct Similarity)

```json
{
  "cue": {
    "embedding": {
      "vector": [0.15, -0.42, 0.81, ...],
      "similarity_threshold": 0.75
    }
  }
}

```

Like trying to remember something from a visual or sensory trigger.

##### Semantic Cue (Natural Language)

```json
{
  "cue": {
    "semantic": {
      "query": "Python programming",
      "fuzzy_threshold": 0.6,
      "required_tags": ["programming"],
      "excluded_tags": ["deprecated"]
    }
  }
}

```

Like remembering from a verbal description or concept.

##### Context Cue (Episodic)

```json
{
  "cue": {
    "context": {
      "time_start": "2024-01-01T00:00:00Z",
      "time_end": "2024-01-31T23:59:59Z",
      "location": "office",
      "participants": ["Alice", "Bob"]
    }
  }
}

```

Like remembering "what happened last January at the office?"

##### Pattern Cue (Partial Memory)

```json
{
  "cue": {
    "pattern": {
      "fragments": [
        {
          "known_fields": {
            "content": "Python was created by"
          },
          "missing_fields": ["date", "location"],
          "fragment_confidence": {
            "value": 0.8
          }
        }
      ],
      "completion_threshold": 0.5
    }
  }
}

```

Like tip-of-the-tongue experiences where you know part of a memory.

#### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `memory_space_id` | string | No | Memory space identifier |
| `cue` | object | Yes | Retrieval cue (see cue types above) |
| `max_results` | int32 | No | Maximum memories to return (default: 10) |
| `include_metadata` | boolean | No | Include recall statistics (default: false) |
| `trace_activation` | boolean | No | Include spreading activation trace (default: false) |

#### Response (200 OK)

```json
{
  "memories": [
    {
      "id": "mem_1698765432_a1b2c3d4",
      "content": "Python was created by Guido van Rossum in 1991",
      "embedding": [0.12, -0.45, 0.78, ...],
      "activation": 0.92,
      "confidence": {
        "value": 0.95,
        "category": "CONFIDENCE_CATEGORY_CERTAIN",
        "reasoning": "Direct semantic match"
      },
      "last_access": "2024-10-27T10:30:00Z",
      "created_at": "2024-10-15T08:00:00Z",
      "decay_rate": 0.05
    }
  ],
  "recall_confidence": {
    "value": 0.89,
    "category": "CONFIDENCE_CATEGORY_HIGH",
    "reasoning": "Strong cue-to-memory match with spreading activation"
  },
  "metadata": {
    "total_activated": 45,
    "above_threshold": 10,
    "avg_activation": 0.67,
    "recall_time_ms": 23,
    "activation_path": [
      "mem_seed_query",
      "mem_1698765432_a1b2c3d4",
      "mem_1698765001_x1y2z3"
    ]
  },
  "traces": [
    {
      "memory_id": "mem_1698765432_a1b2c3d4",
      "activation_level": 0.92,
      "activation_path": ["query", "mem_1698765432_a1b2c3d4"],
      "timestamp": "2024-10-27T10:30:00.123Z"
    }
  ]
}

```

#### Understanding Confidence vs Activation

- **Confidence**: How certain we are that this memory is accurate (intrinsic property)

- **Activation**: How accessible this memory is right now (dynamic, context-dependent)

A memory can have high confidence (95% sure it's true) but low activation (0.3 - hard to access). Conversely, a recent but uncertain memory might have low confidence (0.6) but high activation (0.95).

#### Performance

- P50 latency: 35ms

- P99 latency: 120ms

- Throughput: ~45 requests/sec

Spreading activation adds 10-50ms depending on graph connectivity.

### POST /api/v1/memories/forget

Remove or suppress memories with configurable forgetting modes.

Unlike database DELETE operations, forgetting in Engram can be reversible or gradual, modeling how biological forgetting actually works.

#### Request

```bash
curl -X POST http://localhost:8080/api/v1/memories/forget \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "memory_space_id": "default",
    "memory_id": "mem_1698765432_a1b2c3d4",
    "mode": "FORGET_MODE_SUPPRESS"
  }'

```

#### Forget Modes

| Mode | Reversible | Effect | Use Case |
|------|-----------|--------|----------|
| `FORGET_MODE_SUPPRESS` | Yes | Reduce activation, increase decay | Temporarily downrank memories |
| `FORGET_MODE_DELETE` | No | Permanent removal | GDPR compliance, data cleanup |
| `FORGET_MODE_OVERWRITE` | Partial | Replace with new content | Memory reconsolidation |

#### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `memory_space_id` | string | No | Memory space identifier |
| `memory_id` | string | Yes* | Specific memory to forget |
| `pattern` | object | Yes* | Pattern of memories to forget |
| `mode` | enum | Yes | Forgetting mode (see above) |

*Either `memory_id` or `pattern` required, not both.

#### Response (200 OK)

```json
{
  "memories_affected": 1,
  "forget_confidence": {
    "value": 0.95,
    "category": "CONFIDENCE_CATEGORY_CERTAIN",
    "reasoning": "Memory suppressed, activation reduced to 0.1"
  },
  "reversible": true
}

```

#### Pattern Forgetting

Forget multiple memories matching a pattern:

```json
{
  "memory_space_id": "default",
  "pattern": {
    "semantic": {
      "query": "outdated Python 2 information"
    }
  },
  "mode": "FORGET_MODE_SUPPRESS"
}

```

This reduces activation for all memories matching the pattern, useful for deprecating old information without deleting it permanently.

### POST /api/v1/memories/recognize

Check if a memory pattern feels familiar (recognition vs recall).

Recognition is cognitively easier than recall - it's the difference between "Have I seen this before?" versus "What did I learn yesterday?"

#### Request

```bash
curl -X POST http://localhost:8080/api/v1/memories/recognize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "memory_space_id": "default",
    "content": "Python was created by Guido van Rossum",
    "recognition_threshold": 0.8
  }'

```

#### Response (200 OK)

```json
{
  "recognized": true,
  "recognition_confidence": {
    "value": 0.94,
    "category": "CONFIDENCE_CATEGORY_HIGH",
    "reasoning": "Strong match with existing memory (similarity: 0.96)"
  },
  "similar_memories": [
    {
      "id": "mem_1698765432_a1b2c3d4",
      "content": "Python was created by Guido van Rossum in 1991",
      "similarity": 0.96
    }
  ],
  "familiarity_score": 0.94
}

```

Use recognition for:

- Deduplication before storing new memories

- "Have we seen this before?" checks

- Validating memory accuracy

## Episodic Operations

Episodic memories encode rich contextual information following the what/when/where/who/why/how structure from memory research.

### POST /api/v1/episodes/experience

Record a new episodic memory with full context.

#### Request

```bash
curl -X POST http://localhost:8080/api/v1/episodes/experience \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "memory_space_id": "default",
    "episode": {
      "what": "Learned about memory consolidation during sleep",
      "when": "2024-10-27T09:00:00Z",
      "where_location": "neuroscience lecture hall",
      "who": ["Dr. Smith", "classmates"],
      "why": "preparing for exam",
      "how": "attended lecture and took notes",
      "emotional_valence": 0.6,
      "importance": 0.8,
      "embedding": [0.15, -0.42, ...]
    },
    "immediate_consolidation": false,
    "context_links": ["mem_related_concept_1", "mem_related_concept_2"]
  }'

```

#### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `episode.what` | string | Yes | Core event description |
| `episode.when` | timestamp | Yes | When it occurred |
| `episode.where_location` | string | No | Where it occurred |
| `episode.who` | string[] | No | Who was involved |
| `episode.why` | string | No | Why it happened |
| `episode.how` | string | No | How it happened |
| `episode.emotional_valence` | float | No | Emotion [-1.0 to 1.0] |
| `episode.importance` | float | No | Subjective importance [0.0 to 1.0] |
| `episode.embedding` | float[] | Yes | 768-dimensional semantic vector |
| `immediate_consolidation` | boolean | No | Consolidate immediately (default: false) |
| `context_links` | string[] | No | Link to contextual memories |

#### Response (201 Created)

```json
{
  "episode_id": "ep_1698765432_x1y2z3",
  "encoding_quality": {
    "value": 0.92,
    "category": "CONFIDENCE_CATEGORY_HIGH",
    "reasoning": "Rich contextual encoding with 5/6 fields populated"
  },
  "state": "CONSOLIDATION_STATE_RECENT",
  "context_links_created": 2
}

```

#### Why Episodic Encoding Matters

Research shows episodic memories with rich context have:

- 67% better retrieval success

- 3x longer retention periods

- More resistant to interference

The what/when/where/who/why/how structure follows Tulving's episodic memory theory.

### POST /api/v1/episodes/reminisce

Retrieve episodic memories using contextual cues.

#### Request

```bash
curl -X POST http://localhost:8080/api/v1/episodes/reminisce \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "memory_space_id": "default",
    "about_topic": "neuroscience lectures",
    "time_window_start": "2024-10-01T00:00:00Z",
    "time_window_end": "2024-10-31T23:59:59Z",
    "at_location": "lecture hall",
    "include_emotional": true
  }'

```

#### Response (200 OK)

```json
{
  "episodes": [
    {
      "id": "ep_1698765432_x1y2z3",
      "when": "2024-10-27T09:00:00Z",
      "what": "Learned about memory consolidation during sleep",
      "where_location": "neuroscience lecture hall",
      "who": ["Dr. Smith", "classmates"],
      "encoding_confidence": {
        "value": 0.92,
        "category": "CONFIDENCE_CATEGORY_HIGH"
      }
    }
  ],
  "recall_vividness": {
    "value": 0.87,
    "category": "CONFIDENCE_CATEGORY_HIGH",
    "reasoning": "Recent memories with strong contextual cues"
  },
  "emotional_summary": {
    "mean_valence": 0.55,
    "valence_variance": 0.12,
    "dominant_emotion": "engaged"
  },
  "memory_themes": [
    "learning",
    "neuroscience",
    "academic",
    "preparation"
  ]
}

```

#### Emotional Summary

When `include_emotional: true`, Engram aggregates emotional valence across episodes:

- **mean_valence**: Average emotion [-1.0 to 1.0]

- **valence_variance**: How much emotions varied

- **dominant_emotion**: Inferred emotional category

#### Memory Themes

Themes emerge from clustering episode content, not from explicit tags. This models how your brain naturally categorizes experiences.

## Consolidation Operations

Memory consolidation transforms recent, fragile memories into stable long-term storage, following sleep research.

### POST /api/v1/consolidation/consolidate

Trigger memory consolidation process.

#### Request

```bash
curl -X POST http://localhost:8080/api/v1/consolidation/consolidate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "memory_space_id": "default",
    "criteria": {
      "older_than": "2024-10-26T00:00:00Z",
      "importance_threshold": 0.5,
      "types": ["MEMORY_TYPE_EPISODIC", "MEMORY_TYPE_SEMANTIC"]
    },
    "mode": "CONSOLIDATION_MODE_SYNAPTIC"
  }'

```

#### Consolidation Modes

| Mode | Speed | Scope | Use Case |
|------|-------|-------|----------|
| `CONSOLIDATION_MODE_SYNAPTIC` | Fast (seconds) | Local connections | Recent memories, quick strengthening |
| `CONSOLIDATION_MODE_SYSTEMS` | Slow (minutes) | Distributed patterns | Deep integration, schema formation |
| `CONSOLIDATION_MODE_RECONSOLIDATION` | Medium | Existing memories | Update/modify stable memories |

#### Response (200 OK)

```json
{
  "memories_consolidated": 342,
  "new_associations": 89,
  "state_changes": {
    "mem_1698765432_a1b2c3d4": "CONSOLIDATION_STATE_CONSOLIDATED",
    "mem_1698765001_x1y2z3": "CONSOLIDATION_STATE_CONSOLIDATED"
  },
  "next_consolidation": "2024-10-27T18:00:00Z"
}

```

#### When to Consolidate

- **Nightly**: Systems consolidation during low-traffic periods (like sleep)

- **Hourly**: Synaptic consolidation for recent important memories

- **On-demand**: After batch imports or major knowledge updates

### POST /api/v1/consolidation/dream

Stream dream-like memory replay for insight generation.

This endpoint streams consolidation events in real-time, showing memory replay sequences and emerging insights. Use Server-Sent Events (SSE) for browser consumption.

#### Request

```bash
curl -X POST http://localhost:8080/api/v1/consolidation/dream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Accept: text/event-stream" \
  -d '{
    "memory_space_id": "default",
    "replay_cycles": 10,
    "creativity_factor": 0.7,
    "generate_insights": true
  }'

```

#### Response (200 OK, streaming)

```
event: replay
data: {"memory_ids": ["mem_1", "mem_2", "mem_3"], "sequence_novelty": 0.65, "narrative": "Connection between Python history and memory systems"}

event: insight
data: {"description": "Both Python and Engram prioritize developer experience", "connected_memories": ["mem_1", "mem_5"], "confidence": {"value": 0.78}, "suggested_action": "Create explicit link"}

event: progress
data: {"memories_replayed": 25, "new_connections": 7, "consolidation_strength": 0.82}

event: replay
data: {"memory_ids": ["mem_4", "mem_6"], "sequence_novelty": 0.89, "narrative": "Novel connection between unrelated concepts"}

event: done
data: {"total_replayed": 50, "total_insights": 3, "total_connections": 12}

```

#### Stream Event Types

- **replay**: Memory sequence being replayed

- **insight**: Novel connection discovered

- **progress**: Consolidation progress update

- **done**: Stream completion summary

#### Creativity Factor

Controls how much dream replay explores novel vs familiar connections:

- `0.0`: Conservative, strengthen existing connections

- `0.5`: Balanced exploration and exploitation

- `1.0`: Creative, maximize novel connections

Higher creativity can discover unexpected insights but may create spurious associations.

## Pattern Operations

Pattern completion reconstructs full memories from partial cues, modeling tip-of-the-tongue experiences.

### POST /api/v1/patterns/complete

Complete partial memory patterns.

#### Request

```bash
curl -X POST http://localhost:8080/api/v1/patterns/complete \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "memory_space_id": "default",
    "partial_pattern": {
      "fragments": [
        {
          "known_fields": {
            "content": "Python was created by"
          },
          "missing_fields": ["date", "creator_nationality"],
          "fragment_confidence": {
            "value": 0.8
          }
        }
      ],
      "completion_threshold": 0.7
    },
    "creativity": 0.3,
    "max_completions": 5
  }'

```

#### Response (200 OK)

```json
{
  "completions": [
    {
      "id": "completion_1",
      "content": "Python was created by Guido van Rossum in 1991",
      "embedding": [0.12, -0.45, ...],
      "confidence": {
        "value": 0.94,
        "category": "CONFIDENCE_CATEGORY_HIGH",
        "reasoning": "Strong match with existing memory"
      },
      "completed_fields": {
        "date": "1991",
        "creator_nationality": "Dutch"
      }
    }
  ],
  "completion_confidence": {
    "value": 0.91,
    "reasoning": "High-confidence completion from established memories"
  },
  "field_confidences": {
    "date": 0.95,
    "creator_nationality": 0.87
  }
}

```

### POST /api/v1/patterns/associate

Create or strengthen associations between memories.

#### Request

```bash
curl -X POST http://localhost:8080/api/v1/patterns/associate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "memory_space_id": "default",
    "source_memory": "mem_python_creation",
    "target_memory": "mem_open_source_movement",
    "association_type": "ASSOCIATION_TYPE_TEMPORAL",
    "strength": 0.75,
    "reason": "Both emerged in early 1990s"
  }'

```

#### Association Types

| Type | Meaning | Example |
|------|---------|---------|
| `ASSOCIATION_TYPE_SEMANTIC` | Shared meaning | "Python" ↔ "programming language" |
| `ASSOCIATION_TYPE_TEMPORAL` | Time-based | Events in same period |
| `ASSOCIATION_TYPE_CAUSAL` | Cause-effect | "studied" → "passed exam" |
| `ASSOCIATION_TYPE_SPATIAL` | Location-based | "coffee shop" ↔ "morning routine" |
| `ASSOCIATION_TYPE_EMOTIONAL` | Emotion-based | Memories with similar valence |

#### Response (200 OK)

```json
{
  "created": true,
  "final_strength": 0.78,
  "affected_paths": [
    "mem_python_creation → mem_open_source_movement",
    "mem_python_creation → mem_free_software → mem_open_source_movement"
  ]
}

```

## Monitoring Operations

### GET /api/v1/introspect

Get system self-awareness and statistics.

#### Request

```bash
curl -X GET "http://localhost:8080/api/v1/introspect?memory_space_id=default&include_health=true&include_statistics=true" \
  -H "Authorization: Bearer ${API_KEY}"

```

#### Response (200 OK)

```json
{
  "metrics": {
    "total_memories": 125834,
    "avg_activation": 0.42,
    "avg_confidence": 0.78,
    "consolidation_pending": 342,
    "associations_count": 456789,
    "storage_tier_distribution": {
      "hot": 0.15,
      "warm": 0.35,
      "cold": 0.50
    }
  },
  "health": {
    "healthy": true,
    "components": {
      "storage": {
        "operational": true,
        "performance": 0.95,
        "status_message": "All tiers responding normally"
      },
      "graph_engine": {
        "operational": true,
        "performance": 0.88,
        "status_message": "HNSW index healthy, 125k nodes"
      },
      "consolidation": {
        "operational": true,
        "performance": 0.92,
        "status_message": "342 memories pending consolidation"
      }
    },
    "summary": "System healthy, all components operational"
  },
  "statistics": {
    "total_memories": 125834,
    "by_type": {
      "MEMORY_TYPE_EPISODIC": 45678,
      "MEMORY_TYPE_SEMANTIC": 80156
    },
    "avg_activation": 0.42,
    "avg_confidence": 0.78,
    "total_associations": 456789,
    "oldest_memory": "2024-01-15T10:00:00Z",
    "newest_memory": "2024-10-27T10:30:00Z"
  },
  "active_processes": [
    "consolidation_daemon",
    "decay_processor",
    "activation_spreader"
  ]
}

```

### GET /api/v1/stream/events

Stream real-time memory activity (Server-Sent Events).

#### Request

```bash
curl -X GET "http://localhost:8080/api/v1/stream/events?memory_space_id=default&event_types=ACTIVATION,STORAGE,RECALL&min_importance=0.5" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Accept: text/event-stream"

```

#### Response (streaming)

```
event: storage
data: {"event_type": "STREAM_EVENT_TYPE_STORAGE", "timestamp": "2024-10-27T10:30:15.123Z", "description": "New memory stored: mem_1698765432", "metadata": {"memory_id": "mem_1698765432", "confidence": "0.95"}, "importance": 0.8}

event: activation
data: {"event_type": "STREAM_EVENT_TYPE_ACTIVATION", "timestamp": "2024-10-27T10:30:16.456Z", "description": "Spreading activation from mem_1698765432", "metadata": {"activated_count": "23", "max_activation": "0.87"}, "importance": 0.6}

event: recall
data: {"event_type": "STREAM_EVENT_TYPE_RECALL", "timestamp": "2024-10-27T10:30:17.789Z", "description": "Memory recalled: mem_1698765001", "metadata": {"query": "Python history", "confidence": "0.89"}, "importance": 0.7}

```

## Error Handling

All errors follow a consistent structure with educational messages. See [Error Codes Reference](/reference/error-codes.md) for complete catalog.

### Error Response Format

```json
{
  "error": {
    "code": "ERR-2003",
    "summary": "Activation spreading timeout",
    "context": "Spreading activation did not converge within 5000ms. Started from 1 seed memory, reached 45,123 nodes before timeout.",
    "suggestion": "Reduce max_hops (currently: unlimited) or increase activation_decay (currently: 0.1). For broad queries, consider using pattern completion instead of full spreading.",
    "example": {
      "current": "recall(cue, spread_activation=true)",
      "recommended": "recall(cue, spread_activation=true, max_hops=3, activation_decay=0.3)"
    },
    "error_confidence": {
      "value": 0.95,
      "reasoning": "Timeout threshold exceeded, activation log shows divergence pattern"
    },
    "similar_errors": [
      "ERR-2004: Invalid similarity threshold",
      "ERR-5001: Storage tier unavailable"
    ]
  }
}

```

### Common Errors

| HTTP Status | Error Code | Meaning | Action |
|-------------|------------|---------|--------|
| 400 | ERR-1001 | Embedding dimension mismatch | Verify embedding size is 768 |
| 400 | ERR-1002 | Invalid confidence value | Ensure confidence in [0.0, 1.0] |
| 404 | ERR-1003 | Memory space not found | Check memory_space_id |
| 409 | ERR-3001 | Consolidation in progress | Wait for completion or cancel |
| 429 | ERR-5003 | Rate limit exceeded | Implement exponential backoff |
| 503 | ERR-5004 | Service unavailable | Check system health via /introspect |

## Rate Limiting

Rate limits protect system performance and ensure fair resource allocation.

### Limits by Endpoint

| Endpoint | Limit | Window | Burst |
|----------|-------|--------|-------|
| `/memories/remember` | 100/sec | 1s | 150 |
| `/memories/recall` | 50/sec | 1s | 75 |
| `/consolidation/consolidate` | 10/min | 1m | 15 |
| `/consolidation/dream` | 5/min | 1m | 5 |
| `/introspect` | 20/sec | 1s | 30 |

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1698765432

```

When rate limited:

```json
{
  "error": {
    "code": "ERR-5003",
    "summary": "Rate limit exceeded",
    "context": "100 requests in last second (limit: 100/sec)",
    "suggestion": "Implement exponential backoff or use gRPC streaming for batch operations",
    "retry_after_seconds": 1
  }
}

```

## Performance Characteristics

### Latency Targets (P99)

| Operation | Target | Actual | Notes |
|-----------|--------|--------|-------|
| Remember (single) | <50ms | 45ms | Hot tier storage |
| Recall (10 results) | <150ms | 120ms | With spreading activation |
| Forget (single) | <30ms | 25ms | Suppression mode |
| Consolidate (1000 memories) | <5s | 4.2s | Synaptic mode |
| Pattern complete | <100ms | 85ms | Single completion |

### Throughput (single instance)

| Operation | Throughput | Notes |
|-----------|-----------|-------|
| Remember | 80/sec | REST unary |
| Recall | 45/sec | REST unary |
| Remember (gRPC stream) | 390/sec | 4.9x faster |
| Recall (gRPC stream) | 160/sec | 3.6x faster |

### When to Use gRPC

Switch to [gRPC API](/reference/grpc-api.md) when:

- Throughput requirement >100 req/sec

- Batch operations (>10 memories)

- Streaming consolidation needed

- Mobile/edge deployments (binary efficiency)

## Multi-Tenant Isolation

Each API key is scoped to a memory space for tenant isolation.

### Explicit Memory Space

```bash
curl -X POST http://localhost:8080/api/v1/memories/remember \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "memory_space_id": "tenant_42_memories",
    "memory": {...}
  }'

```

Server validates JWT claim matches `memory_space_id`. Returns `ERR-4004` if mismatch.

### Default Memory Space

Omit `memory_space_id` to use authenticated user's default space:

```bash
curl -X POST http://localhost:8080/api/v1/memories/remember \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "memory": {...}
  }'

```

## Next Steps

- **Learning**: Try the [15-minute API Quickstart](/tutorials/api-quickstart.md)

- **Performance**: Read [gRPC API Reference](/reference/grpc-api.md) for high-throughput

- **Troubleshooting**: See [Error Codes Catalog](/reference/error-codes.md)

- **Examples**: Explore [Multi-Language Examples](/reference/api-examples/)

- **Operations**: Check [Production Operations Guide](/operations/)

## Neo4j Migration Guide

Coming from Neo4j? Here's how Engram concepts map:

| Neo4j | Engram | Key Difference |
|-------|--------|----------------|
| `CREATE (n:Node)` | `POST /memories/remember` | Automatic relationship creation |
| `MATCH (n) WHERE ...` | `POST /memories/recall` | Confidence scoring, not exact match |
| `DELETE (n)` | `POST /memories/forget` | Gradual/reversible forgetting |
| `MERGE` | `POST /memories/recognize` then `remember` | Recognition check first |
| Cypher query | Natural language cue | Probabilistic vs deterministic |
| Relationship | Association with type | Typed, weighted, confidence-scored |

See [Database Migration Guide](/operations/database-migration.md) for detailed migration examples.
