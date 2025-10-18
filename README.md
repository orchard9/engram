# Engram

A high-performance cognitive graph database with biologically-inspired memory consolidation and probabilistic query processing.

## Overview

Engram is not another graph database - it's a **cognitive memory system** that thinks like you do. Combining cognitive science principles with modern systems engineering, Engram features:

- **Spreading Activation**: Memories activate related memories, just like human thought
- **Probabilistic Queries**: Every result includes confidence intervals and uncertainty tracking
- **Temporal Dynamics**: Every memory carries precise timestamps with configurable decay following psychological research
- **Graceful Degradation**: Robust error handling with automatic recovery strategies
- **Lock-Free Concurrency**: SIM D-optimized operations and zero-copy data structures

## Quick Start

See [quickstart.md](quickstart.md) for a 60-second guide to your first memory.

```bash
# Start server
./target/debug/engram start

# Store a memory
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "The mitochondria is the powerhouse of the cell", "confidence": 0.95, "timestamp": "2024-01-04T08:15:00Z"}'

# Recall it (notice spreading activation finds it with partial info)
curl http://localhost:7432/api/v1/memories/recall?query=mitochondria
```

## Installation

### Prerequisites

- Rust 1.75+ (Edition 2024)
- For SMT verification features: Z3 SMT Solver

### macOS Setup (Z3)

```bash
brew install z3

# Add to your shell profile (.zshrc, .bash_profile, etc.):
export Z3_SYS_Z3_HEADER="/opt/homebrew/include/z3.h"
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
export BINDGEN_EXTRA_CLANG_ARGS="-I/opt/homebrew/include"
```

### Building

```bash
git clone https://github.com/orchard9/engram.git
cd engram
cargo build --release

# Run tests
cargo test --workspace
```

## Architecture

### Core Components

- **`engram-core`**: Cognitive graph engine with spreading activation
- **`engram-cli`**: HTTP/gRPC server and command-line interface
- **`engram-storage`**: Tiered storage with NUMA awareness

### Key Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Spreading Activation** | ✅ Production | Neural-inspired memory retrieval through graph propagation |
| **Probabilistic Queries** | ✅ Production | Confidence intervals with uncertainty quantification |
| **HNSW Index** | ✅ Production | High-performance approximate nearest neighbor search |
| **Psychological Decay** | ✅ Production | Ebbinghaus forgetting curves, spaced repetition |
| **Pattern Completion** | ✅ Beta | Reconstruct missing details from partial memories |
| **Memory Consolidation** | 🚧 In Development | Episodic→semantic transformation (Milestone 6) |
| **SMT Verification** | ✅ Production | Correctness proofs for probability propagation |
| **Streaming Monitoring** | ✅ Production | Real-time SSE streams of memory dynamics |

## What Makes Engram Different?

### Traditional Databases
```sql
SELECT * FROM memories WHERE content LIKE '%mitochondria%';
-- Returns: Exact matches only
-- Confidence: None (binary yes/no)
-- Related memories: Must explicitly join
```

### Engram
```bash
curl http://localhost:7432/api/v1/memories/recall?query=mitochondria
```
Returns:
- **Direct matches** with confidence scores
- **Associated memories** via spreading activation (e.g., "cellular respiration")
- **Reconstructed memories** from partial information
- **Confidence intervals** showing uncertainty
- **Activation paths** explaining how memories were found

## Development

### Running the Server

```bash
# Start with default configuration
./target/debug/engram start

# Check status
./target/debug/engram status

# Stop gracefully
./target/debug/engram stop
```

### CLI Commands

```bash
# Probabilistic query with confidence intervals
./target/debug/engram query "cellular biology" --format table

# Memory operations
./target/debug/engram memory create "ATP synthase generates energy"
./target/debug/engram memory search "energy"
./target/debug/engram memory list

# Configuration
./target/debug/engram config set feature_flags.spreading_api_beta true
./target/debug/engram config get feature_flags
```

### API Documentation

- Interactive docs: http://localhost:7432/docs
- OpenAPI spec: http://localhost:7432/api-docs/openapi.json
- Architecture docs: [vision.md](vision.md)
- Roadmap: [roadmap/](roadmap/)

#### Consolidated Beliefs (Upcoming)

Consolidation runs asynchronously after `remember` writes, transforming episodic memories into semantic beliefs with complete provenance trails. We are stabilizing a dedicated `/api/v1/consolidations/{pattern_id}` endpoint that will surface:

- the synthesized belief (semantic pattern) and its schema-level confidence
- `source_episodes` citations with the timestamps and confidence captured during replay
- decay and reinforcement signals so you can trace why a belief strengthened or weakened

Until that endpoint ships, `remember` responses include a `consolidation_state` hint. Subscribe to `/api/v1/stream/consolidation` to receive notifications when new beliefs are available.

## Testing

Comprehensive test coverage:

```bash
# All tests
cargo test --workspace

# Specific test suites
cargo test --test probabilistic_api_tests
cargo test --test http_api_tests
cargo test --lib activation::parallel::tests

# With features
cargo test --features "hnsw_index,psychological_decay"
```

## Contributing

1. Read [coding_guidelines.md](coding_guidelines.md)
2. Check [current milestones](roadmap/)
3. Run `make quality` (zero clippy warnings required)
4. Follow error handling patterns (see below)
5. Submit PRs with clear descriptions

### Error Handling

Engram uses comprehensive error recovery:

```rust
use engram_core::error::{EngramError, RecoveryStrategy, ErrorRecovery};

// Automatic retry with exponential backoff
let result = ErrorRecovery::with_retry(
    || async { store.write_episode(&episode) },
    RecoveryStrategy::Retry {
        max_attempts: 3,
        backoff_ms: 100,
    },
).await?;

// Graceful fallback
let result = ErrorRecovery::with_fallback(
    || hnsw_search(&query),      // Try HNSW first
    || linear_search(&query),    // Fallback to linear
)?;
```

## Troubleshooting

### Build Issues

```bash
# macOS Z3 errors
brew install z3
# Then set environment variables (see macOS Setup above)

# Clean build
cargo clean && cargo build

# Check Rust version
rustc --version  # Should be 1.75+
```

### Performance Tuning

For optimal performance:
- Use `--release` builds in production
- Enable appropriate features for your use case
- Consider NUMA topology for large deployments
- Monitor with `curl http://localhost:7432/metrics`

## License

[License information]

---

## Real-World Scenario: Why Engram?

Imagine you're building a **research assistant** that helps scientists explore literature. Traditional databases fall short because:

### The Problem with Traditional Approaches

**Vector Database (e.g., Pinecone):**
```python
# Query for "CRISPR gene editing"
results = vector_db.query("CRISPR gene editing", top_k=10)
# Returns: Top 10 most similar papers
# Missing: WHY they're related, confidence in matches, temporal context
```

**Graph Database (e.g., Neo4j):**
```cypher
MATCH (p:Paper)-[:CITES]->(related:Paper)
WHERE p.title CONTAINS "CRISPR"
RETURN related
// Returns: Papers explicitly cited
// Missing: Implicit connections, confidence, semantic similarity
```

### The Engram Way

```bash
# Store research papers with context
curl -X POST http://localhost:7432/api/v1/episodes/remember -d '{
  "what": "CRISPR-Cas9 enables precise genome editing in human cells",
  "when": "2023-03-15T10:00:00Z",
  "where": "Nature Biotechnology",
  "confidence": 0.95
}'

curl -X POST http://localhost:7432/api/v1/episodes/remember -d '{
  "what": "Base editing allows single nucleotide changes without double-strand breaks",
  "when": "2023-06-20T14:30:00Z",
  "where": "Cell",
  "confidence": 0.90
}'

curl -X POST http://localhost:7432/api/v1/episodes/remember -d '{
  "what": "Prime editing achieves targeted insertions and deletions",
  "when": "2023-09-10T09:15:00Z",
  "where": "Science",
  "confidence": 0.88
}'

# Now query with partial information
curl "http://localhost:7432/api/v1/query/probabilistic?query=genome%20editing&include_evidence=true&include_uncertainty=true"
```

**What You Get:**

```json
{
  "memories": [
    {
      "content": "CRISPR-Cas9 enables precise genome editing...",
      "confidence": {"value": 0.95, "category": "High"},
      "activation_level": 1.0,
      "similarity_score": 0.92,
      "retrieval_path": "Direct match + spreading activation"
    },
    {
      "content": "Base editing allows single nucleotide changes...",
      "confidence": {"value": 0.90, "category": "High"},
      "activation_level": 0.78,
      "similarity_score": 0.85,
      "retrieval_path": "Spreading activation (1 hop from CRISPR)"
    },
    {
      "content": "Prime editing achieves targeted insertions...",
      "confidence": {"value": 0.88, "category": "High"},
      "activation_level": 0.65,
      "similarity_score": 0.81,
      "retrieval_path": "Spreading activation (2 hops from genome editing)"
    }
  ],
  "confidence_interval": {
    "point": 0.87,
    "lower": 0.82,
    "upper": 0.91,
    "width": 0.09
  },
  "evidence_chain": [
    {
      "source_type": "direct_match",
      "strength": 0.95,
      "description": "Query 'genome editing' directly matches stored content"
    },
    {
      "source_type": "spreading_activation",
      "strength": 0.82,
      "description": "Related concepts activated through semantic network"
    }
  ],
  "uncertainty_sources": [
    {
      "source_type": "spreading_activation_noise",
      "impact": 0.05,
      "explanation": "Activation spreading variance 0.050 with path diversity 0.100"
    }
  ]
}
```

### Why This Matters

1. **Confidence Tracking**: Know exactly how certain each result is
2. **Spreading Activation**: Find related papers even without explicit citations
3. **Evidence Chain**: Understand WHY each result was returned
4. **Uncertainty Quantification**: See confidence intervals, not just point estimates
5. **Temporal Decay**: Recent papers naturally rank higher (configurable)
6. **Probabilistic Queries**: Ask vague questions, get calibrated answers

### 6 Months Later...

```bash
# Query again - notice temporal decay
curl "http://localhost:7432/api/v1/memories/recall?query=CRISPR"
```

The **March paper** now has `confidence: 0.82` (decayed from 0.95) while the **September paper** maintains `confidence: 0.87` - reflecting the psychological reality that recent memories are more accessible.

### The Novel Part

Unlike vector databases (pure similarity) or graph databases (explicit relationships), Engram gives you:

- **Implicit connections** through spreading activation
- **Confidence calibration** showing uncertainty
- **Temporal dynamics** matching human memory
- **Evidence trails** explaining each result
- **Probabilistic semantics** for vague queries

**Perfect for:** Research assistants, personal knowledge bases, chatbot memory, diagnostic systems, recommendation engines where *confidence matters*.

Try it yourself: [quickstart.md](quickstart.md)
