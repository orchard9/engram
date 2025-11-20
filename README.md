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

## Novelty Showcase Demo

Want to see what makes Engram genuinely different? Run the 8-minute interactive demo showcasing:

- **Psychological Decay** - Ebbinghaus forgetting curves at the storage layer
- **Spreading Activation** - Associative memory retrieval through neural dynamics
- **Memory Consolidation** - Automatic pattern learning from experiences

```bash
cd demos/novelty-showcase
./demo.sh
```

The demo includes:
- Live comparison with traditional databases
- Real-time spreading activation visualization
- Automatic pattern extraction from episodes
- Key performance metrics and competitive positioning

See [demos/novelty-showcase/README.md](demos/novelty-showcase/README.md) for details.

## Installation

### Prerequisites

- Rust 1.75+ (Edition 2024)
- For SMT verification features: Z3 SMT Solver
- Optional: Zig 0.13.0 for performance kernels (15-35% faster operations)

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

### System Requirements

- **Rust**: 1.82.0 or newer (required for Edition 2024)
- **System libraries**: hwloc, pkg-config, libudev (Linux)
- **Optional**: CUDA 11.0+ for GPU acceleration

See [docs/reference/system-requirements.md](docs/reference/system-requirements.md) for detailed requirements and troubleshooting.

### Building

```bash
git clone https://github.com/orchard9/engram.git
cd engram
cargo build --release

# Run tests
cargo test --workspace
```

### Building with Zig Performance Kernels (Optional)

For 15-35% performance improvements on compute-intensive operations:

```bash
# Install Zig 0.13.0
brew install zig  # macOS
# Or download from https://ziglang.org/download/

# Build with performance kernels
./scripts/build_with_zig.sh release

# Verify Zig kernels are active
nm target/release/engram-cli | grep engram_vector_similarity
```

See [Zig Performance Kernels Operations Guide](docs/operations/zig_performance_kernels.md) for deployment instructions.

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
| **Query Language** | ✅ Production | SQL-like syntax for RECALL, SPREAD, CONSOLIDATE, COMPLETE, IMAGINE operations |
| **HNSW Index** | ✅ Production | High-performance approximate nearest neighbor search |
| **Psychological Decay** | ✅ Production | Ebbinghaus forgetting curves, spaced repetition, spacing effect |
| **Pattern Completion** | ✅ Production | Reconstruct missing details from partial memories (CA3/CA1 hippocampal model) |
| **Memory Consolidation** | ✅ Production | Asynchronous episodic→semantic transformation with pattern detection |
| **Cognitive Patterns** | ✅ Production | Semantic priming, interference effects, reconsolidation, false memory validation |
| **Memory Interference** | ✅ Production | Proactive/retroactive interference, fan effect (Anderson 1974, McGeoch 1942) |
| **Reconsolidation** | ✅ Production | Memory restabilization after recall (Nader 2000) |
| **GPU Acceleration** | ✅ Production | CUDA kernels for vector similarity and activation spreading (10-50x speedup) |
| **SMT Verification** | ✅ Production | Correctness proofs for probability propagation |
| **Streaming Monitoring** | ✅ Production | Real-time SSE streams of memory dynamics |
| **Memory Spaces** | ✅ Production | Multi-tenant isolation with per-space metrics |
| **Cluster (SWIM)** | ✅ Production | Distributed cluster with gossip membership, replication factor=2, DNS/static discovery |
| **Zig Performance Kernels** | ✅ Production | SIMD-accelerated operations (15-35% faster) |
| **Zero-Overhead Metrics** | ✅ Production | Sub-1% monitoring overhead with feature flags |

## Project Status

**Current Phase**: Milestone 17 (Dual Memory Architecture)
**Completed Milestones**: M0-M13, M15-M16, M17 (partial) - 17/20 milestones
**Test Health**: 100% (all tests passing, zero clippy warnings)
**Production Ready**: Single-node AND cluster deployment validated

**Recent Achievements** (2025-11-20):
- M7 Memory Spaces: 100% complete - Multi-tenant isolation with <5% overhead
- M8 Pattern Completion: Metrics integration complete
- M11 Cluster: 100% complete - SWIM-based distributed architecture production-ready
- M17 Dual Memory: In progress - Hierarchical spreading and System 1/2 integration
- Cluster Deployment: Docker Compose and Kubernetes validated with comprehensive cookbook

**Deployment Options**:
- Single-node: Production ready
- Docker Compose: 3-node cluster with monitoring (validated)
- Kubernetes: StatefulSet with DNS discovery (validated with Kind)

**Next Steps**:
- M17 completion: Dual memory architecture with hierarchical spreading
- M18-M20: Advanced cognitive features and optimization

See [roadmap/](roadmap/) for milestone tracking.

### Memory Spaces (Multi-Tenancy)

Engram supports isolated **memory spaces** for multi-tenant deployments. Each space maintains separate:
- Memory storage and persistence
- Spreading activation graphs
- Health metrics and diagnostics
- WAL (Write-Ahead Log) and tier storage

#### Using Memory Spaces

**HTTP API** - Use the `X-Memory-Space` header:

```bash
# Create memory in "research" space
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -H "X-Memory-Space: research" \
  -d '{"content": "CRISPR gene editing", "confidence": 0.95}'

# Recall from "research" space only
curl -H "X-Memory-Space: research" \
  http://localhost:7432/api/v1/memories/recall?query=CRISPR
```

**CLI Commands:**

```bash
# List all memory spaces
./target/debug/engram space list

# Create a new space
./target/debug/engram space create production

# Check per-space health metrics
./target/debug/engram status --space production
```

**Default Space:** Without specifying a space, all operations use the `default` space, ensuring backward compatibility with existing deployments.

**Isolation Guarantees:**
- ✅ Storage: Each space gets dedicated directory structure
- ✅ Persistence: Isolated WAL and tier storage per space
- ✅ Metrics: Per-space health, pressure, and consolidation tracking
- ✅ Concurrency: Thread-safe concurrent access across spaces

For migration guidance and advanced configuration, see [docs/operations/memory-space-migration.md](docs/operations/memory-space-migration.md).

## Deployment Options

### Docker Compose (3-Node Cluster)

Production-ready cluster deployment with monitoring:

```bash
cd deployments/docker/cluster

# Build and start 3-node cluster
docker compose build
docker compose up -d

# Start with monitoring (Prometheus + Grafana)
docker compose --profile monitoring up -d

# Verify cluster convergence
docker exec engram-node1 /usr/local/bin/engram status
# Expected: Shows cluster members alive: 2
```

See [Cluster Verification Cookbook](docs/operations/cluster_verification_cookbook.md) for comprehensive testing scenarios.

### Kubernetes (StatefulSet with DNS Discovery)

```bash
# Create Kind cluster for local testing
kind create cluster --config deployments/kubernetes/kind-cluster.yaml

# Build and load image
docker build -f deployments/docker/Dockerfile -t engram/engram:latest .
kind load docker-image engram/engram:latest

# Deploy 3-node cluster
kubectl apply -f deployments/kubernetes/engram-cluster.yaml

# Verify cluster formation
kubectl logs -n engram-cluster engram-0 | grep "cluster"
```

**Features**:
- DNS-based peer discovery via headless service
- StatefulSet ensures stable pod identity
- Pod anti-affinity distributes nodes across hosts
- Automatic rejoin after pod deletion

See [deployments/kubernetes/engram-cluster.yaml](deployments/kubernetes/engram-cluster.yaml) for complete manifest.

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

### Pattern Completion Example

Reconstruct missing memory details from partial information:

```bash
# Complete a partial memory
curl -X POST http://localhost:7432/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "partial": {
      "what": "Einstein published theory",
      "when": null,
      "where": "Physics lecture"
    },
    "params": {
      "ca3_sparsity": 0.05,
      "ca1_threshold": 0.7,
      "num_hypotheses": 3
    }
  }'
```

Returns:

```json
{
  "completed": {
    "what": "Einstein published theory of relativity in 1915",
    "when": "2024-01-05T10:00:00Z",
    "where": "Physics lecture",
    "confidence": 0.87
  },
  "source": "Reconstructed",
  "completion_confidence": 0.82,
  "alternatives": [
    {
      "what": "Einstein published theory of relativity",
      "confidence": 0.79
    }
  ]
}
```

**Key features:**

- `source: "Reconstructed"` - CA3 attractor dynamics filled in missing details
- `completion_confidence: 0.82` - Multi-factor confidence (convergence speed, energy reduction, field consensus)
- `alternatives` - System 2 reasoning provides alternative hypotheses for metacognitive checking

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

# Query Language (SQL-like syntax)
./target/debug/engram query "RECALL what='mitochondria' CONFIDENCE > 0.8"
./target/debug/engram query "SPREAD FROM what='CRISPR' HOPS 2"
./target/debug/engram query "CONSOLIDATE SPACE 'research' MIN_CLUSTER_SIZE 3"

# Configuration
./target/debug/engram config set feature_flags.spreading_api_beta true
./target/debug/engram config get feature_flags
```

### API Documentation

- Interactive docs: http://localhost:7432/docs
- OpenAPI spec: http://localhost:7432/api-docs/openapi.json
- Architecture docs: [vision.md](vision.md)
- Roadmap: [roadmap/](roadmap/)

#### Consolidated Beliefs API

Consolidation runs asynchronously after `remember` writes, transforming episodic memories into semantic beliefs with complete provenance trails. Query the new endpoints to inspect those beliefs:

- `GET /api/v1/consolidations` — returns the scheduler-backed snapshot of semantic beliefs with citation metadata, timestamps (`observed_at`, `stored_at`, `last_access`), and freshness metrics without running on-demand consolidation
- `GET /api/v1/consolidations/{pattern_id}` — drills into a specific belief with contributing episodes and decay signals
- `GET /api/v1/stream/consolidation` — SSE stream that emits belief updates whenever consolidation strengthens or forms new schemas

Write responses include `observed_at`, `stored_at`, and a `links.consolidation` pointer so clients can jump directly into belief inspection.

### Consolidation Timing

Memory consolidation is biologically-inspired and runs asynchronously with configurable behavior:

- **Episode Age Requirement**: Episodes must be at least 1 day old before consolidation (default). This mimics biological memory consolidation during sleep, not immediate encoding.
- **Consolidation Cadence**: Runs automatically every 60 seconds (configurable)
- **Why Age Threshold?**: Prevents premature consolidation of volatile recent memories, allowing time for interference patterns to stabilize

**Testing with fresh episodes**: Use backdated timestamps to test consolidation immediately:

```bash
curl -X POST http://localhost:7432/api/v1/episodes/remember \
  -H "Content-Type: application/json" \
  -d '{
    "what": "Test episode for immediate consolidation",
    "when": "2025-10-19T10:00:00Z",
    "confidence": 0.9
  }'
```

The `when` field is 2+ days in the past, making the episode immediately eligible for consolidation.

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
rustc --version  # Should be 1.82.0+
```

### Memory Spaces (Multi-Tenancy)

Common issues with multi-tenant deployments:

**Space Not Found**
```bash
# List all spaces
./target/debug/engram space list

# Create missing space explicitly
./target/debug/engram space create production
```

**Cross-Space Data Leakage**
```bash
# Verify space isolation
curl -H "X-Memory-Space: tenant-a" \
  http://localhost:7432/api/v1/memories/recall?query=test

# Should return only tenant-a memories (not tenant-b data)
```

**WAL Recovery Failures**
```bash
# Check recovery logs on startup
./target/debug/engram start 2>&1 | grep "Recovered"

# Expected output shows recovery per space:
# INFO Recovered 'default': 1200 entries, 0 corrupted, took 45ms
```

For comprehensive multi-tenant troubleshooting, see [docs/operations/memory-space-migration.md](docs/operations/memory-space-migration.md#troubleshooting).

### Performance Tuning

For optimal performance:

- Use `--release` builds in production
- Enable appropriate features for your use case
- Consider NUMA topology for large deployments
- Monitor with `curl http://localhost:7432/metrics`

**Zig Performance Kernels:**

- [Operations Guide](docs/operations/zig_performance_kernels.md) - Deployment, configuration, and troubleshooting
- [Rollback Procedures](docs/operations/zig_rollback_procedures.md) - Emergency and gradual rollback strategies
- [Architecture Documentation](docs/internal/zig_architecture.md) - Internal design for contributors

**Pattern Completion Tuning:**

- [Pattern Completion Parameter Tuning Guide](docs/tuning/completion_parameters.md) - Optimize CA3 sparsity, CA1 thresholds, and pattern weights
- [Pattern Completion Monitoring Operations](docs/operations/completion_monitoring.md) - Production monitoring, troubleshooting, and capacity planning

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
