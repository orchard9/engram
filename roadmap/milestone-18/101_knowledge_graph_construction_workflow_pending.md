# Task 001: Knowledge Graph Construction End-to-End Workflow

## Objective
Validate complete knowledge graph construction workflow from streaming ingestion through consolidation to semantic recall. This tests the most common production use case: building a knowledge base from document streams with automatic concept formation.

## Background
Knowledge graph construction represents ~40% of production graph database workloads (based on Neo4j/Stardog usage surveys). Workflow pattern:
1. Documents arrive via streaming API (bursty, variable size)
2. Episodes created from document embeddings (store operations)
3. Background consolidation forms concepts from related documents
4. Queries blend episodic (specific documents) and semantic (concepts) recall

This workflow stresses:
- Streaming backpressure handling
- Concurrent consolidation during writes
- Blended recall correctness
- Memory space isolation (multi-tenant)

## Requirements

### Functional Requirements
1. Ingest 10,000 documents via streaming API over 5 minutes (non-uniform arrival)
2. Automatic concept formation during ingestion (consolidation running concurrently)
3. Validate blended recall returns both specific documents and formed concepts
4. Confirm memory space isolation (3 parallel knowledge graphs)
5. Verify zero data loss across consolidation cycles

### Non-Functional Requirements
1. Streaming throughput >100 documents/sec sustained
2. Consolidation P99 latency <500ms (doesn't block ingestion)
3. Recall P99 latency <50ms with blended mode
4. Memory overhead <20% vs episodic-only mode
5. Zero errors throughout workflow

## Technical Specification

### Test Scenario Architecture

**Test Data**: Wikipedia article embeddings (real-world distribution)
- 10,000 articles from 3 domains (science, history, arts)
- Embeddings: 768-dimensional, pre-computed with sentence-transformers
- Arrival pattern: Poisson process (λ=50/sec) simulating real ingestion
- Ground truth: Known concept clusters for validation

**Memory Spaces**:
- Space 1: Science articles (3,500 documents)
- Space 2: History articles (3,500 documents)
- Space 3: Arts articles (3,000 documents)

**Workflow Phases**:
1. **Ingestion** (0-5 minutes): Stream documents via WebSocket
2. **Consolidation** (concurrent): Background concept formation every 30 seconds
3. **Validation** (5-10 minutes): Query each space, verify blended recall
4. **Integrity Check** (10-12 minutes): Validate data completeness and isolation

### Files to Create/Modify

#### Test Implementation
**File**: `/Users/jordanwashburn/Workspace/orchard9/engram/tests/integration/knowledge_graph_construction.rs`

```rust
/// End-to-end knowledge graph construction validation
///
/// Workflow:
/// 1. Stream 10K Wikipedia articles via WebSocket (5 min)
/// 2. Concurrent consolidation forms concepts (30s cadence)
/// 3. Blended recall returns documents + concepts
/// 4. Validate data integrity and isolation
#[tokio::test]
#[ignore] // Long-running test, manual execution
async fn test_knowledge_graph_construction_workflow() -> Result<()> {
    // Setup: Start server with dual memory enabled
    let server = start_test_server(DualMemoryConfig {
        enable_concept_formation: true,
        formation_interval: Duration::from_secs(30),
        enable_blended_recall: true,
        semantic_weight: 0.2,
        ..Default::default()
    }).await?;

    // Load test data
    let articles = load_wikipedia_embeddings("tests/fixtures/wikipedia_10k.jsonl")?;

    // Create 3 memory spaces
    let spaces = vec!["science", "history", "arts"];
    for space in &spaces {
        server.create_memory_space(space).await?;
    }

    // Phase 1: Streaming ingestion (5 minutes)
    let ingestion_start = Instant::now();
    let mut clients = Vec::new();

    for (space, docs) in partition_by_domain(&articles) {
        let client = WebSocketClient::connect(&server, space).await?;

        // Spawn ingestion task
        let handle = tokio::spawn(async move {
            let mut metrics = IngestionMetrics::default();

            for doc in docs {
                // Poisson arrival (λ=50/sec)
                let delay = poisson_delay(50.0);
                sleep(delay).await;

                let result = client.store_observation(doc).await;
                metrics.record(result);
            }

            metrics
        });

        clients.push(handle);
    }

    // Wait for all ingestion to complete
    let metrics: Vec<IngestionMetrics> = futures::future::join_all(clients)
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    let ingestion_duration = ingestion_start.elapsed();

    // Phase 2: Wait for final consolidation cycle
    sleep(Duration::from_secs(35)).await; // Allow last cycle to complete

    // Phase 3: Blended recall validation
    for space in &spaces {
        // Query 1: Specific document recall (should return episode)
        let specific_query = create_query_from_article(&articles[0]);
        let specific_results = server.recall(space, &specific_query).await?;

        assert!(specific_results.len() > 0, "Should recall specific document");
        assert!(
            specific_results.iter().any(|r| r.memory_type == MemoryType::Episode),
            "Should include episodic memory"
        );

        // Query 2: Abstract concept recall (should return concept)
        let concept_query = create_abstract_query(space);
        let concept_results = server.recall(space, &concept_query).await?;

        assert!(
            concept_results.iter().any(|r| r.memory_type == MemoryType::Concept),
            "Should include semantic concept"
        );

        // Query 3: Blended recall (should mix both)
        let blended_query = create_blended_query(space);
        let blended_results = server.recall(space, &blended_query).await?;

        let has_episodes = blended_results.iter().any(|r| r.memory_type == MemoryType::Episode);
        let has_concepts = blended_results.iter().any(|r| r.memory_type == MemoryType::Concept);

        assert!(has_episodes && has_concepts, "Blended recall should mix episode and concept");
    }

    // Phase 4: Data integrity and isolation
    let integrity = server.validate_integrity().await?;

    assert_eq!(integrity.total_episodes, 10_000, "All documents stored");
    assert_eq!(integrity.corrupted_count, 0, "Zero data corruption");
    assert_eq!(integrity.lost_count, 0, "Zero data loss");

    // Verify space isolation
    for space in &spaces {
        let space_stats = server.get_space_stats(space).await?;

        // Concepts should only reference episodes in same space
        for concept_id in &space_stats.concept_ids {
            let bindings = server.get_concept_bindings(concept_id).await?;

            for binding in bindings {
                let episode_space = server.get_episode_space(&binding.episode_id).await?;
                assert_eq!(
                    episode_space, space,
                    "Concept bindings must not cross space boundaries"
                );
            }
        }
    }

    // Performance validation
    let total_throughput = 10_000.0 / ingestion_duration.as_secs_f64();
    assert!(
        total_throughput >= 100.0,
        "Ingestion throughput {} < 100 docs/sec",
        total_throughput
    );

    // Collect metrics for reporting
    let consolidation_metrics = server.get_consolidation_metrics().await?;
    assert!(
        consolidation_metrics.p99_latency_ms < 500.0,
        "Consolidation P99 {} >= 500ms",
        consolidation_metrics.p99_latency_ms
    );

    Ok(())
}
```

#### Test Fixture Data
**File**: `/Users/jordanwashburn/Workspace/orchard9/engram/tests/fixtures/wikipedia_10k.jsonl.gz`
- 10,000 Wikipedia article embeddings
- Format: `{"title": str, "embedding": [f32; 768], "domain": str}`
- Compression: gzip for repository storage
- Source: sentence-transformers/all-MiniLM-L6-v2 model

#### Load Test Scenario
**File**: `/Users/jordanwashburn/Workspace/orchard9/engram/scenarios/knowledge_graph_construction.toml`

```toml
name = "Knowledge Graph Construction"
description = "Streaming ingestion with concurrent consolidation (Wikipedia articles)"

[duration]
total_seconds = 300  # 5 minutes

[arrival]
pattern = "poisson"
rate = 50.0  # λ=50 docs/sec

[operations]
store_weight = 1.0   # Pure ingestion
recall_weight = 0.0
embedding_search_weight = 0.0
pattern_completion_weight = 0.0

[data]
source = "tests/fixtures/wikipedia_10k.jsonl.gz"
num_nodes = 10000
embedding_dim = 768
memory_spaces = 3

[consolidation]
enabled = true
interval_seconds = 30
coherence_threshold = 0.85
min_cluster_size = 10

[validation]
expected_throughput_ops_sec = 100.0
expected_consolidation_p99_ms = 500.0
max_error_rate = 0.0
zero_data_loss = true
```

#### Acceptance Test Script
**File**: `/Users/jordanwashburn/Workspace/orchard9/engram/scripts/acceptance/001_knowledge_graph_construction.sh`

```bash
#!/usr/bin/env bash
# Acceptance test for Task 001: Knowledge Graph Construction Workflow
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Task 001: Knowledge Graph Construction Acceptance Test ==="

# Pre-flight checks
echo "1. Checking prerequisites..."
if ! command -v cargo &> /dev/null; then
    echo "FAIL: cargo not found"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/tests/fixtures/wikipedia_10k.jsonl.gz" ]; then
    echo "FAIL: Test fixture missing (wikipedia_10k.jsonl.gz)"
    echo "Run: ./scripts/download_test_fixtures.sh"
    exit 1
fi

# Build test binary
echo "2. Building integration tests..."
cd "$PROJECT_ROOT"
cargo build --tests --release --features dual_memory_types

# Start server
echo "3. Starting Engram server..."
./target/release/engram start --config configs/test_dual_memory.toml &
SERVER_PID=$!
sleep 5

# Verify server health
if ! curl -s http://localhost:7432/health | grep -q "healthy"; then
    echo "FAIL: Server not healthy"
    kill $SERVER_PID
    exit 1
fi

# Run integration test
echo "4. Executing knowledge graph construction test..."
START_TIME=$(date +%s)
cargo test --release --features dual_memory_types \
    test_knowledge_graph_construction_workflow -- --ignored --nocapture

TEST_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Stop server
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

# Evaluate results
echo ""
echo "=== Test Results ==="
echo "Duration: ${DURATION}s"
echo "Exit Code: $TEST_EXIT_CODE"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "Status: PASS"
    echo ""
    echo "Acceptance Criteria Met:"
    echo "✓ 10,000 documents ingested via streaming"
    echo "✓ Concurrent consolidation formed concepts"
    echo "✓ Blended recall returned episodes + concepts"
    echo "✓ Memory space isolation validated"
    echo "✓ Zero data loss confirmed"
    echo "✓ Performance targets met (>100 docs/sec, <500ms consolidation)"
    exit 0
else
    echo "Status: FAIL"
    echo ""
    echo "Review test output above for failure details"
    exit 1
fi
```

## Acceptance Criteria

### Pass Criteria (ALL must be met)
1. **Ingestion Success**: 10,000 documents stored across 3 memory spaces
   - Validation: `total_episodes == 10000` in integrity check

2. **Concurrent Consolidation**: Concepts formed during ingestion without blocking
   - Validation: >0 concepts formed, consolidation P99 <500ms

3. **Blended Recall Correctness**: Queries return both episodes and concepts
   - Validation: Test queries find both MemoryType::Episode and MemoryType::Concept

4. **Space Isolation**: Concept bindings never cross memory space boundaries
   - Validation: All bindings reference episodes in same space

5. **Zero Data Loss**: All ingested documents retrievable
   - Validation: `lost_count == 0` in integrity check

6. **Performance Targets**:
   - Ingestion throughput ≥100 docs/sec sustained
   - Consolidation P99 latency <500ms
   - Recall P99 latency <50ms

7. **Zero Errors**: No errors during any workflow phase
   - Validation: Error metrics remain at 0 throughout test

### Fail Criteria (ANY triggers failure)
- Data loss detected (even single document)
- Space isolation violated (cross-space bindings)
- Performance below targets (throughput <100/sec OR consolidation >500ms)
- Any errors in ingestion, consolidation, or recall paths
- Blended recall missing either episode or concept results

## Performance Budget

**Baseline**: M17 baseline (0.501ms P99, 999.9 ops/sec)
**Regression Threshold**: <5% on core operations

Specific targets:
- Store operations: P99 <0.53ms (within 5% of baseline)
- Recall operations: P99 <50ms (blended recall overhead acceptable)
- Consolidation: P99 <500ms (background operation, not critical path)
- Memory overhead: <20% vs episodic-only (dual memory storage cost)

## Testing Approach

### Local Development Testing
```bash
# Quick validation (subset)
cargo test knowledge_graph_construction --features dual_memory_types

# Full acceptance test
./scripts/acceptance/001_knowledge_graph_construction.sh
```

### CI Integration
```bash
# Run in CI with timeout
timeout 15m ./scripts/acceptance/001_knowledge_graph_construction.sh
```

### Observability Validation
During test execution, verify Grafana dashboards capture:
- Streaming ingestion rate (docs/sec)
- Consolidation cycle timing and concept formation
- Blended recall latency distribution
- Memory space isolation (no cross-space queries)

Check Prometheus metrics:
```bash
# During test
curl http://localhost:9090/api/v1/query?query=rate(observations_stored_total[1m])
curl http://localhost:9090/api/v1/query?query=consolidation_duration_seconds
curl http://localhost:9090/api/v1/query?query=blended_recall_latency_seconds
```

## Dependencies

**Code Dependencies**:
- M17 Task 001 (Dual memory types) - COMPLETE
- M17 Task 004 (Concept formation) - COMPLETE
- M17 Task 006 (Consolidation integration) - COMPLETE
- M11 (Streaming interface) - COMPLETE

**Infrastructure Dependencies**:
- WebSocket streaming support
- Background consolidation scheduler
- Blended recall implementation (M17 Task 009)
- Memory space multi-tenancy (M7)

**Test Data Dependencies**:
- Wikipedia embeddings fixture (10K documents)
- Sentence-transformers model for query generation

## Estimated Time
2 days:
- Day 1: Implement integration test, load scenario, fixtures
- Day 2: Acceptance script, observability validation, documentation

## Follow-Up Tasks
- If blended recall not yet complete: Task depends on M17 Task 009
- If performance fails: Create optimization task in M18 Phase 4
- If isolation fails: Critical bug, must fix before proceeding

## References
- M17 Dual Memory Overview: roadmap/milestone-17/000_milestone_overview_dual_memory.md
- M11 Streaming Interface: Milestone 11 completion summary
- M7 Memory Spaces: Milestone 7 completion summary
- Load Test Documentation: docs/operations/load-testing.md
- Wikipedia Embeddings: tests/fixtures/README.md (to be created)
