# Task 002: Recommendation Engine End-to-End Workflow

## Objective
Validate recommendation engine workflow combining vector similarity search, graph traversal, and temporal decay. This tests high-read, moderate-write workloads with temporal bias - a critical production pattern for personalization systems.

## Background
Recommendation engines represent ~30% of production graph database workloads. Workflow pattern:
1. User interactions stored as episodes (views, purchases, ratings)
2. Vector similarity finds related items (content-based filtering)
3. Graph traversal discovers user-item networks (collaborative filtering)
4. Temporal decay prioritizes recent interactions
5. Spreading activation ranks recommendations by activation strength

This workflow stresses:
- Read-heavy workloads (95% reads, 5% writes)
- Temporal query patterns (recent memories stronger)
- HNSW index performance under concurrent queries
- Activation spreading for ranking

## Requirements

### Functional Requirements
1. Ingest 50,000 user-item interactions over 10 minutes (write phase)
2. Execute 100,000 recommendation queries over 20 minutes (read phase)
3. Recommendations blend vector similarity (content) + graph traversal (collaborative)
4. Temporal decay applied (recent interactions weighted 2x)
5. Validate spreading activation ranks results correctly

### Non-Functional Requirements
1. Write throughput >80 interactions/sec
2. Read throughput >80 queries/sec sustained
3. Recommendation P99 latency <100ms
4. HNSW search P99 latency <20ms
5. Memory overhead <15% vs episodic-only

## Technical Specification

### Test Scenario Architecture

**Test Data**: MovieLens 100K dataset (real user-item interactions)
- 50,000 ratings from 943 users on 1,682 movies
- Embeddings: Movie metadata → 384-dimensional vectors
- Temporal distribution: Realistic timestamp distribution (1997-1998)
- Ground truth: Known user preferences for validation

**Workload Pattern**:
- **Write Phase** (0-10 min): Ingest all 50K interactions (83/sec average)
- **Read Phase** (10-30 min): 100K recommendation queries (83/sec sustained)
- **Query Types**:
  - User-based: "What should user X watch next?"
  - Item-based: "Movies similar to Y"
  - Hybrid: "Movies for user X similar to Y"

**Temporal Bias**:
- Recent interactions (last 7 days): 2x weight
- Medium interactions (7-30 days): 1x weight
- Old interactions (>30 days): 0.5x weight

### Files to Create/Modify

#### Test Implementation
**File**: `/Users/jordanwashburn/Workspace/orchard9/engram/tests/integration/recommendation_engine.rs`

```rust
/// Recommendation engine workflow validation
///
/// Workflow:
/// 1. Ingest 50K user-item interactions (10 min write phase)
/// 2. Execute 100K recommendation queries (20 min read phase)
/// 3. Validate temporal decay applied correctly
/// 4. Verify spreading activation ranking quality
#[tokio::test]
#[ignore]
async fn test_recommendation_engine_workflow() -> Result<()> {
    let server = start_test_server(RecommendationConfig {
        enable_temporal_decay: true,
        decay_function: DecayFunction::TwoComponent {
            recent_half_life: Duration::from_days(7),
            remote_half_life: Duration::from_days(90),
        },
        enable_spreading_activation: true,
        max_spread_hops: 3,
        activation_threshold: 0.1,
        ..Default::default()
    }).await?;

    // Load MovieLens dataset
    let interactions = load_movielens_100k("tests/fixtures/ml-100k/u.data")?;
    let movie_embeddings = load_movie_embeddings("tests/fixtures/ml-100k/embeddings.jsonl")?;

    // Create memory space
    server.create_memory_space("movielens").await?;

    // PHASE 1: Write Phase (10 minutes)
    println!("=== Write Phase: Ingesting 50K interactions ===");
    let write_start = Instant::now();
    let mut write_metrics = WriteMetrics::default();

    for interaction in &interactions {
        // Store user-item interaction as episode
        let episode = create_interaction_episode(interaction, &movie_embeddings)?;
        let result = server.store("movielens", episode).await;

        write_metrics.record(result);

        // Realistic arrival rate: ~83/sec with variance
        let delay = exponential_delay(83.0);
        sleep(delay).await;
    }

    let write_duration = write_start.elapsed();
    let write_throughput = interactions.len() as f64 / write_duration.as_secs_f64();

    println!("Write phase complete: {} interactions/sec", write_throughput);
    assert!(write_throughput >= 80.0, "Write throughput too low");

    // PHASE 2: Read Phase (20 minutes)
    println!("=== Read Phase: Executing 100K recommendation queries ===");
    let read_start = Instant::now();
    let mut read_metrics = ReadMetrics::default();

    // Generate 100K recommendation queries
    let queries = generate_recommendation_queries(&interactions, 100_000);

    for query in &queries {
        let result = match query.query_type {
            QueryType::UserBased => {
                // User-based: Spread activation from user's history
                let user_history = get_user_interactions(&interactions, query.user_id);
                server.spread_activation("movielens", user_history, 3).await
            }
            QueryType::ItemBased => {
                // Item-based: Vector similarity search
                let item_embedding = movie_embeddings.get(&query.item_id)?;
                server.embedding_search("movielens", item_embedding, 10).await
            }
            QueryType::Hybrid => {
                // Hybrid: Combine spreading + similarity
                let user_history = get_user_interactions(&interactions, query.user_id);
                let item_embedding = movie_embeddings.get(&query.item_id)?;
                server.hybrid_recommend("movielens", user_history, item_embedding).await
            }
        };

        read_metrics.record(result);

        // High read rate
        let delay = exponential_delay(83.0);
        sleep(delay).await;
    }

    let read_duration = read_start.elapsed();
    let read_throughput = queries.len() as f64 / read_duration.as_secs_f64();

    println!("Read phase complete: {} queries/sec", read_throughput);
    assert!(read_throughput >= 80.0, "Read throughput too low");

    // PHASE 3: Temporal Decay Validation
    println!("=== Validating Temporal Decay ===");

    // Query user with interactions spanning time range
    let test_user_id = 1;
    let user_history = get_user_interactions(&interactions, test_user_id);

    // Get activation strengths for recent vs old interactions
    let activations = server.get_activation_strengths("movielens", &user_history).await?;

    let recent_avg = activations
        .iter()
        .filter(|a| a.age < Duration::from_days(7))
        .map(|a| a.strength)
        .sum::<f32>()
        / activations.iter().filter(|a| a.age < Duration::from_days(7)).count() as f32;

    let old_avg = activations
        .iter()
        .filter(|a| a.age > Duration::from_days(30))
        .map(|a| a.strength)
        .sum::<f32>()
        / activations.iter().filter(|a| a.age > Duration::from_days(30)).count() as f32;

    // Recent should be ~2x stronger than old
    let decay_ratio = recent_avg / old_avg;
    assert!(
        decay_ratio >= 1.8 && decay_ratio <= 2.2,
        "Temporal decay ratio {} not in expected range [1.8, 2.2]",
        decay_ratio
    );

    // PHASE 4: Spreading Activation Quality
    println!("=== Validating Spreading Activation Ranking ===");

    // For users with known preferences, validate recommendations
    let validation_users = vec![1, 42, 100]; // Users with diverse taste

    for user_id in validation_users {
        let recommendations = server
            .spread_activation("movielens", get_user_interactions(&interactions, user_id), 3)
            .await?;

        // Recommendations should be ranked by activation strength
        let mut prev_activation = f32::MAX;
        for rec in &recommendations {
            assert!(
                rec.activation <= prev_activation,
                "Recommendations not sorted by activation"
            );
            prev_activation = rec.activation;
        }

        // Top recommendation should have high activation (>0.5)
        assert!(
            recommendations[0].activation > 0.5,
            "Top recommendation has low activation: {}",
            recommendations[0].activation
        );
    }

    // PHASE 5: Performance Validation
    assert!(
        read_metrics.p99_latency < Duration::from_millis(100),
        "Read P99 {} exceeds 100ms",
        read_metrics.p99_latency.as_millis()
    );

    assert!(
        read_metrics.hnsw_p99_latency < Duration::from_millis(20),
        "HNSW P99 {} exceeds 20ms",
        read_metrics.hnsw_p99_latency.as_millis()
    );

    Ok(())
}
```

#### Load Test Scenario
**File**: `/Users/jordanwashburn/Workspace/orchard9/engram/scenarios/recommendation_engine.toml`

```toml
name = "Recommendation Engine"
description = "High-read workload with temporal decay (MovieLens 100K)"

[duration]
write_phase_seconds = 600   # 10 minutes
read_phase_seconds = 1200   # 20 minutes

[arrival]
pattern = "exponential"
rate = 83.0  # Both reads and writes

[operations]
# Write phase
store_weight = 1.0
recall_weight = 0.0

# Read phase (switched after write_phase_seconds)
# store_weight = 0.05
# recall_weight = 0.60
# embedding_search_weight = 0.35

[data]
source = "tests/fixtures/ml-100k/u.data"
embeddings = "tests/fixtures/ml-100k/embeddings.jsonl"
num_interactions = 50000
num_users = 943
num_items = 1682
embedding_dim = 384

[temporal]
enabled = true
decay_function = "two_component"
recent_half_life_days = 7
remote_half_life_days = 90

[spreading]
enabled = true
max_hops = 3
activation_threshold = 0.1

[validation]
expected_write_throughput = 80.0
expected_read_throughput = 80.0
expected_p99_latency_ms = 100.0
expected_hnsw_p99_ms = 20.0
```

## Acceptance Criteria

### Pass Criteria (ALL must be met)
1. **Write Performance**: 50K interactions ingested at ≥80/sec
   - Validation: `write_throughput >= 80.0`

2. **Read Performance**: 100K queries executed at ≥80/sec sustained
   - Validation: `read_throughput >= 80.0`

3. **Latency Bounds**:
   - Recommendation P99 <100ms
   - HNSW search P99 <20ms
   - Validation: Prometheus metrics + test assertions

4. **Temporal Decay Correctness**: Recent interactions 1.8-2.2x stronger
   - Validation: Statistical test on activation strengths

5. **Spreading Activation Ranking**: Results sorted by activation, top >0.5
   - Validation: Test assertions on recommendation ordering

6. **Zero Errors**: No errors in write or read phases
   - Validation: `error_count == 0`

### Fail Criteria
- Throughput below targets (write <80/sec OR read <80/sec)
- Latency exceeds bounds (P99 >100ms OR HNSW >20ms)
- Temporal decay ratio outside [1.8, 2.2]
- Recommendations unsorted or top activation <0.5
- Any errors during workflow

## Performance Budget

**Regression Threshold**: <5% from M17 baseline on core operations
- Store: P99 <0.53ms
- Recall: P99 <0.53ms
- Embedding search: P99 <20ms (within HNSW design target)

## Testing Approach

### Acceptance Script
**File**: `/Users/jordanwashburn/Workspace/orchard9/engram/scripts/acceptance/002_recommendation_engine.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Task 002: Recommendation Engine Acceptance Test ==="

# Download MovieLens 100K if missing
if [ ! -f "tests/fixtures/ml-100k/u.data" ]; then
    echo "Downloading MovieLens 100K dataset..."
    ./scripts/download_movielens_100k.sh
fi

# Generate movie embeddings if missing
if [ ! -f "tests/fixtures/ml-100k/embeddings.jsonl" ]; then
    echo "Generating movie embeddings..."
    python3 scripts/generate_movie_embeddings.py
fi

# Run test
cargo test --release --features dual_memory_types \
    test_recommendation_engine_workflow -- --ignored --nocapture

if [ $? -eq 0 ]; then
    echo "PASS: Recommendation engine workflow validated"
    exit 0
else
    echo "FAIL: Review test output"
    exit 1
fi
```

## Dependencies
- M17 Task 006 (Consolidation) - for temporal decay
- M11 (Streaming) - for high-throughput ingestion
- M3 (Spreading activation) - for collaborative filtering
- M2 (HNSW) - for content-based filtering

## Estimated Time
2 days:
- Day 1: MovieLens fixture processing, test implementation
- Day 2: Temporal decay validation, spreading quality checks

## Follow-Up Tasks
- If HNSW latency >20ms: Optimize index parameters (M18 Task 011)
- If temporal decay incorrect: Review decay function implementation
- If spreading quality low: Tune activation thresholds

## References
- MovieLens 100K: https://grouplens.org/datasets/movielens/100k/
- M3 Spreading Activation: Milestone 3 overview
- M2 HNSW Implementation: Milestone 2 completion summary
- Temporal Decay: docs/explanation/memory-dynamics.md
