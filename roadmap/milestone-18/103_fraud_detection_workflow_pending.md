# Task 003: Fraud Detection Pattern Matching Workflow

## Objective
Validate fraud detection workflow using graph traversal, pattern completion, and temporal analysis. Tests complex graph queries with pattern matching - critical for security and anomaly detection systems.

## Background
Fraud detection represents ~15% of production graph database workloads. Workflow pattern:
1. Transaction graph construction (account→transaction→merchant)
2. Pattern matching for known fraud signatures (rings, chains, mules)
3. Temporal analysis (velocity checks, unusual timing)
4. Pattern completion for partial fraud indicators

This workflow stresses:
- Complex graph traversal (multi-hop, typed edges)
- Pattern completion under uncertainty
- Temporal query patterns (time windows, velocity)
- Real-time query latency (<50ms for fraud scoring)

## Requirements

### Functional Requirements
1. Ingest 100,000 financial transactions forming account-merchant graph
2. Detect 50 known fraud patterns (rings, chains, velocity anomalies)
3. Pattern completion identifies 20 partial fraud indicators
4. Temporal velocity checks within 5-minute windows
5. Real-time fraud scoring <50ms P99 latency

### Non-Functional Requirements
1. Ingestion throughput >200 transactions/sec
2. Pattern matching P99 latency <50ms
3. Pattern completion P99 latency <100ms
4. Detection precision >90% (low false positives)
5. Detection recall >85% (catch most fraud)

## Technical Specification

### Test Data
Synthetic financial transaction graph:
- 100,000 transactions
- 10,000 accounts
- 5,000 merchants
- 50 injected fraud patterns (ground truth)
- 20 partial fraud indicators for pattern completion

### Fraud Patterns
1. **Transaction Rings**: Circular money flow (A→B→C→A)
2. **Mule Chains**: Sequential transfers through intermediaries
3. **Velocity Anomalies**: Unusual transaction frequency
4. **Location Jumps**: Impossible geographic transitions
5. **Amount Patterns**: Structured amounts (just under reporting limits)

### Files to Create

#### Test Implementation
**File**: `/Users/jordanwashburn/Workspace/orchard9/engram/tests/integration/fraud_detection.rs`

```rust
#[tokio::test]
#[ignore]
async fn test_fraud_detection_workflow() -> Result<()> {
    let server = start_test_server(FraudDetectionConfig {
        enable_pattern_completion: true,
        enable_temporal_queries: true,
        velocity_window: Duration::from_secs(300), // 5 minutes
        max_pattern_hops: 5,
        completion_confidence_threshold: 0.7,
        ..Default::default()
    }).await?;

    // Load synthetic fraud dataset
    let transactions = load_fraud_dataset("tests/fixtures/fraud_100k.jsonl")?;
    let ground_truth_fraud = load_ground_truth("tests/fixtures/fraud_ground_truth.json")?;

    server.create_memory_space("fraud_detection").await?;

    // PHASE 1: Ingest transaction graph
    let ingest_start = Instant::now();
    for txn in &transactions {
        server.store("fraud_detection", txn_to_episode(txn)?).await?;
    }
    let ingest_duration = ingest_start.elapsed();
    let throughput = transactions.len() as f64 / ingest_duration.as_secs_f64();
    assert!(throughput >= 200.0, "Ingestion throughput {} < 200/sec", throughput);

    // PHASE 2: Pattern matching for known fraud
    let mut detected_fraud = Vec::new();
    let mut pattern_latencies = Vec::new();

    for pattern in FRAUD_PATTERNS {
        let start = Instant::now();
        let matches = server.match_pattern("fraud_detection", pattern).await?;
        let latency = start.elapsed();

        detected_fraud.extend(matches);
        pattern_latencies.push(latency);
    }

    let p99_latency = percentile(&pattern_latencies, 0.99);
    assert!(
        p99_latency < Duration::from_millis(50),
        "Pattern matching P99 {} >= 50ms",
        p99_latency.as_millis()
    );

    // Validate detection quality
    let precision = calculate_precision(&detected_fraud, &ground_truth_fraud);
    let recall = calculate_recall(&detected_fraud, &ground_truth_fraud);

    assert!(precision >= 0.90, "Precision {} < 0.90", precision);
    assert!(recall >= 0.85, "Recall {} < 0.85", recall);

    // PHASE 3: Pattern completion for partial indicators
    let partial_patterns = load_partial_patterns("tests/fixtures/fraud_partial.json")?;
    let mut completion_latencies = Vec::new();
    let mut completed = 0;

    for partial in &partial_patterns {
        let start = Instant::now();
        let completion = server.complete_pattern("fraud_detection", partial).await?;
        let latency = start.elapsed();

        completion_latencies.push(latency);

        if completion.confidence >= 0.7 {
            completed += 1;
        }
    }

    let completion_p99 = percentile(&completion_latencies, 0.99);
    assert!(
        completion_p99 < Duration::from_millis(100),
        "Pattern completion P99 {} >= 100ms",
        completion_p99.as_millis()
    );

    assert!(
        completed >= 18, // 90% of 20 partial patterns
        "Pattern completion success {} < 18/20",
        completed
    );

    // PHASE 4: Temporal velocity checks
    let velocity_checks = generate_velocity_queries(1000);
    let mut velocity_latencies = Vec::new();

    for check in &velocity_checks {
        let start = Instant::now();
        let result = server.velocity_check(
            "fraud_detection",
            check.account_id,
            Duration::from_secs(300),
        ).await?;
        velocity_latencies.push(start.elapsed());
    }

    let velocity_p99 = percentile(&velocity_latencies, 0.99);
    assert!(
        velocity_p99 < Duration::from_millis(50),
        "Velocity check P99 {} >= 50ms",
        velocity_p99.as_millis()
    );

    Ok(())
}
```

#### Load Test Scenario
**File**: `/Users/jordanwashburn/Workspace/orchard9/engram/scenarios/fraud_detection.toml`

```toml
name = "Fraud Detection"
description = "Pattern matching and temporal analysis for fraud detection"

[duration]
total_seconds = 600  # 10 minutes

[operations]
store_weight = 0.3       # Transaction ingestion
recall_weight = 0.2      # Account history queries
pattern_match_weight = 0.3  # Fraud pattern detection
pattern_completion_weight = 0.2  # Partial pattern completion

[data]
source = "tests/fixtures/fraud_100k.jsonl"
num_transactions = 100000
num_accounts = 10000
num_merchants = 5000
fraud_patterns = 50

[temporal]
enabled = true
velocity_window_seconds = 300

[validation]
expected_throughput_ops_sec = 200.0
expected_pattern_match_p99_ms = 50.0
expected_completion_p99_ms = 100.0
min_precision = 0.90
min_recall = 0.85
```

## Acceptance Criteria

### Pass Criteria
1. Ingestion throughput ≥200 txn/sec
2. Pattern matching P99 <50ms
3. Pattern completion P99 <100ms
4. Detection precision ≥90%
5. Detection recall ≥85%
6. Velocity check P99 <50ms

### Fail Criteria
- Any performance target missed
- Precision <90% OR recall <85%
- Any errors during workflow

## Estimated Time
2 days:
- Day 1: Synthetic fraud data generation, pattern definitions
- Day 2: Pattern matching validation, completion testing

## Dependencies
- M8 (Pattern completion)
- M4 (Temporal dynamics)
- M3 (Spreading activation for graph traversal)

## References
- Pattern Completion: docs/explanation/pattern-completion.md
- Graph Queries: docs/howto/graph-queries.md
