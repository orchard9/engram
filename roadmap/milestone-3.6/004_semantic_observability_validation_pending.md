# Task 004: Semantic Recall Observability & Validation Suite

## Objective
Establish rigorous evaluation, monitoring, and regression infrastructure covering multilingual recall, synonym expansion, and figurative interpretation so we can ship with measurable guarantees and catch drift early.

## Priority
P0 (Critical Path)

## Effort Estimate
1.5 days

## Dependencies
- Task 001: Multilingual Embedding Pipeline
- Task 002: Semantic Query Expansion & Recall Routing
- Task 003: Figurative Language Interpretation Engine

## Current Baseline Analysis

### Existing Infrastructure
- **Benchmark Framework**: `engram-core/benches/support/` provides reusable ANN testing infrastructure
  - `datasets.rs`: Dataset loading with synthetic generation (`generate_synthetic(10_000, 100)`)
  - `ann_common.rs`: `BenchmarkFramework` for comparing implementations
  - `engram_ann.rs`, `faiss_ann.rs`, `annoy_ann.rs`: Adapter patterns for different backends

- **Metrics System**: `engram-core/src/metrics/` provides lock-free streaming metrics
  - `mod.rs:159-166`: `MetricsRegistry::streaming_snapshot()` aggregates windows with spreading summary
  - `streaming.rs:12-98`: `StreamingAggregator` with lock-free queues (ArrayQueue, 64K buffer)
  - `streaming.rs:140-357`: `AggregatedMetrics` with 1s/10s/1m/5m windows, schema versioning (1.0.0)
  - `streaming.rs:393-428`: `SpreadingSummary` with per-tier latency, circuit breaker state, pool metrics

- **SSE Streaming**: `engram-cli/src/api.rs` implements Server-Sent Events
  - Lines 1310-1361: `stream_activities()` with event type filtering, importance thresholds
  - Lines 1466-1618: `create_activity_stream()` maps `MemoryEvent` to SSE with session tracking
  - Lines 1621-1743: `create_memory_stream()` filters formation/retrieval by confidence

- **Integration Tests**: `engram-cli/tests/streaming_tests.rs` validates SSE delivery
  - Lines 638-704: End-to-end test verifies remember→SSE delivery (StoreResult.streaming_delivered)
  - Lines 712-813: Real event consumption test subscribes, performs recall, asserts MemoryEvent::Recalled

### Gaps for Semantic Observability
1. No semantic-specific metrics (query expansion, multilingual, figurative interpretation)
2. No nDCG/recall@k evaluation harness
3. No benchmark corpus for multilingual/figurative queries
4. No CI integration for regression detection
5. SSE payloads lack semantic diagnostics (expansion terms, language detection, figurative flags)

## Implementation Plan

### 1. Benchmark Corpus (`datasets/semantic_eval/`)

**Files to Create**:
- `datasets/semantic_eval/multilingual_queries.jsonl`: Query-document pairs across languages
  ```json
  {"query_lang": "es", "query": "depurar codigo", "doc_id": "debugging_001", "doc_lang": "en", "expected_nDCG": 0.86, "category": "cross_lingual"}
  {"query_lang": "en", "query": "debugging code", "doc_id": "debugging_001", "doc_lang": "en", "expected_nDCG": 0.95, "category": "literal"}
  ```

- `datasets/semantic_eval/synonym_expansion.jsonl`: Synonym-based recall tests
  ```json
  {"query": "debugging concurrent code", "expansion_terms": ["race condition", "thread safety", "mutex"], "expected_delta_nDCG": 0.18}
  ```

- `datasets/semantic_eval/figurative_interpretation.jsonl`: Metaphor/idiom tests
  ```json
  {"query": "debugging is detective work", "figurative_type": "metaphor", "expected_docs": ["troubleshooting_process", "diagnostic_methods"], "min_confidence": 0.7}
  ```

**Generation Script**: `scripts/generate_semantic_eval.rs`
- Extend `engram-core/benches/support/datasets.rs::DatasetLoader` pattern
- Add `generate_semantic_eval()` function with language distribution (40% English, 20% Spanish, 15% French, 10% German, 5% Chinese, 10% other)
- Include Pareto distribution for query frequency (80% common, 20% edge cases)

### 2. Evaluation Harness (`engram-core/benches/semantic_recall.rs`)

**Structure** (follow existing `ann_comparison.rs` pattern):
```rust
use criterion::{Criterion, black_box};
use support::datasets::SemanticEvalDataset; // New module
use engram_core::{MemoryStore, Cue, Confidence};

pub fn semantic_recall_benchmark(c: &mut Criterion) {
    c.bench_function("semantic_recall_multilingual", |b| {
        let dataset = SemanticEvalDataset::load("datasets/semantic_eval/multilingual_queries.jsonl");
        let mut store = MemoryStore::new(10_000);
        // Populate store with docs

        b.iter(|| {
            for query in &dataset.queries {
                let cue = Cue::semantic(query.id.clone(), query.text.clone(), Confidence::HIGH);
                let result = store.recall(&cue);
                black_box(compute_ndcg(&result, &query.expected_docs, 10));
            }
        });
    });
}
```

**Metrics to Compute**:
- **nDCG@10**: Using formula `DCG@k / IDCG@k` where `DCG = Σ(relevance_i / log2(i+1))`
- **Recall@k**: Fraction of relevant docs in top k
- **Cross-lingual consistency**: Spearman rank correlation between query variants (target >0.8)
- **Expansion effectiveness**: Delta nDCG from expansion vs literal-only
- **Confidence calibration**: Expected Calibration Error (ECE) = avg(|predicted_conf - actual_correctness|)

**New Support Module**: `engram-core/benches/support/semantic_datasets.rs`
```rust
pub struct SemanticEvalDataset {
    pub queries: Vec<SemanticQuery>,
}

pub struct SemanticQuery {
    pub id: String,
    pub text: String,
    pub lang: String,
    pub expected_docs: Vec<String>,
    pub expected_nDCG: f64,
    pub category: QueryCategory, // Literal, CrossLingual, Synonym, Figurative
}

pub enum QueryCategory {
    Literal,
    CrossLingual { source_lang: String, target_lang: String },
    SynonymExpansion { original_terms: Vec<String>, expanded_terms: Vec<String> },
    Figurative { figurative_type: String }, // metaphor, idiom, sarcasm
}
```

**CLI Tool**: `engram-cli/src/bin/semantic-eval.rs`
```rust
// Standalone binary for offline semantic evaluation
fn main() {
    let args = SemanticEvalArgs::parse();
    let dataset = SemanticEvalDataset::load(&args.dataset_path);
    let store = load_memory_store(&args.memory_db);

    let results = evaluate_semantic_recall(&store, &dataset);
    write_results(&results, &args.output_path);

    if results.ndcg_at_10 < args.threshold {
        eprintln!("FAILED: nDCG@10 {:.3} below threshold {:.3}", results.ndcg_at_10, args.threshold);
        std::process::exit(1);
    }
}
```

### 3. Continuous Regression Tests (CI Integration)

**Extend Makefile**:
```makefile
.PHONY: semantic-eval semantic-eval-fast

semantic-eval:
	cargo bench --bench semantic_recall -- --save-baseline semantic_baseline
	cargo run -p engram-cli --bin semantic-eval -- \
		--dataset datasets/semantic_eval/ \
		--threshold 0.75 \
		--output target/semantic_eval/$(shell date +%Y%m%d_%H%M%S).json

semantic-eval-fast:
	# Subset for PR checks (5 min max)
	cargo run -p engram-cli --bin semantic-eval -- \
		--dataset datasets/semantic_eval/subset.jsonl \
		--threshold 0.70

quality: fmt lint test semantic-eval-fast
```

**GitHub Actions** (if `.github/workflows/` exists, add nightly job):
```yaml
name: Nightly Semantic Regression
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
  workflow_dispatch:

jobs:
  semantic-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      # Pin CPU frequency to reduce variance
      - name: Set CPU governor
        run: |
          echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

      - name: Warm caches
        run: cargo build --release --benches

      - name: Run semantic evaluation
        run: make semantic-eval

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: semantic-eval-results
          path: target/semantic_eval/*.json

      - name: Check regression threshold
        run: |
          # Parse JSON and fail if nDCG@10 drops >5% from baseline
          python scripts/check_regression.py \
            --current target/semantic_eval/latest.json \
            --baseline target/semantic_eval/baseline.json \
            --threshold 0.05
```

### 4. Observability Hooks (Metrics + SSE Enhancements)

**Extend `engram-core/src/metrics/mod.rs`**:
```rust
// Add to existing constant declarations (after line 43)
const SEMANTIC_RECALL_CONFIDENCE: &str = "engram_semantic_recall_confidence";
const QUERY_EXPANSION_LATENCY_SECONDS: &str = "engram_query_expansion_latency_seconds";
const FIGURATIVE_SUCCESS_TOTAL: &str = "engram_figurative_success_total";
const MULTILINGUAL_QUERY_TOTAL: &str = "engram_multilingual_query_total";
const EXPANSION_TERMS_COUNT: &str = "engram_expansion_terms_count";
const LANGUAGE_DETECTION_LATENCY_MS: &str = "engram_language_detection_latency_ms";

// Add to MetricsRegistry impl
pub fn record_semantic_recall(&self, confidence: f64, expansion_applied: bool, figurative_detected: bool) {
    self.observe_histogram(SEMANTIC_RECALL_CONFIDENCE, confidence);
    if expansion_applied {
        self.increment_counter(QUERY_EXPANSION_LATENCY_SECONDS, 1);
    }
    if figurative_detected {
        self.increment_counter(FIGURATIVE_SUCCESS_TOTAL, 1);
    }
}
```

**Extend `SemanticSummary` in `engram-core/src/metrics/streaming.rs`**:
```rust
// Add after SpreadingSummary (line 428)
#[derive(Debug, Clone, Serialize, Default)]
pub struct SemanticSummary {
    /// Average recall confidence across queries (0.0-1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_recall_confidence: Option<f64>,

    /// Query expansion latency summary
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansion_latency_ms: Option<TierLatencySummary>,

    /// Total queries with expansion applied
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansion_applied_total: Option<u64>,

    /// Total figurative interpretations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub figurative_success_total: Option<u64>,

    /// Per-language query distribution
    pub per_language: BTreeMap<String, LanguageMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LanguageMetrics {
    pub queries_total: u64,
    pub avg_confidence: f64,
    pub cross_lingual_ratio: f64, // Fraction querying in different language than result
}

// Add to AggregatedMetrics (line 357)
#[serde(skip_serializing_if = "Option::is_none")]
pub semantic: Option<SemanticSummary>,
```

**Augment SSE Payloads** (`engram-cli/src/api.rs`):

Modify `create_activity_stream()` (line 1495-1580) to add semantic diagnostics:
```rust
MemoryEvent::Recalled { id, activation, confidence } => {
    // Existing code...

    // NEW: Add semantic diagnostics
    let mut semantic_diag = json!({});
    if let Some(expansion_info) = get_expansion_info(&id) {
        semantic_diag["expansion_applied"] = json!(true);
        semantic_diag["expansion_terms"] = json!(expansion_info.terms);
        semantic_diag["original_query"] = json!(expansion_info.original);
    }
    if let Some(lang_info) = get_language_info(&id) {
        semantic_diag["query_lang"] = json!(lang_info.query_lang);
        semantic_diag["result_lang"] = json!(lang_info.result_lang);
        semantic_diag["cross_lingual"] = json!(lang_info.query_lang != lang_info.result_lang);
    }
    if let Some(fig_info) = get_figurative_info(&id) {
        semantic_diag["figurative_detected"] = json!(true);
        semantic_diag["figurative_type"] = json!(fig_info.fig_type);
    }

    event_data["semantic_diagnostics"] = semantic_diag;
}
```

**Structured Logging** (`semantic.log`):
```rust
// Add to engram-cli/src/main.rs logging setup
use tracing_appender::rolling::{RollingFileAppender, Rotation};

let semantic_appender = RollingFileAppender::new(
    Rotation::DAILY,
    "logs",
    "semantic.log"
);

tracing_subscriber::fmt()
    .with_writer(semantic_appender)
    .with_target(true)
    .json()
    .init();

// Usage in query handling code
tracing::info!(
    target: "engram::semantic",
    query_id = %query_id,
    original_text = %query.text,
    detected_language = %lang,
    expansion_applied = expansion.is_some(),
    expansion_terms = ?expansion.as_ref().map(|e| &e.terms),
    figurative_detected = figurative.is_some(),
    top_results = ?results.iter().take(10).map(|r| &r.id).collect::<Vec<_>>(),
    ndcg_at_10 = metrics.ndcg,
    "semantic query processed"
);
```

### 5. Operator Dashboards

**Grafana Dashboard JSON** (`docs/operations/semantic_dashboard.json`):
```json
{
  "dashboard": {
    "title": "Engram Semantic Recall Quality",
    "panels": [
      {
        "id": 1,
        "title": "nDCG@10 Trend",
        "targets": [{
          "expr": "engram_semantic_recall_confidence{quantile=\"0.9\"}",
          "legend": "p90 Recall Confidence"
        }],
        "alert": {
          "name": "nDCG Regression",
          "conditions": [{
            "evaluator": {"params": [0.75], "type": "lt"},
            "query": {"model": "A"},
            "reducer": {"type": "avg"}
          }],
          "message": "nDCG@10 dropped below 0.75 threshold"
        }
      },
      {
        "id": 2,
        "title": "Query Expansion Effectiveness",
        "targets": [{
          "expr": "rate(engram_query_expansion_latency_seconds_sum[5m]) / rate(engram_query_expansion_latency_seconds_count[5m])",
          "legend": "Avg Expansion Latency"
        }]
      },
      {
        "id": 3,
        "title": "Cross-Lingual Query Heatmap",
        "type": "heatmap",
        "targets": [{
          "expr": "engram_multilingual_query_total",
          "format": "heatmap"
        }]
      },
      {
        "id": 4,
        "title": "Figurative Success Rate",
        "targets": [{
          "expr": "rate(engram_figurative_success_total[5m])",
          "legend": "Figurative Interpretations/sec"
        }]
      }
    ]
  }
}
```

**Alert Runbooks** (`docs/operations/semantic_alerts.md`):
```markdown
# Semantic Recall Alert Runbook

## Alert: nDCG Regression (P1)
**Condition**: nDCG@10 < 0.75 for >3 consecutive intervals
**Impact**: User-facing recall quality degradation

### Diagnosis
1. Check recent deployments: `git log --since="1 day ago" --oneline`
2. Inspect semantic.log for error patterns: `grep ERROR logs/semantic.log | tail -50`
3. Compare language distribution: `jq '.per_language' target/semantic_eval/latest.json`

### Mitigation
1. Rollback to previous version if deployment within 24h
2. Increase recall threshold temporarily: `curl -X POST /api/v1/config/recall_threshold -d '{"value": 0.6}'`
3. Disable figurative interpretation if causing failures: `DISABLE_FIGURATIVE=1`

### Root Cause Analysis
- Check embedding model drift: compare embedding distributions
- Validate synonym expansion dictionary freshness
- Review query log for new language patterns not in training

## Alert: Query Expansion Latency Spike (P2)
**Condition**: p95 expansion latency > 100ms
**Impact**: Slower overall query response time

### Diagnosis
1. Check GPU utilization: `nvidia-smi`
2. Review expansion cache hit rate: `curl /metrics | grep expansion_cache_hit`
3. Inspect query complexity distribution

### Mitigation
1. Increase expansion cache size
2. Reduce max expansion terms from 5 to 3
3. Route expensive queries to batch queue

## Alert: Cross-Lingual Performance Gap (P2)
**Condition**: Low-resource language nDCG <0.6 while English >0.85
**Impact**: Unfair experience for non-English users

### Diagnosis
1. Identify underperforming languages: `jq '.per_language | to_entries | sort_by(.value.avg_confidence)' latest.json`
2. Check embedding alignment quality for language pair
3. Review training data coverage for low-resource languages

### Mitigation
1. Increase cross-lingual training data for affected languages
2. Adjust confidence thresholds per language
3. Document known limitations in user-facing docs
```

## Acceptance Criteria

### Must Have
- [x] Benchmark corpus created with 1000+ multilingual queries covering 5+ languages
- [x] `cargo bench --bench semantic_recall` runs locally with nDCG/recall@k metrics
- [x] CLI tool `semantic-eval` runs offline evaluation with JSON output
- [x] Metrics system extended with semantic-specific counters/histograms
- [x] SSE payloads include `semantic_diagnostics` field (backward compatible)
- [x] Structured logging to `semantic.log` with rotation
- [x] Grafana dashboard JSON committed with 4+ panels
- [x] Alert runbooks documented with P1/P2 severity tiers
- [x] `make quality` includes `semantic-eval-fast` (subset evaluation)
- [x] Regression boundary: >5% nDCG@10 drop triggers CI failure

### Should Have
- [ ] Nightly CI job running full evaluation suite (if CI infrastructure available)
- [ ] Historical results stored for trend analysis (30 days retention)
- [ ] Confidence calibration plots (reliability diagrams)
- [ ] A/B testing framework for semantic feature flags

### Could Have
- [ ] Real-time anomaly detection on semantic metrics
- [ ] Automated retraining triggers based on quality degradation
- [ ] User feedback loop integration for ground truth labeling

## Testing Approach

### Deterministic Validation
1. **Seeded Datasets**: Generate synthetic queries with fixed random seed (`rand::seed(42)`)
2. **Deterministic Mode**: Disable non-deterministic features (HNSW randomization, GPU if non-deterministic)
3. **Snapshot Testing**: Compare full evaluation output JSON against committed golden file

### Unit Tests (` engram-core/tests/semantic_metrics_test.rs`)
```rust
#[test]
fn test_semantic_summary_serialization() {
    let summary = SemanticSummary {
        avg_recall_confidence: Some(0.82),
        expansion_applied_total: Some(1500),
        // ...
    };

    let json = serde_json::to_string(&summary).unwrap();
    let deserialized: SemanticSummary = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.avg_recall_confidence, Some(0.82));
}

#[test]
fn test_sse_backward_compatibility() {
    // Old clients should ignore new semantic_diagnostics field
    let event_with_diag = create_recall_event_with_semantics();
    let event_without_diag = create_recall_event_legacy();

    // Both should deserialize to same MemoryEvent type
    assert!(matches!(event_with_diag, MemoryEvent::Recalled { .. }));
    assert!(matches!(event_without_diag, MemoryEvent::Recalled { .. }));
}
```

### Integration Tests (`engram-cli/tests/semantic_streaming_test.rs`)
```rust
#[tokio::test]
async fn test_semantic_diagnostics_in_sse() {
    let store = setup_store_with_multilingual_docs();
    let mut event_rx = store.subscribe_to_events().unwrap();

    // Perform cross-lingual query (Spanish -> English result)
    let cue = Cue::semantic("test".into(), "depurar codigo".into(), Confidence::HIGH);
    let result = store.recall(&cue);

    // Receive SSE event
    let event = timeout(Duration::from_secs(2), event_rx.recv()).await.unwrap().unwrap();

    match event {
        MemoryEvent::Recalled { id, .. } => {
            // Parse semantic diagnostics from structured log or SSE payload
            let diag = get_semantic_diagnostics(&id);
            assert_eq!(diag.query_lang, "es");
            assert_eq!(diag.result_lang, "en");
            assert!(diag.cross_lingual);
        }
        _ => panic!("Expected Recalled event"),
    }
}
```

### Manual Verification (Staging)
1. Deploy to staging environment with Grafana instance
2. Import `semantic_dashboard.json` and verify all panels render
3. Trigger test queries covering all categories (literal, cross-lingual, synonym, figurative)
4. Verify alerts fire correctly for simulated regressions

## Risk Mitigation

### Benchmark Dataset Licensing
- **Action**: Use only CC-BY or CC0 data sources (e.g., Wikipedia, MIRACL dataset)
- **Validation**: Add `datasets/semantic_eval/LICENSE` documenting all sources
- **Anonymization**: Strip any PII from query examples before commit

### CI Flakiness Prevention
- **CPU Pinning**: Use `numactl --cpunodebind=0 --membind=0` in CI
- **Cache Warming**: Run `cargo build --release` before benchmarks to stabilize filesystem cache
- **Variance Collection**: Run each benchmark 10x, report mean ± stddev, fail only if >3 sigma from baseline
- **Timeout Guards**: Set `timeout: 600000` (10 min) for benchmarks to catch infinite loops

### Backward Compatibility
- **Schema Versioning**: `AggregatedMetrics::SCHEMA_VERSION = "1.1.0"` (bump minor for additive changes)
- **Optional Fields**: All new SSE fields use `#[serde(skip_serializing_if = "Option::is_none")]`
- **Deprecation Policy**: Keep old field names for 2 releases before removal

## Implementation Order

1. **Day 1 Morning**: Create benchmark corpus + datasets module (4h)
2. **Day 1 Afternoon**: Implement evaluation harness + CLI tool (4h)
3. **Day 2 Morning**: Extend metrics system + SSE diagnostics (3h)
4. **Day 2 Midday**: Write integration tests + CI configuration (3h)
5. **Day 2 Afternoon**: Create Grafana dashboard + runbooks (2h)

## Files to Create/Modify

### New Files (15)
- `datasets/semantic_eval/multilingual_queries.jsonl`
- `datasets/semantic_eval/synonym_expansion.jsonl`
- `datasets/semantic_eval/figurative_interpretation.jsonl`
- `datasets/semantic_eval/LICENSE`
- `datasets/semantic_eval/README.md`
- `scripts/generate_semantic_eval.rs`
- `engram-core/benches/semantic_recall.rs`
- `engram-core/benches/support/semantic_datasets.rs`
- `engram-cli/src/bin/semantic-eval.rs`
- `engram-core/tests/semantic_metrics_test.rs`
- `engram-cli/tests/semantic_streaming_test.rs`
- `docs/operations/semantic_dashboard.json`
- `docs/operations/semantic_alerts.md`
- `.github/workflows/semantic-regression.yml` (if applicable)
- `scripts/check_regression.py`

### Modified Files (5)
- `Makefile`: Add `semantic-eval` and `semantic-eval-fast` targets
- `engram-core/src/metrics/mod.rs`: Add semantic metric constants and recording methods
- `engram-core/src/metrics/streaming.rs`: Add `SemanticSummary` struct
- `engram-cli/src/api.rs`: Augment SSE events with `semantic_diagnostics`
- `engram-cli/src/main.rs`: Add semantic logging configuration

## Notes

- **Coordination**: Share `SemanticEvalDataset` format with verification-testing-lead agent for differential testing
- **Future Integration**: Design `SemanticSummary` to be extensible for probabilistic query milestone (Task 002's routing decisions)
- **Performance**: Benchmark overhead <1% for metric collection (validated via `benches/deterministic_overhead.rs` pattern)
- **Documentation**: Add semantic evaluation guide to `docs/howto/semantic_evaluation.md` referencing this implementation
