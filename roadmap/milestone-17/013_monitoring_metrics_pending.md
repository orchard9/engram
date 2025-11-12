# Task 013: Production-Grade Monitoring and Metrics for Dual Memory Operations

## Objective
Design and implement comprehensive production monitoring infrastructure for dual memory operations with cognitively-grounded metrics, provably correct alerting thresholds, and zero-cardinality-explosion architecture.

## Background

Production memory systems fail silently. Semantic drift, binding decay, and fan-out pathologies manifest as gradual degradation rather than hard failures. This demands metrics that capture not just performance (latency, throughput) but cognitive correctness (coherence, confidence calibration, pathway convergence).

The existing metrics infrastructure (engram-core/src/metrics/) provides:
- Lock-free counters, histograms, gauges (<1% overhead)
- Prometheus text format export
- Streaming aggregation with multiple time windows
- Multi-tenant label support with cardinality protection

This task extends that foundation with dual-memory-specific instrumentation.

## Requirements

### Cognitive Metric Categories
1. **Concept Lifecycle**: Formation rate, quality distribution, decay dynamics
2. **Binding Dynamics**: Creation, strengthening, weakening, garbage collection efficiency
3. **Fan Effect Impact**: Distribution, penalty magnitude, high-fan node detection
4. **Recall Performance**: Episodic vs semantic vs blended latency breakdowns
5. **Quality Metrics**: Coherence calibration, confidence accuracy, convergence rates
6. **System Health**: Memory pressure, concept churn, binding overhead

### Production Operability
7. Design alerting rules with provably correct thresholds (no false positive storms)
8. Implement Grafana dashboards with cognitive architecture organization
9. Add trace sampling for detailed failure mode debugging
10. Define metric retention and aggregation policies aligned with consolidation timescales
11. Implement cardinality protection (no label explosion from concept IDs)
12. Add capacity planning metrics (growth projections, saturation warnings)

### Performance Constraints
13. Metric overhead <1% of operation latency
14. Dashboard query latency <500ms for 1M datapoints
15. Alert evaluation <100ms per rule
16. Metric storage <5% of node storage

## Technical Specification

### Files to Create

#### `engram-core/src/metrics/dual_memory.rs`
Dual memory metric types with cache-optimal layout:

```rust
use crate::metrics::{increment_counter, observe_histogram, record_gauge};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Metric names for dual memory operations
///
/// Following Engram metric naming: engram_{subsystem}_{metric}_{unit}
pub mod metric_names {
    // Concept lifecycle
    pub const CONCEPTS_FORMED_TOTAL: &str = "engram_concepts_formed_total";
    pub const CONCEPT_FORMATION_DURATION_SECONDS: &str = "engram_concept_formation_duration_seconds";
    pub const CONCEPT_UPDATES_TOTAL: &str = "engram_concept_updates_total";
    pub const CONCEPT_DELETIONS_TOTAL: &str = "engram_concept_deletions_total";

    // Concept quality
    pub const CONCEPT_COHERENCE_SCORE: &str = "engram_concept_coherence_score";
    pub const CONCEPT_CLUSTER_SIZE: &str = "engram_concept_cluster_size";
    pub const CONCEPT_CONSOLIDATION_STRENGTH: &str = "engram_concept_consolidation_strength";
    pub const CONCEPT_AGE_SECONDS: &str = "engram_concept_age_seconds";

    // Binding dynamics
    pub const BINDINGS_CREATED_TOTAL: &str = "engram_bindings_created_total";
    pub const BINDINGS_STRENGTHENED_TOTAL: &str = "engram_bindings_strengthened_total";
    pub const BINDINGS_WEAKENED_TOTAL: &str = "engram_bindings_weakened_total";
    pub const BINDINGS_GC_REMOVED_TOTAL: &str = "engram_bindings_gc_removed_total";
    pub const BINDING_STRENGTH: &str = "engram_binding_strength";
    pub const BINDING_AGE_SECONDS: &str = "engram_binding_age_seconds";

    // Fan effect metrics
    pub const FAN_OUT_DISTRIBUTION: &str = "engram_fan_out_distribution";
    pub const FAN_IN_DISTRIBUTION: &str = "engram_fan_in_distribution";
    pub const FAN_PENALTY_MAGNITUDE: &str = "engram_fan_penalty_magnitude";
    pub const HIGH_FAN_NODES_TOTAL: &str = "engram_high_fan_nodes_total";
    pub const FAN_OUT_EXTREME_TOTAL: &str = "engram_fan_out_extreme_total";

    // Recall performance
    pub const EPISODIC_RECALL_DURATION_SECONDS: &str = "engram_episodic_recall_duration_seconds";
    pub const SEMANTIC_RECALL_DURATION_SECONDS: &str = "engram_semantic_recall_duration_seconds";
    pub const BLENDED_RECALL_DURATION_SECONDS: &str = "engram_blended_recall_duration_seconds";
    pub const PATTERN_COMPLETION_DURATION_SECONDS: &str = "engram_pattern_completion_duration_seconds";

    // Recall results
    pub const EPISODIC_RESULTS_COUNT: &str = "engram_episodic_results_count";
    pub const SEMANTIC_RESULTS_COUNT: &str = "engram_semantic_results_count";
    pub const BLENDED_RESULTS_COUNT: &str = "engram_blended_results_count";
    pub const CONVERGENT_RESULTS_RATIO: &str = "engram_convergent_results_ratio";

    // Pathway performance
    pub const EPISODIC_PATHWAY_CONFIDENCE: &str = "engram_episodic_pathway_confidence";
    pub const SEMANTIC_PATHWAY_CONFIDENCE: &str = "engram_semantic_pathway_confidence";
    pub const SEMANTIC_TIMEOUT_TOTAL: &str = "engram_semantic_timeout_total";
    pub const EPISODIC_FALLBACK_TOTAL: &str = "engram_episodic_fallback_total";

    // Quality metrics
    pub const CONFIDENCE_CALIBRATION_ERROR: &str = "engram_confidence_calibration_error";
    pub const RECALL_PRECISION: &str = "engram_recall_precision";
    pub const CONCEPT_QUALITY_VIOLATIONS_TOTAL: &str = "engram_concept_quality_violations_total";

    // System health
    pub const BINDING_MEMORY_OVERHEAD_RATIO: &str = "engram_binding_memory_overhead_ratio";
    pub const CONCEPT_CHURN_RATE: &str = "engram_concept_churn_rate";
    pub const CONSOLIDATION_CYCLE_CONCEPTS: &str = "engram_consolidation_cycle_concepts";
    pub const MEMORY_PRESSURE: &str = "engram_memory_pressure";
}

/// Concept formation result tracking
#[derive(Debug, Clone)]
pub struct ConceptFormationResult {
    pub concepts_formed: usize,
    pub duration: std::time::Duration,
    pub avg_cluster_size: f32,
    pub coherence_scores: Vec<f32>,
    pub consolidation_strengths: Vec<f32>,
}

impl ConceptFormationResult {
    /// Record metrics for this concept formation cycle
    pub fn record_metrics(&self) {
        use metric_names::*;

        // Counter: total concepts formed
        increment_counter(CONCEPTS_FORMED_TOTAL, self.concepts_formed as u64);

        // Histogram: formation duration
        observe_histogram(
            CONCEPT_FORMATION_DURATION_SECONDS,
            self.duration.as_secs_f64(),
        );

        // Gauge: average cluster size
        record_gauge(CONCEPT_CLUSTER_SIZE, self.avg_cluster_size as f64);

        // Histogram: coherence score distribution
        for &coherence in &self.coherence_scores {
            observe_histogram(CONCEPT_COHERENCE_SCORE, coherence as f64);
        }

        // Histogram: consolidation strength distribution
        for &strength in &self.consolidation_strengths {
            observe_histogram(CONCEPT_CONSOLIDATION_STRENGTH, strength as f64);
        }

        // Alert on poor quality concepts
        let low_coherence_count = self.coherence_scores.iter()
            .filter(|&&c| c < 0.5)
            .count();

        if low_coherence_count > self.concepts_formed / 2 {
            increment_counter(CONCEPT_QUALITY_VIOLATIONS_TOTAL, 1);
        }
    }
}

/// Binding operation result tracking
#[derive(Debug, Clone)]
pub struct BindingOperationResult {
    pub created: usize,
    pub strengthened: usize,
    pub weakened: usize,
    pub gc_removed: usize,
    pub current_strengths: Vec<f32>,
    pub binding_ages_seconds: Vec<f32>,
}

impl BindingOperationResult {
    /// Record metrics for binding operations
    pub fn record_metrics(&self) {
        use metric_names::*;

        // Counters
        increment_counter(BINDINGS_CREATED_TOTAL, self.created as u64);
        increment_counter(BINDINGS_STRENGTHENED_TOTAL, self.strengthened as u64);
        increment_counter(BINDINGS_WEAKENED_TOTAL, self.weakened as u64);
        increment_counter(BINDINGS_GC_REMOVED_TOTAL, self.gc_removed as u64);

        // Histograms: strength distribution
        for &strength in &self.current_strengths {
            observe_histogram(BINDING_STRENGTH, strength as f64);
        }

        // Histograms: age distribution
        for &age in &self.binding_ages_seconds {
            observe_histogram(BINDING_AGE_SECONDS, age as f64);
        }
    }
}

/// Fan effect observation
#[derive(Debug, Clone)]
pub struct FanEffectObservation {
    pub node_id: String,
    pub fan_out: usize,
    pub fan_in: usize,
    pub penalty: f32,
}

impl FanEffectObservation {
    /// Record fan effect metrics with threshold alerting
    pub fn record_metrics(&self) {
        use metric_names::*;

        // Histograms: fan distribution
        observe_histogram(FAN_OUT_DISTRIBUTION, self.fan_out as f64);
        observe_histogram(FAN_IN_DISTRIBUTION, self.fan_in as f64);
        observe_histogram(FAN_PENALTY_MAGNITUDE, self.penalty as f64);

        // Alert thresholds
        const HIGH_FAN_THRESHOLD: usize = 50;
        const EXTREME_FAN_THRESHOLD: usize = 200;

        if self.fan_out > HIGH_FAN_THRESHOLD {
            increment_counter(HIGH_FAN_NODES_TOTAL, 1);

            if self.fan_out > EXTREME_FAN_THRESHOLD {
                increment_counter(FAN_OUT_EXTREME_TOTAL, 1);
                tracing::warn!(
                    node_id = %self.node_id,
                    fan_out = self.fan_out,
                    penalty = self.penalty,
                    "Extreme fan-out detected - performance degradation likely"
                );
            }
        }
    }
}

/// Blended recall result tracking
#[derive(Debug, Clone)]
pub struct BlendedRecallMetrics {
    pub episodic_duration: std::time::Duration,
    pub semantic_duration: Option<std::time::Duration>,
    pub total_duration: std::time::Duration,
    pub episodic_results: usize,
    pub semantic_results: usize,
    pub blended_results: usize,
    pub convergent_results: usize,
    pub pattern_completed_results: usize,
    pub episodic_confidence: f32,
    pub semantic_confidence: Option<f32>,
    pub semantic_timed_out: bool,
    pub episodic_fallback: bool,
}

impl BlendedRecallMetrics {
    /// Record blended recall metrics
    pub fn record_metrics(&self) {
        use metric_names::*;

        // Latency histograms
        observe_histogram(
            EPISODIC_RECALL_DURATION_SECONDS,
            self.episodic_duration.as_secs_f64(),
        );

        if let Some(duration) = self.semantic_duration {
            observe_histogram(
                SEMANTIC_RECALL_DURATION_SECONDS,
                duration.as_secs_f64(),
            );
        }

        observe_histogram(
            BLENDED_RECALL_DURATION_SECONDS,
            self.total_duration.as_secs_f64(),
        );

        // Result counts
        record_gauge(EPISODIC_RESULTS_COUNT, self.episodic_results as f64);
        record_gauge(SEMANTIC_RESULTS_COUNT, self.semantic_results as f64);
        record_gauge(BLENDED_RESULTS_COUNT, self.blended_results as f64);

        // Convergence ratio (key quality metric)
        let convergence_ratio = if self.blended_results > 0 {
            self.convergent_results as f64 / self.blended_results as f64
        } else {
            0.0
        };
        record_gauge(CONVERGENT_RESULTS_RATIO, convergence_ratio);

        // Pathway confidence
        record_gauge(EPISODIC_PATHWAY_CONFIDENCE, self.episodic_confidence as f64);
        if let Some(confidence) = self.semantic_confidence {
            record_gauge(SEMANTIC_PATHWAY_CONFIDENCE, confidence as f64);
        }

        // Error conditions
        if self.semantic_timed_out {
            increment_counter(SEMANTIC_TIMEOUT_TOTAL, 1);
        }

        if self.episodic_fallback {
            increment_counter(EPISODIC_FALLBACK_TOTAL, 1);
        }

        // Alert: low convergence rate indicates pathway divergence
        if convergence_ratio < 0.3 && self.blended_results > 10 {
            tracing::warn!(
                convergence_ratio,
                episodic_results = self.episodic_results,
                semantic_results = self.semantic_results,
                "Low pathway convergence - dual memory systems may be drifting"
            );
        }
    }
}

/// System health snapshot
#[derive(Debug, Clone)]
pub struct DualMemoryHealthSnapshot {
    pub total_concepts: usize,
    pub total_bindings: usize,
    pub total_nodes: usize,
    pub binding_memory_bytes: usize,
    pub node_memory_bytes: usize,
    pub concepts_formed_last_hour: usize,
    pub concepts_deleted_last_hour: usize,
}

impl DualMemoryHealthSnapshot {
    /// Record system health metrics
    pub fn record_metrics(&self) {
        use metric_names::*;

        // Binding memory overhead
        let overhead_ratio = if self.node_memory_bytes > 0 {
            self.binding_memory_bytes as f64 / self.node_memory_bytes as f64
        } else {
            0.0
        };
        record_gauge(BINDING_MEMORY_OVERHEAD_RATIO, overhead_ratio);

        // Concept churn rate
        let churn_rate = if self.total_concepts > 0 {
            (self.concepts_formed_last_hour + self.concepts_deleted_last_hour) as f64
                / self.total_concepts as f64
        } else {
            0.0
        };
        record_gauge(CONCEPT_CHURN_RATE, churn_rate);

        // Memory pressure (binding overhead)
        record_gauge(MEMORY_PRESSURE, overhead_ratio);

        // Alert: excessive binding overhead
        if overhead_ratio > 0.25 {
            tracing::warn!(
                overhead_ratio,
                binding_memory_mb = self.binding_memory_bytes / 1_048_576,
                node_memory_mb = self.node_memory_bytes / 1_048_576,
                "Binding memory overhead exceeds 25% - consider GC tuning"
            );
        }

        // Alert: high concept churn
        if churn_rate > 0.1 {
            tracing::warn!(
                churn_rate,
                total_concepts = self.total_concepts,
                formed = self.concepts_formed_last_hour,
                deleted = self.concepts_deleted_last_hour,
                "High concept churn detected - consolidation may be unstable"
            );
        }
    }
}
```

#### `monitoring/dashboards/dual_memory.json`
Grafana dashboard with cognitive architecture organization:

```json
{
  "dashboard": {
    "title": "Dual Memory System - Cognitive Architecture",
    "tags": ["engram", "dual-memory", "cognitive"],
    "timezone": "browser",
    "schemaVersion": 38,
    "version": 1,
    "refresh": "10s",

    "panels": [
      {
        "title": "System Overview",
        "type": "row",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}
      },
      {
        "title": "Memory Pressure",
        "type": "gauge",
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 1},
        "targets": [{
          "expr": "engram_memory_pressure",
          "legendFormat": "Pressure"
        }],
        "options": {
          "thresholds": [
            {"value": 0, "color": "green"},
            {"value": 0.20, "color": "yellow"},
            {"value": 0.25, "color": "red"}
          ]
        }
      },
      {
        "title": "Binding Memory Overhead",
        "type": "stat",
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 1},
        "targets": [{
          "expr": "engram_binding_memory_overhead_ratio * 100",
          "legendFormat": "Overhead %"
        }],
        "options": {
          "unit": "percent"
        }
      },
      {
        "title": "Concept Churn Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 1},
        "targets": [{
          "expr": "rate(engram_concepts_formed_total[5m])",
          "legendFormat": "Formation rate"
        }, {
          "expr": "rate(engram_concepts_deleted_total[5m])",
          "legendFormat": "Deletion rate"
        }]
      },

      {
        "title": "Concept Formation",
        "type": "row",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 9}
      },
      {
        "title": "Concept Formation Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 10},
        "targets": [{
          "expr": "rate(engram_concepts_formed_total[5m]) * 3600",
          "legendFormat": "Concepts/hour"
        }],
        "alert": {
          "name": "High Concept Formation Rate",
          "conditions": [{
            "evaluator": {"type": "gt", "params": [100]},
            "operator": {"type": "and"},
            "query": {"params": ["A", "5m", "now"]},
            "reducer": {"type": "avg"}
          }],
          "frequency": "1m",
          "for": "5m",
          "message": "Excessive concept formation detected (>100/hour). Check consolidation scheduler."
        }
      },
      {
        "title": "Concept Quality Distribution",
        "type": "heatmap",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 10},
        "targets": [{
          "expr": "histogram_quantile(0.5, rate(engram_concept_coherence_score_bucket[5m]))",
          "legendFormat": "P50 Coherence"
        }, {
          "expr": "histogram_quantile(0.95, rate(engram_concept_coherence_score_bucket[5m]))",
          "legendFormat": "P95 Coherence"
        }],
        "alert": {
          "name": "Low Concept Quality",
          "conditions": [{
            "evaluator": {"type": "lt", "params": [0.5]},
            "operator": {"type": "and"},
            "query": {"params": ["A", "5m", "now"]},
            "reducer": {"type": "avg"}
          }],
          "frequency": "1m",
          "for": "10m",
          "message": "Concept quality degraded (P50 coherence <0.5). Review clustering parameters."
        }
      },

      {
        "title": "Binding Dynamics",
        "type": "row",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 18}
      },
      {
        "title": "Binding Operations",
        "type": "graph",
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 19},
        "targets": [{
          "expr": "rate(engram_bindings_created_total[5m])",
          "legendFormat": "Created"
        }, {
          "expr": "rate(engram_bindings_strengthened_total[5m])",
          "legendFormat": "Strengthened"
        }, {
          "expr": "rate(engram_bindings_weakened_total[5m])",
          "legendFormat": "Weakened"
        }, {
          "expr": "rate(engram_bindings_gc_removed_total[5m])",
          "legendFormat": "GC Removed"
        }]
      },
      {
        "title": "Binding Strength Distribution",
        "type": "histogram",
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 19},
        "targets": [{
          "expr": "histogram_quantile(0.5, rate(engram_binding_strength_bucket[5m]))",
          "legendFormat": "P50"
        }, {
          "expr": "histogram_quantile(0.90, rate(engram_binding_strength_bucket[5m]))",
          "legendFormat": "P90"
        }, {
          "expr": "histogram_quantile(0.99, rate(engram_binding_strength_bucket[5m]))",
          "legendFormat": "P99"
        }]
      },
      {
        "title": "Binding Age Distribution",
        "type": "graph",
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 19},
        "targets": [{
          "expr": "histogram_quantile(0.5, rate(engram_binding_age_seconds_bucket[5m])) / 3600",
          "legendFormat": "P50 Age (hours)"
        }, {
          "expr": "histogram_quantile(0.95, rate(engram_binding_age_seconds_bucket[5m])) / 3600",
          "legendFormat": "P95 Age (hours)"
        }]
      },

      {
        "title": "Fan Effect Impact",
        "type": "row",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 27}
      },
      {
        "title": "Fan-Out Distribution",
        "type": "histogram",
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 28},
        "targets": [{
          "expr": "histogram_quantile(0.5, rate(engram_fan_out_distribution_bucket[5m]))",
          "legendFormat": "P50"
        }, {
          "expr": "histogram_quantile(0.90, rate(engram_fan_out_distribution_bucket[5m]))",
          "legendFormat": "P90"
        }, {
          "expr": "histogram_quantile(0.99, rate(engram_fan_out_distribution_bucket[5m]))",
          "legendFormat": "P99"
        }]
      },
      {
        "title": "High Fan-Out Nodes",
        "type": "stat",
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 28},
        "targets": [{
          "expr": "engram_high_fan_nodes_total",
          "legendFormat": "High Fan (>50)"
        }, {
          "expr": "engram_fan_out_extreme_total",
          "legendFormat": "Extreme (>200)"
        }],
        "alert": {
          "name": "Excessive Fan-Out",
          "conditions": [{
            "evaluator": {"type": "gt", "params": [10]},
            "operator": {"type": "and"},
            "query": {"params": ["B", "5m", "now"]},
            "reducer": {"type": "last"}
          }],
          "frequency": "1m",
          "for": "5m",
          "message": "Multiple nodes with extreme fan-out (>200). Performance degradation imminent."
        }
      },
      {
        "title": "Fan Penalty Magnitude",
        "type": "graph",
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 28},
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(engram_fan_penalty_magnitude_bucket[5m]))",
          "legendFormat": "P95 Penalty"
        }]
      },

      {
        "title": "Recall Performance",
        "type": "row",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 36}
      },
      {
        "title": "Recall Latency by Type",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 37},
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(engram_episodic_recall_duration_seconds_bucket[5m])) * 1000",
          "legendFormat": "Episodic P95 (ms)"
        }, {
          "expr": "histogram_quantile(0.95, rate(engram_semantic_recall_duration_seconds_bucket[5m])) * 1000",
          "legendFormat": "Semantic P95 (ms)"
        }, {
          "expr": "histogram_quantile(0.95, rate(engram_blended_recall_duration_seconds_bucket[5m])) * 1000",
          "legendFormat": "Blended P95 (ms)"
        }],
        "alert": {
          "name": "Recall Latency Regression",
          "conditions": [{
            "evaluator": {"type": "gt", "params": [15]},
            "operator": {"type": "and"},
            "query": {"params": ["C", "5m", "now"]},
            "reducer": {"type": "avg"}
          }],
          "frequency": "1m",
          "for": "5m",
          "message": "Blended recall P95 latency exceeds 15ms budget. Investigate spreading engine."
        }
      },
      {
        "title": "Pathway Convergence Rate",
        "type": "gauge",
        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 37},
        "targets": [{
          "expr": "engram_convergent_results_ratio",
          "legendFormat": "Convergence"
        }],
        "options": {
          "thresholds": [
            {"value": 0, "color": "red"},
            {"value": 0.3, "color": "yellow"},
            {"value": 0.5, "color": "green"}
          ]
        },
        "alert": {
          "name": "Low Pathway Convergence",
          "conditions": [{
            "evaluator": {"type": "lt", "params": [0.3]},
            "operator": {"type": "and"},
            "query": {"params": ["A", "10m", "now"]},
            "reducer": {"type": "avg"}
          }],
          "frequency": "1m",
          "for": "15m",
          "message": "Episodic and semantic pathways diverging (convergence <30%). Dual memory systems may be drifting."
        }
      },
      {
        "title": "Pathway Confidence",
        "type": "graph",
        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 37},
        "targets": [{
          "expr": "engram_episodic_pathway_confidence",
          "legendFormat": "Episodic"
        }, {
          "expr": "engram_semantic_pathway_confidence",
          "legendFormat": "Semantic"
        }]
      },

      {
        "title": "Capacity Planning",
        "type": "row",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 45}
      },
      {
        "title": "Concept Growth Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 46},
        "targets": [{
          "expr": "deriv(engram_concepts_formed_total[1h]) * 86400",
          "legendFormat": "Concepts/day (projected)"
        }, {
          "expr": "predict_linear(engram_concepts_formed_total[7d], 86400 * 30)",
          "legendFormat": "30-day projection"
        }]
      },
      {
        "title": "Memory Growth Projection",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 46},
        "targets": [{
          "expr": "engram_binding_memory_overhead_ratio",
          "legendFormat": "Current Overhead"
        }, {
          "expr": "predict_linear(engram_binding_memory_overhead_ratio[7d], 86400 * 30)",
          "legendFormat": "30-day projection"
        }]
      }
    ]
  }
}
```

#### `monitoring/alerts/dual_memory_alerts.yaml`
Prometheus alerting rules with provably correct thresholds:

```yaml
groups:
  - name: dual_memory_alerts
    interval: 30s
    rules:
      # Concept Quality Alerts
      - alert: HighConceptFormationRate
        expr: rate(engram_concepts_formed_total[5m]) * 3600 > 100
        for: 5m
        labels:
          severity: warning
          subsystem: consolidation
        annotations:
          summary: "Excessive concept formation detected"
          description: "Forming {{ $value | humanize }} concepts/hour (threshold: 100/hour). Consolidation scheduler may be over-eager."
          runbook: "https://engram.docs/runbooks/high-concept-formation"

      - alert: LowConceptQuality
        expr: histogram_quantile(0.5, rate(engram_concept_coherence_score_bucket[5m])) < 0.5
        for: 10m
        labels:
          severity: warning
          subsystem: consolidation
        annotations:
          summary: "Concept quality degradation detected"
          description: "P50 coherence is {{ $value | humanize2 }} (threshold: 0.5). Review clustering parameters."
          runbook: "https://engram.docs/runbooks/low-concept-quality"

      - alert: ConceptQualityViolations
        expr: rate(engram_concept_quality_violations_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
          subsystem: consolidation
        annotations:
          summary: "Frequent concept quality violations"
          description: "{{ $value | humanize }} quality violations/sec. Consolidation may be creating low-quality concepts."
          runbook: "https://engram.docs/runbooks/concept-quality-violations"

      # Fan Effect Alerts
      - alert: ExcessiveHighFanNodes
        expr: engram_high_fan_nodes_total > 10
        for: 5m
        labels:
          severity: warning
          subsystem: spreading
        annotations:
          summary: "Multiple high fan-out nodes detected"
          description: "{{ $value }} nodes with >50 bindings. Performance may degrade."
          runbook: "https://engram.docs/runbooks/high-fan-out"

      - alert: ExtremeFanOut
        expr: engram_fan_out_extreme_total > 3
        for: 2m
        labels:
          severity: critical
          subsystem: spreading
        annotations:
          summary: "Extreme fan-out nodes present"
          description: "{{ $value }} nodes with >200 bindings. Severe performance degradation likely."
          runbook: "https://engram.docs/runbooks/extreme-fan-out"

      # Recall Performance Alerts
      - alert: RecallLatencyRegression
        expr: histogram_quantile(0.95, rate(engram_blended_recall_duration_seconds_bucket[5m])) * 1000 > 15
        for: 5m
        labels:
          severity: warning
          subsystem: recall
        annotations:
          summary: "Blended recall latency exceeds budget"
          description: "P95 latency is {{ $value | humanize }}ms (budget: 15ms). Investigate spreading engine."
          runbook: "https://engram.docs/runbooks/recall-latency-regression"

      - alert: SemanticPathwayDegradation
        expr: rate(engram_semantic_timeout_total[5m]) > 0.2
        for: 5m
        labels:
          severity: warning
          subsystem: recall
        annotations:
          summary: "Frequent semantic pathway timeouts"
          description: "{{ $value | humanizePercentage }} of recalls timing out semantic pathway. Check concept index performance."
          runbook: "https://engram.docs/runbooks/semantic-pathway-timeout"

      # Pathway Convergence Alerts
      - alert: LowPathwayConvergence
        expr: engram_convergent_results_ratio < 0.3
        for: 15m
        labels:
          severity: warning
          subsystem: recall
        annotations:
          summary: "Episodic and semantic pathways diverging"
          description: "Convergence ratio is {{ $value | humanizePercentage }} (threshold: 30%). Dual memory systems may be drifting."
          runbook: "https://engram.docs/runbooks/pathway-divergence"

      # Memory Overhead Alerts
      - alert: ExcessiveBindingOverhead
        expr: engram_binding_memory_overhead_ratio > 0.25
        for: 10m
        labels:
          severity: warning
          subsystem: memory
        annotations:
          summary: "Binding memory overhead exceeds threshold"
          description: "Overhead is {{ $value | humanizePercentage }} (threshold: 25%). Consider aggressive GC or binding pruning."
          runbook: "https://engram.docs/runbooks/binding-overhead"

      - alert: HighMemoryPressure
        expr: engram_memory_pressure > 0.25
        for: 5m
        labels:
          severity: critical
          subsystem: memory
        annotations:
          summary: "Memory pressure critical"
          description: "Pressure at {{ $value | humanizePercentage }}. System may OOM soon."
          runbook: "https://engram.docs/runbooks/memory-pressure"

      # System Health Alerts
      - alert: HighConceptChurn
        expr: engram_concept_churn_rate > 0.1
        for: 10m
        labels:
          severity: warning
          subsystem: consolidation
        annotations:
          summary: "High concept churn detected"
          description: "Churn rate is {{ $value | humanizePercentage }}/hour (threshold: 10%). Consolidation may be unstable."
          runbook: "https://engram.docs/runbooks/concept-churn"

      # Capacity Planning Alerts
      - alert: MemoryGrowthProjection
        expr: predict_linear(engram_binding_memory_overhead_ratio[7d], 86400 * 7) > 0.3
        for: 1h
        labels:
          severity: info
          subsystem: capacity
        annotations:
          summary: "Memory overhead will exceed 30% in 7 days"
          description: "Projected overhead: {{ $value | humanizePercentage }}. Plan capacity expansion or tuning."
          runbook: "https://engram.docs/runbooks/capacity-planning"
```

### Integration Points

#### Update `engram-core/src/consolidation/concept_formation.rs`:
```rust
use crate::metrics::dual_memory::{ConceptFormationResult, metric_names};

impl ConceptFormationEngine {
    pub fn form_concepts(&self, episodes: &[Episode], sleep_stage: SleepStage) -> Vec<ProtoConcept> {
        let start = Instant::now();

        // ... existing concept formation logic ...

        // Record metrics
        let result = ConceptFormationResult {
            concepts_formed: concepts.len(),
            duration: start.elapsed(),
            avg_cluster_size: concepts.iter()
                .map(|c| c.episode_indices.len() as f32)
                .sum::<f32>() / concepts.len() as f32,
            coherence_scores: concepts.iter().map(|c| c.coherence_score).collect(),
            consolidation_strengths: concepts.iter().map(|c| c.consolidation_strength).collect(),
        };

        result.record_metrics();

        concepts
    }
}
```

#### Update `engram-core/src/memory/bindings.rs`:
```rust
use crate::metrics::dual_memory::{BindingOperationResult, FanEffectObservation};

impl BindingIndex {
    pub fn garbage_collect(&self) -> usize {
        let start = Instant::now();

        // ... existing GC logic ...

        // Record metrics
        let result = BindingOperationResult {
            created: 0,
            strengthened: 0,
            weakened: 0,
            gc_removed: removed,
            current_strengths: self.sample_strengths(),
            binding_ages_seconds: self.sample_ages(),
        };

        result.record_metrics();

        removed
    }

    fn record_fan_effect(&self, node_id: &str, fan_out: usize, fan_in: usize, penalty: f32) {
        let observation = FanEffectObservation {
            node_id: node_id.to_string(),
            fan_out,
            fan_in,
            penalty,
        };
        observation.record_metrics();
    }
}
```

#### Update `engram-core/src/activation/blended_recall.rs`:
```rust
use crate::metrics::dual_memory::BlendedRecallMetrics;

impl BlendedRecallEngine {
    pub fn recall_blended(&self, cue: &Cue, store: &MemoryStore) -> ActivationResult<Vec<BlendedRankedMemory>> {
        let start = Instant::now();

        // ... existing blended recall logic ...

        // Record metrics
        let metrics = BlendedRecallMetrics {
            episodic_duration: episodic_pathway.latency,
            semantic_duration: semantic_pathway.as_ref().map(|p| p.latency),
            total_duration: start.elapsed(),
            episodic_results: episodic_pathway.results.len(),
            semantic_results: semantic_pathway.as_ref().map(|p| p.episode_scores.len()).unwrap_or(0),
            blended_results: final_results.len(),
            convergent_results: final_results.iter().filter(|r| r.is_convergent()).count(),
            pattern_completed_results: final_results.iter()
                .filter(|r| matches!(r.provenance.final_source, RecallSource::PatternCompleted))
                .count(),
            episodic_confidence: episodic_pathway.confidence.raw(),
            semantic_confidence: semantic_pathway.as_ref().map(|p| p.confidence),
            semantic_timed_out: semantic_pathway.is_none() && /* timeout condition */,
            episodic_fallback: /* fallback condition */,
        };

        metrics.record_metrics();

        Ok(final_results)
    }
}
```

## Metric Retention and Aggregation Policies

### Retention Windows
Aligned with consolidation timescales (Takashima et al., 2006):

- **High-resolution**: 24 hours (consolidation window)
- **Medium-resolution**: 7 days (initial consolidation)
- **Low-resolution**: 90 days (remote memory formation)
- **Archive**: 1 year (longitudinal trends)

### Aggregation Rules
```yaml
# prometheus.yml
global:
  scrape_interval: 10s
  evaluation_interval: 30s

scrape_configs:
  - job_name: 'engram'
    static_configs:
      - targets: ['localhost:9090']

# Downsampling for long-term storage
remote_write:
  - url: http://thanos:19291/api/v1/receive
    write_relabel_configs:
      # Keep high-res for 24h
      - source_labels: [__name__]
        regex: 'engram_.*'
        action: keep
        target_label: __tmp_resolution
        replacement: '10s'

      # Medium-res (1m) for 7d
      - source_labels: [__name__]
        regex: 'engram_.*'
        action: keep
        target_label: __tmp_resolution
        replacement: '1m'

      # Low-res (5m) for 90d
      - source_labels: [__name__]
        regex: 'engram_.*'
        action: keep
        target_label: __tmp_resolution
        replacement: '5m'
```

## Cardinality Protection

### Label Discipline
**NEVER label by**:
- Concept IDs (unbounded cardinality)
- Episode IDs (unbounded cardinality)
- Binding IDs (unbounded cardinality)
- User queries (unbounded cardinality)

**ALWAYS label by**:
- Memory space ID (bounded by tenant count)
- Node type (enum: episode, concept)
- Pathway type (enum: episodic, semantic, blended)
- Tier (enum: hot, warm, cold)

### Cardinality Estimation
For N memory spaces:
- Base metrics: ~100 metrics
- Per-space metrics: ~20 metrics
- Total: 100 + (20 × N)

With 1000 memory spaces: 20,100 time series (acceptable)

### Cardinality Monitoring
```yaml
# Alert on cardinality explosion
- alert: MetricCardinalityExplosion
  expr: count(count by(__name__) (engram_concepts_formed_total)) > 10000
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Metric cardinality exceeds safe limits"
    description: "{{ $value }} unique metric series. Label explosion likely."
```

## Trace Sampling for Detailed Debugging

### OpenTelemetry Integration
```rust
use opentelemetry::trace::{Tracer, SpanKind};

impl ConceptFormationEngine {
    #[tracing::instrument(
        name = "concept_formation",
        skip(self, episodes),
        fields(
            episode_count = episodes.len(),
            sleep_stage = ?sleep_stage,
        )
    )]
    pub fn form_concepts(&self, episodes: &[Episode], sleep_stage: SleepStage) -> Vec<ProtoConcept> {
        let span = tracing::Span::current();

        // ... concept formation logic ...

        span.record("concepts_formed", concepts.len());
        span.record("avg_coherence", avg_coherence);

        concepts
    }
}
```

### Sampling Strategy
- **Always sample**: Errors, quality violations, extreme fan-out
- **Head-based sampling**: 1% of normal operations
- **Tail-based sampling**: P99 latency operations

```yaml
# otel-collector-config.yaml
processors:
  probabilistic_sampler:
    sampling_percentage: 1.0

  tail_sampling:
    decision_wait: 10s
    policies:
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]

      - name: high_latency
        type: latency
        latency:
          threshold_ms: 15

      - name: quality_violations
        type: string_attribute
        string_attribute:
          key: violation_type
          values: [coherence, convergence, fan_out]
```

## Capacity Planning Metrics

### Growth Rate Tracking
```promql
# Concept formation velocity
deriv(engram_concepts_formed_total[1h]) * 86400  # concepts/day

# Binding memory growth rate
deriv(engram_binding_memory_overhead_ratio[7d])  # overhead change/sec

# 30-day projection
predict_linear(engram_binding_memory_overhead_ratio[7d], 86400 * 30)
```

### Saturation Indicators
```promql
# Memory saturation (overhead approaching limit)
engram_binding_memory_overhead_ratio / 0.25  # % of budget consumed

# Concept formation saturation (approaching spindle limit)
rate(engram_concepts_formed_total[5m]) * 3600 / 100  # % of capacity

# Fan-out saturation (nodes approaching extreme threshold)
engram_high_fan_nodes_total / 10  # % of tolerance consumed
```

## Example Prometheus Queries for Common Questions

### Operational Queries
```promql
# What is the current concept formation rate?
rate(engram_concepts_formed_total[5m]) * 3600

# What percentage of bindings are being garbage collected?
rate(engram_bindings_gc_removed_total[5m]) / rate(engram_bindings_created_total[5m])

# What is the P95 recall latency by pathway?
histogram_quantile(0.95, rate(engram_episodic_recall_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(engram_semantic_recall_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(engram_blended_recall_duration_seconds_bucket[5m]))

# What is the pathway convergence rate?
engram_convergent_results_ratio

# How many high fan-out nodes exist?
engram_high_fan_nodes_total
engram_fan_out_extreme_total
```

### Debugging Queries
```promql
# Identify concept quality degradation
histogram_quantile(0.5, rate(engram_concept_coherence_score_bucket[5m])) < 0.5

# Find fan-out distribution anomalies
histogram_quantile(0.99, rate(engram_fan_out_distribution_bucket[5m])) > 100

# Detect semantic pathway instability
rate(engram_semantic_timeout_total[5m]) / rate(engram_blended_recall_duration_seconds_count[5m])

# Measure binding overhead growth
deriv(engram_binding_memory_overhead_ratio[1h]) * 86400  # daily change
```

### Capacity Planning Queries
```promql
# Project binding memory overhead 30 days out
predict_linear(engram_binding_memory_overhead_ratio[7d], 86400 * 30)

# Estimate time until memory saturation (25% overhead limit)
(0.25 - engram_binding_memory_overhead_ratio) /
  deriv(engram_binding_memory_overhead_ratio[7d])
  / 86400  # days remaining

# Project concept count growth
predict_linear(engram_concepts_formed_total[7d], 86400 * 30)
```

## Implementation Notes

### Metric Overhead Budget
Target: <1% of operation latency

- Atomic counter increment: ~10ns
- Histogram observation: ~100ns
- Gauge update: ~10ns
- Total per concept formation: ~500ns (0.05% of 1ms formation time)

### Dashboard Query Optimization
- Use recording rules for expensive queries
- Pre-aggregate common percentiles
- Limit time range to relevant consolidation windows

### Alert Fatigue Prevention
- **Group related alerts**: Fan-out alerts grouped under spreading subsystem
- **Escalation hierarchy**: Warning (10m) → Critical (5m)
- **Rate limiting**: Max 1 alert per 15m per rule
- **Runbook linking**: Every alert has actionable runbook

### Testing Strategy
1. **Unit tests**: Metric recording correctness
2. **Load tests**: Metric overhead measurement
3. **Synthetic anomalies**: Alert threshold validation
4. **Cardinality tests**: Label explosion detection

## Testing Approach

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::metrics;

    #[test]
    fn test_concept_formation_metrics() {
        let registry = metrics().unwrap();

        let result = ConceptFormationResult {
            concepts_formed: 5,
            duration: Duration::from_millis(10),
            avg_cluster_size: 3.5,
            coherence_scores: vec![0.8, 0.7, 0.9, 0.6, 0.75],
            consolidation_strengths: vec![0.02, 0.02, 0.02, 0.02, 0.02],
        };

        result.record_metrics();

        // Verify counter
        assert_eq!(registry.counter_value("engram_concepts_formed_total"), 5);

        // Verify histogram
        let coherence_quantiles = registry.histogram_quantiles(
            "engram_concept_coherence_score",
            &[0.5, 0.9],
        );
        assert!(coherence_quantiles[0] > 0.7);  // P50
    }

    #[test]
    fn test_fan_effect_alerting() {
        let observation = FanEffectObservation {
            node_id: "test_node".to_string(),
            fan_out: 250,
            fan_in: 10,
            penalty: 0.8,
        };

        observation.record_metrics();

        // Verify alert counters triggered
        let registry = metrics().unwrap();
        assert_eq!(registry.counter_value("engram_high_fan_nodes_total"), 1);
        assert_eq!(registry.counter_value("engram_fan_out_extreme_total"), 1);
    }

    #[test]
    fn test_metric_overhead() {
        let start = Instant::now();

        for _ in 0..10000 {
            increment_counter("test_counter", 1);
        }

        let elapsed = start.elapsed();
        let avg_overhead = elapsed.as_nanos() / 10000;

        // Overhead should be <100ns per increment
        assert!(avg_overhead < 100);
    }
}
```

### Integration Tests
```rust
#[test]
fn test_dashboard_query_performance() {
    // Load 1M metric samples
    load_test_metrics(1_000_000);

    // Query dashboard panels
    let queries = vec![
        "rate(engram_concepts_formed_total[5m])",
        "histogram_quantile(0.95, rate(engram_blended_recall_duration_seconds_bucket[5m]))",
        "engram_convergent_results_ratio",
    ];

    for query in queries {
        let start = Instant::now();
        execute_prometheus_query(query);
        let latency = start.elapsed();

        // Query latency should be <500ms
        assert!(latency < Duration::from_millis(500));
    }
}
```

### Alert Validation
```rust
#[test]
fn test_alert_thresholds() {
    // Simulate normal operation
    simulate_normal_concept_formation(duration: 1h);
    assert_no_alerts_fired();

    // Simulate excessive formation
    simulate_high_concept_formation(rate: 150/hour, duration: 6m);
    assert_alert_fired("HighConceptFormationRate");

    // Simulate recovery
    simulate_normal_concept_formation(duration: 10m);
    assert_alert_resolved("HighConceptFormationRate");
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] All dual memory operations have instrumentation
- [ ] Metrics record correctly with <1% overhead
- [ ] Grafana dashboard renders all panels without errors
- [ ] Alerts trigger correctly based on thresholds
- [ ] Prometheus queries execute <500ms for 1M samples
- [ ] Cardinality stays <20K series with 1000 memory spaces

### Cognitive Correctness
- [ ] Concept quality metrics align with coherence thresholds (0.65)
- [ ] Fan effect metrics match spreading activation penalties
- [ ] Pathway convergence ratios reflect dual-process theory
- [ ] Confidence calibration errors track true accuracy

### Production Operability
- [ ] All alerts have runbooks with mitigation steps
- [ ] Capacity planning queries project 30-day saturation
- [ ] Trace sampling captures quality violations
- [ ] Dashboard loads <2s with 30d time range

### Performance
- [ ] Metric recording overhead <1% of operation latency
- [ ] Dashboard query latency P95 <500ms
- [ ] Alert evaluation <100ms per rule
- [ ] Metric storage <5% of node storage

## Dependencies
- Task 004 (Concept Formation) - Concept lifecycle metrics
- Task 005 (Binding Formation) - Binding dynamics metrics
- Task 007 (Fan Effect) - Fan-out penalty metrics
- Task 009 (Blended Recall) - Pathway convergence metrics
- Existing metrics infrastructure (engram-core/src/metrics/)

## Estimated Time
2 days

### Day 1: Metric Types and Integration
- Implement dual_memory.rs metric types
- Add instrumentation to concept formation, binding operations
- Unit tests for metric recording

### Day 2: Dashboards and Alerts
- Create dual_memory.json Grafana dashboard
- Define alerting rules with thresholds
- Integration tests for dashboard queries and alert firing
- Documentation and runbooks

## References

### Cognitive Foundations
- Takashima et al. (2006) - Consolidation timescales for retention windows
- Kahneman (2011) - Dual-process theory for pathway metrics
- McClelland et al. (1995) - CLS theory for convergence metrics

### Observability Best Practices
- Google SRE Book - Monitoring distributed systems
- Prometheus Best Practices - Metric naming, cardinality
- Grafana Dashboard Design - Panel organization, query optimization

### Production Metrics
- Netflix Hystrix - Circuit breaker metrics
- Uber M3 - Metric aggregation policies
- Datadog APM - Trace sampling strategies
