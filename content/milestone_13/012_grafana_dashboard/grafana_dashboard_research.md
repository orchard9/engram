# Grafana Dashboard: Research and Technical Foundation

## Visualizing Cognitive Metrics

Task 001 implements zero-overhead metrics collection. Task 011 implements cognitive tracing. This task creates Grafana dashboards that visualize cognitive phenomena in real-time, enabling operators to monitor biological plausibility in production.

## Dashboard Categories

**Dashboard 1: Cognitive Pattern Health**
- Priming effect magnitudes (current vs expected from Neely 1977)
- Interference rates (PI/RI/fan percentages)
- False memory generation rate (DRM paradigm adherence)
- Spacing effect magnitude (comparison to Cepeda et al. 2006)

**Dashboard 2: Memory Operations**
- Encoding rate (items/sec)
- Retrieval latency (p50, p95, p99)
- Consolidation queue depth
- Reconsolidation window count (active lability windows)

**Dashboard 3: Graph Statistics**
- Node count, edge count
- Fan distribution (histogram of node fan-out)
- Activation spreading depth distribution
- Similarity computation cache hit rate

**Dashboard 4: System Performance**
- Memory usage (by subsystem)
- CPU utilization (by cognitive operation)
- Lock contention (should be near-zero with lock-free design)
- Metric collection overhead (should be <1%)

## Prometheus Metrics Export

Metrics from Task 001 are exported in Prometheus format:

```rust
pub struct PrometheusExporter {
    registry: Registry,
}

impl PrometheusExporter {
    pub fn register_cognitive_metrics(&mut self) {
        // Priming metrics
        self.registry.register(
            "engram_priming_boost_magnitude",
            "Distribution of priming boost values",
            HistogramVec::new(
                vec![0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
                vec!["priming_type"]  // semantic, repetition, associative
            )
        );

        // Interference metrics
        self.registry.register(
            "engram_interference_reduction_percent",
            "Recall reduction percentage due to interference",
            HistogramVec::new(
                vec![0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0],
                vec!["interference_type"]  // PI, RI, fan
            )
        );

        // False memory metrics
        self.registry.register(
            "engram_drm_false_recall_rate",
            "Percentage of critical lures falsely recalled",
            Gauge::new("rate", "Gauge")
        );

        // Performance metrics
        self.registry.register(
            "engram_retrieval_latency_microseconds",
            "Retrieval operation latency",
            Histogram::new(vec![10, 50, 100, 200, 500, 1000, 2000, 5000])
        );
    }

    pub fn export(&self) -> String {
        // Generate Prometheus text format
        let mut buffer = String::new();

        buffer.push_str("# HELP engram_priming_boost_magnitude Distribution of priming boost values\n");
        buffer.push_str("# TYPE engram_priming_boost_magnitude histogram\n");

        for (labels, histogram) in self.registry.histograms() {
            for (bucket, count) in histogram {
                buffer.push_str(&format!(
                    "engram_priming_boost_magnitude_bucket{{priming_type=\"{}\"}} {}\n",
                    labels.get("priming_type").unwrap(),
                    count
                ));
            }
        }

        // ... export all metrics ...

        buffer
    }
}
```

## Grafana Panel Configurations

**Panel 1: Priming Magnitude vs Expected**
```json
{
  "title": "Semantic Priming: Observed vs Neely (1977)",
  "targets": [
    {
      "expr": "histogram_quantile(0.50, engram_priming_boost_magnitude{priming_type=\"semantic\"})",
      "legendFormat": "Observed (p50)"
    },
    {
      "expr": "0.35",
      "legendFormat": "Expected (Neely 1977: 30-50ms on 1000ms baseline)"
    }
  ],
  "thresholds": [
    { "value": 0.30, "color": "green" },
    { "value": 0.20, "color": "yellow" },
    { "value": 0.0, "color": "red" }
  ]
}
```

**Panel 2: DRM False Recall Rate**
```json
{
  "title": "DRM False Memory Rate vs Roediger & McDermott (1995)",
  "targets": [
    {
      "expr": "engram_drm_false_recall_rate",
      "legendFormat": "Observed Rate"
    }
  ],
  "thresholds": [
    { "value": 0.55, "color": "green" },
    { "value": 0.45, "color": "yellow" },
    { "value": 0.0, "color": "red" }
  ],
  "min": 0.0,
  "max": 1.0,
  "expectedRange": { "min": 0.55, "max": 0.65 }
}
```

**Panel 3: Fan Effect Latency**
```json
{
  "title": "Fan Effect: Latency by Fan Size (Anderson 1974)",
  "targets": [
    {
      "expr": "histogram_quantile(0.95, engram_retrieval_latency_microseconds{fan=\"1\"})",
      "legendFormat": "Fan 1 (baseline)"
    },
    {
      "expr": "histogram_quantile(0.95, engram_retrieval_latency_microseconds{fan=\"2\"})",
      "legendFormat": "Fan 2 (expect +100-150ms)"
    },
    {
      "expr": "histogram_quantile(0.95, engram_retrieval_latency_microseconds{fan=\"3\"})",
      "legendFormat": "Fan 3 (expect +200-300ms)"
    }
  ]
}
```

## Alerting Rules

Define alerts for when cognitive patterns deviate from expected:

```yaml
groups:
  - name: cognitive_plausibility
    rules:
      - alert: DRM_FalseRecallOutOfRange
        expr: engram_drm_false_recall_rate < 0.45 OR engram_drm_false_recall_rate > 0.75
        for: 10m
        annotations:
          summary: "DRM false recall rate outside expected 55-65% ± 10%"
          description: "Current rate: {{ $value }}, expected: 0.55-0.65 (Roediger & McDermott 1995)"

      - alert: PrimingEffectWeak
        expr: histogram_quantile(0.50, engram_priming_boost_magnitude{priming_type=\"semantic\"}) < 0.20
        for: 5m
        annotations:
          summary: "Semantic priming effect weaker than expected"
          description: "Median boost: {{ $value }}, expected: 0.30-0.50 (Neely 1977)"

      - alert: InterferenceExcessive
        expr: histogram_quantile(0.95, engram_interference_reduction_percent{interference_type=\"RI\"}) > 0.60
        for: 10m
        annotations:
          summary: "Retroactive interference exceeds published ranges"
          description: "95th percentile: {{ $value }}, expected: <0.50 (Postman & Underwood 1973)"
```

## Deployment Configuration

```yaml
# docker-compose.yml
version: '3'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
    volumes:
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
      - grafana-data:/var/lib/grafana

  engram:
    build: .
    ports:
      - "8080:8080"  # HTTP API
      - "9091:9091"  # Metrics endpoint
    environment:
      - RUST_LOG=info
      - ENGRAM_FEATURES=metrics,tracing
```

## Dashboard JSON Export

Provide pre-configured dashboard JSON for import:

```json
{
  "dashboard": {
    "title": "Engram Cognitive Patterns",
    "panels": [
      /* Panel configurations... */
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
```

Users can import this JSON into Grafana for instant visualization of cognitive phenomena.

## Performance Impact

Dashboard itself has zero performance impact - it queries Prometheus, which scrapes metrics endpoint periodically (default: 15s interval).

Metrics collection overhead (from Task 001): <1% when enabled, 0% when disabled.
