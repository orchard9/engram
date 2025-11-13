# Task 002: Extended Soak Testing Infrastructure

**Status**: Pending
**Estimated Duration**: 4-5 days
**Priority**: High - Critical for production readiness

## Objective

Build 24+ hour sustained load testing infrastructure with automated memory leak detection, latency drift analysis, and resource exhaustion monitoring. Validate system stability under continuous operation beyond short-duration regression tests.

## Problem Analysis

M17 performance checks run for 60 seconds - sufficient for regression detection but insufficient for:
- **Memory leaks**: Small per-operation leaks accumulate over hours
- **Latency drift**: Cache pollution, fragmentation cause gradual slowdown
- **Resource exhaustion**: File descriptors, threads, connections leak slowly
- **Consolidation impact**: M17 consolidation runs every 60s, needs multi-cycle validation
- **GC/allocator pressure**: Arena exhaustion, fragmentation over millions of operations

Production systems run 24/7. A memory leak of 1KB/minute crashes after 17 days. Latency drift of 0.1% per hour doubles P99 in 30 days. Critical to catch these issues before production.

## Architecture

### Soak Test Orchestrator

```rust
// File: tools/loadtest/src/soak/orchestrator.rs

pub struct SoakTestConfig {
    /// Target duration (default: 24 hours)
    pub duration: Duration,

    /// Checkpoint interval for snapshot + analysis (default: 1 hour)
    pub checkpoint_interval: Duration,

    /// Baseline scenario to run continuously
    pub scenario: PathBuf,

    /// Memory leak detection thresholds
    pub memory_leak_threshold_mb_per_hour: f64,

    /// Latency drift thresholds
    pub latency_drift_threshold_pct_per_hour: f64,

    /// Resource monitoring intervals
    pub resource_check_interval: Duration,

    /// Automatic abort conditions
    pub abort_conditions: AbortConditions,
}

pub struct AbortConditions {
    /// Abort if RSS exceeds this (MB)
    pub max_rss_mb: usize,

    /// Abort if P99 exceeds this (ms)
    pub max_p99_ms: f64,

    /// Abort if error rate exceeds this
    pub max_error_rate: f64,

    /// Abort if any checkpoint fails regression
    pub abort_on_regression: bool,
}

pub struct SoakTestOrchestrator {
    config: SoakTestConfig,
    start_time: Instant,
    checkpoints: Vec<Checkpoint>,
    metrics_collector: MetricsCollector,
    resource_monitor: ResourceMonitor,
}

impl SoakTestOrchestrator {
    pub async fn run(&mut self) -> Result<SoakTestReport> {
        let mut checkpoint_timer = interval(self.config.checkpoint_interval);
        let mut resource_timer = interval(self.config.resource_check_interval);

        loop {
            select! {
                _ = checkpoint_timer.tick() => {
                    let checkpoint = self.capture_checkpoint().await?;
                    self.analyze_checkpoint(&checkpoint)?;

                    if self.should_abort(&checkpoint)? {
                        return Err(anyhow!("Abort condition triggered at checkpoint {}", self.checkpoints.len()));
                    }

                    self.checkpoints.push(checkpoint);
                }

                _ = resource_timer.tick() => {
                    let resources = self.resource_monitor.snapshot()?;
                    self.check_resources(&resources)?;
                }

                _ = sleep_until(self.start_time + self.config.duration) => {
                    break; // Duration complete
                }
            }
        }

        self.generate_report()
    }

    fn capture_checkpoint(&self) -> Result<Checkpoint> {
        // Capture metrics without stopping load test
        let metrics = self.metrics_collector.snapshot();
        let resources = self.resource_monitor.snapshot()?;
        let heap_profile = self.capture_heap_profile()?;

        Ok(Checkpoint {
            timestamp: self.start_time.elapsed(),
            metrics,
            resources,
            heap_profile,
        })
    }

    fn analyze_checkpoint(&self, checkpoint: &Checkpoint) -> Result<()> {
        // Detect memory leaks
        if let Some(leak_rate) = self.detect_memory_leak(checkpoint) {
            warn!("Memory leak detected: {:.2} MB/hour", leak_rate);
            if leak_rate > self.config.memory_leak_threshold_mb_per_hour {
                return Err(anyhow!("Memory leak exceeds threshold"));
            }
        }

        // Detect latency drift
        if let Some(drift_rate) = self.detect_latency_drift(checkpoint) {
            warn!("Latency drift detected: {:.2}% per hour", drift_rate);
            if drift_rate > self.config.latency_drift_threshold_pct_per_hour {
                return Err(anyhow!("Latency drift exceeds threshold"));
            }
        }

        Ok(())
    }
}
```

### Memory Leak Detection

```rust
// File: tools/loadtest/src/soak/leak_detector.rs

pub struct LeakDetector {
    /// Baseline RSS at start (MB)
    baseline_rss: f64,

    /// Historical RSS measurements
    rss_history: Vec<(Duration, f64)>,

    /// Minimum samples for regression
    min_samples: usize,
}

impl LeakDetector {
    /// Detect memory leak using linear regression on RSS over time
    pub fn detect_leak(&self) -> Option<f64> {
        if self.rss_history.len() < self.min_samples {
            return None;
        }

        // Linear regression: RSS = baseline + leak_rate * time
        let (slope, r_squared) = self.linear_regression();

        // Only report if strong correlation (R^2 > 0.8)
        if r_squared > 0.8 && slope > 0.0 {
            Some(slope) // MB per hour
        } else {
            None
        }
    }

    fn linear_regression(&self) -> (f64, f64) {
        let n = self.rss_history.len() as f64;
        let sum_x: f64 = self.rss_history.iter().map(|(t, _)| t.as_secs_f64()).sum();
        let sum_y: f64 = self.rss_history.iter().map(|(_, rss)| *rss).sum();
        let sum_xy: f64 = self.rss_history.iter().map(|(t, rss)| t.as_secs_f64() * rss).sum();
        let sum_x2: f64 = self.rss_history.iter().map(|(t, _)| t.as_secs_f64().powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate R^2
        let mean_y = sum_y / n;
        let ss_tot: f64 = self.rss_history.iter().map(|(_, rss)| (rss - mean_y).powi(2)).sum();
        let ss_res: f64 = self.rss_history.iter().map(|(t, rss)| {
            let predicted = intercept + slope * t.as_secs_f64();
            (rss - predicted).powi(2)
        }).sum();
        let r_squared = 1.0 - (ss_res / ss_tot);

        (slope * 3600.0, r_squared) // Convert to MB/hour
    }

    /// Capture heap profile using jemalloc stats
    pub fn capture_heap_profile(&self) -> Result<HeapProfile> {
        #[cfg(feature = "jemalloc")]
        {
            use tikv_jemalloc_ctl::{epoch, stats};

            // Update statistics
            epoch::mib()?.advance()?;

            Ok(HeapProfile {
                allocated: stats::allocated::mib()?.read()?,
                active: stats::active::mib()?.read()?,
                metadata: stats::metadata::mib()?.read()?,
                resident: stats::resident::mib()?.read()?,
                mapped: stats::mapped::mib()?.read()?,
            })
        }

        #[cfg(not(feature = "jemalloc"))]
        {
            Ok(HeapProfile::default())
        }
    }
}

#[derive(Debug, Clone)]
pub struct HeapProfile {
    pub allocated: usize,  // Bytes allocated
    pub active: usize,     // Bytes in active pages
    pub metadata: usize,   // Allocator metadata
    pub resident: usize,   // Resident set size
    pub mapped: usize,     // Total mapped memory
}
```

### Latency Drift Detection

```rust
// File: tools/loadtest/src/soak/drift_detector.rs

pub struct DriftDetector {
    /// Historical latency measurements
    latency_history: Vec<(Duration, LatencySnapshot)>,

    /// Window size for drift calculation (default: 4 hours)
    window: Duration,
}

#[derive(Debug, Clone)]
pub struct LatencySnapshot {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
}

impl DriftDetector {
    /// Detect latency drift using linear regression on windowed data
    pub fn detect_drift(&self) -> Option<f64> {
        if self.latency_history.is_empty() {
            return None;
        }

        // Use windowed regression (last 4 hours) for recent trend
        let now = self.latency_history.last().unwrap().0;
        let window_start = now.saturating_sub(self.window);

        let windowed: Vec<_> = self.latency_history.iter()
            .filter(|(t, _)| *t >= window_start)
            .collect();

        if windowed.len() < 4 {
            return None;
        }

        // Regress P99 latency over time
        let (slope, r_squared) = self.regress_p99(&windowed);

        // Only report if strong correlation and upward trend
        if r_squared > 0.7 && slope > 0.0 {
            let baseline_p99 = windowed.first().unwrap().1.p99;
            Some((slope / baseline_p99) * 100.0) // Percent per hour
        } else {
            None
        }
    }

    fn regress_p99(&self, data: &[&(Duration, LatencySnapshot)]) -> (f64, f64) {
        // Linear regression on P99 latency
        // ... (similar to LeakDetector::linear_regression)
    }
}
```

### Resource Monitoring

```rust
// File: tools/loadtest/src/soak/resource_monitor.rs

pub struct ResourceMonitor {
    pid: u32,
    baseline: ResourceSnapshot,
}

#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    pub timestamp: Instant,
    pub cpu_percent: f64,
    pub rss_mb: f64,
    pub vsize_mb: f64,
    pub threads: usize,
    pub open_fds: usize,
    pub network_connections: usize,
    pub disk_io_mb: f64,
}

impl ResourceMonitor {
    pub fn snapshot(&self) -> Result<ResourceSnapshot> {
        #[cfg(target_os = "linux")]
        {
            self.snapshot_linux()
        }

        #[cfg(target_os = "macos")]
        {
            self.snapshot_macos()
        }
    }

    #[cfg(target_os = "linux")]
    fn snapshot_linux(&self) -> Result<ResourceSnapshot> {
        // Parse /proc/{pid}/stat, /proc/{pid}/status
        let stat = std::fs::read_to_string(format!("/proc/{}/stat", self.pid))?;
        let status = std::fs::read_to_string(format!("/proc/{}/status", self.pid))?;

        // Parse RSS, threads, etc.
        // ... (implementation details)
    }

    #[cfg(target_os = "macos")]
    fn snapshot_macos(&self) -> Result<ResourceSnapshot> {
        // Use ps command for macOS
        let output = Command::new("ps")
            .args(&["-p", &self.pid.to_string(), "-o", "rss,vsz,%cpu,nlwp"])
            .output()?;

        // Parse output
        // ... (implementation details)
    }
}
```

### Soak Test Scenario

```toml
# File: scenarios/soak/24h_production.toml
name = "24-Hour Production Soak Test"
description = "Sustained load with consolidation cycles, leak detection, drift analysis"

[duration]
total_seconds = 86400  # 24 hours

[soak]
checkpoint_interval_sec = 3600  # 1 hour
resource_check_interval_sec = 60  # 1 minute

[soak.abort_conditions]
max_rss_mb = 8192  # 8GB
max_p99_ms = 100.0
max_error_rate = 0.05
abort_on_regression = true

[soak.thresholds]
memory_leak_mb_per_hour = 10.0  # Abort if >10MB/hour leak
latency_drift_pct_per_hour = 2.0  # Abort if >2%/hour drift

[arrival]
pattern = "diurnal"
baseline_ops = 500.0
peak_multiplier = 2.0
peak_hour = 14
trough_hour = 3

[operations]
store_weight = 0.35
recall_weight = 0.35
embedding_search_weight = 0.3

[data]
num_nodes = 100_000
embedding_dim = 768
memory_spaces = 1

[validation]
expected_p99_latency_ms = 10.0
expected_throughput_ops_sec = 400.0
max_error_rate = 0.01
```

## Implementation Plan

### 1. Build Soak Test Orchestrator

**New file:** `tools/loadtest/src/soak/orchestrator.rs`

**Key responsibilities:**
- Run load test continuously for specified duration
- Capture checkpoints at regular intervals
- Abort on threshold violations
- Generate comprehensive report

### 2. Implement Leak Detection

**New file:** `tools/loadtest/src/soak/leak_detector.rs`

**Key algorithms:**
- Linear regression on RSS over time
- R^2 > 0.8 requirement for confidence
- jemalloc heap profiling integration

### 3. Implement Drift Detection

**New file:** `tools/loadtest/src/soak/drift_detector.rs`

**Key algorithms:**
- Windowed regression (last 4 hours)
- Focus on P99 latency drift
- Statistical significance testing

### 4. Build Resource Monitor

**New file:** `tools/loadtest/src/soak/resource_monitor.rs`

**Platform-specific implementations:**
- Linux: Parse /proc/{pid}/* files
- macOS: Use ps/top commands
- Windows: Use Performance Counters API

### 5. Create Soak Test CLI

**Modify:** `tools/loadtest/src/main.rs`

```rust
Commands::Soak {
    /// Soak test scenario
    #[arg(short, long)]
    scenario: PathBuf,

    /// Duration in hours (default: 24)
    #[arg(short, long, default_value = "24")]
    duration_hours: u64,

    /// Checkpoint interval in hours (default: 1)
    #[arg(long, default_value = "1")]
    checkpoint_interval: u64,

    /// Output directory for checkpoints
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Engram endpoint URL
    #[arg(long, default_value = "http://localhost:7432")]
    endpoint: String,
}
```

### 6. Create Analysis Scripts

**New file:** `scripts/analyze_soak_test.sh`

```bash
#!/bin/bash
# Analyze completed soak test results

CHECKPOINT_DIR=$1

# Generate plots
python3 scripts/plot_soak_results.py \
    --input "$CHECKPOINT_DIR" \
    --output "$CHECKPOINT_DIR/analysis"

# Check for leaks
if grep -q "MEMORY_LEAK_DETECTED" "$CHECKPOINT_DIR/report.txt"; then
    echo "[WARN] Memory leak detected - see report.txt"
    exit 1
fi

# Check for drift
if grep -q "LATENCY_DRIFT_DETECTED" "$CHECKPOINT_DIR/report.txt"; then
    echo "[WARN] Latency drift detected - see report.txt"
    exit 1
fi

echo "[OK] Soak test passed all checks"
```

**New file:** `scripts/plot_soak_results.py`

```python
#!/usr/bin/env python3
# Generate plots from soak test checkpoints

import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_memory_over_time(checkpoints, output_dir):
    times = [c['timestamp_sec'] / 3600 for c in checkpoints]
    rss = [c['resources']['rss_mb'] for c in checkpoints]

    plt.figure(figsize=(12, 6))
    plt.plot(times, rss)
    plt.xlabel('Time (hours)')
    plt.ylabel('RSS (MB)')
    plt.title('Memory Usage Over Time')
    plt.grid(True)
    plt.savefig(output_dir / 'memory_over_time.png')

def plot_latency_over_time(checkpoints, output_dir):
    times = [c['timestamp_sec'] / 3600 for c in checkpoints]
    p99 = [c['metrics']['p99_latency_ms'] for c in checkpoints]

    plt.figure(figsize=(12, 6))
    plt.plot(times, p99)
    plt.xlabel('Time (hours)')
    plt.ylabel('P99 Latency (ms)')
    plt.title('P99 Latency Over Time')
    plt.grid(True)
    plt.savefig(output_dir / 'latency_over_time.png')

# ... (additional plots)
```

## File Changes

### New Files
- `tools/loadtest/src/soak/orchestrator.rs` (580 lines)
- `tools/loadtest/src/soak/leak_detector.rs` (320 lines)
- `tools/loadtest/src/soak/drift_detector.rs` (280 lines)
- `tools/loadtest/src/soak/resource_monitor.rs` (450 lines)
- `scenarios/soak/24h_production.toml` (65 lines)
- `scenarios/soak/7d_stress.toml` (70 lines)
- `scenarios/soak/README.md` (180 lines)
- `scripts/analyze_soak_test.sh` (80 lines)
- `scripts/plot_soak_results.py` (250 lines)
- `tools/loadtest/tests/soak_tests.rs` (400 lines)

### Modified Files
- `tools/loadtest/src/main.rs`: Add soak command
- `tools/loadtest/Cargo.toml`: Add jemalloc dependency
- `.gitignore`: Ignore soak test checkpoint directories

**Total new code**: ~2,600 lines
**Total modified code**: ~150 lines

## Testing Strategy

### Unit Tests
- LeakDetector: Verify linear regression with synthetic data
- DriftDetector: Verify drift detection with manufactured trends
- ResourceMonitor: Verify parsing of /proc and ps output

### Integration Tests
- Short soak test (1 hour accelerated to 10min)
- Inject artificial memory leak, verify detection
- Inject artificial latency drift, verify detection

### Validation Tests
- 24-hour real soak test on staging hardware
- Verify no false positives in stable system
- Verify detection of known leak (add temporary leak code)

## Performance Validation

**Overhead measurement:**
- Checkpoint capture: <500ms
- Resource monitoring: <50ms per check
- Total overhead: <0.1% of test duration

**No impact on M17 baseline** (soak tests are separate command).

## Success Criteria

1. **Memory Leak Detection**: Detect 10MB/hour leak with >95% confidence (R^2 > 0.8)
2. **Latency Drift Detection**: Detect 2%/hour drift with >90% confidence
3. **24-Hour Stability**: Complete 24h soak without crashes or aborts on stable build
4. **False Positive Rate**: <5% false alarms on stable build
5. **Automated Analysis**: Generate plots and reports without manual intervention
6. **CI Integration**: Nightly soak tests run automatically

## Risk Mitigation

1. **Long duration**: Start with 1-hour mini-soak, extend to 24h after validation
2. **Platform differences**: Test on both Linux and macOS, graceful degradation if metrics unavailable
3. **False positives**: Conservative thresholds (10MB/hour, 2%/hour) to minimize noise
4. **Resource overhead**: Checkpoint interval ≥1 hour to minimize impact

## Dependencies

- Task 001 (production workload patterns) for realistic load
- jemalloc for heap profiling (optional)
- Python + matplotlib for plotting (optional, graceful degradation)

## Next Steps After Completion

1. Run 7-day stress test on production hardware
2. Establish baseline memory growth rate for normal operation
3. Document acceptable latency drift thresholds
4. Create production monitoring based on soak test insights
