# GPU Performance Tuning Guide

**Audience**: Performance engineers, SREs, systems architects

**Last Updated**: 2025-10-26

## IMPORTANT: Implementation Status

**This performance tuning guide describes optimization for FUTURE GPU implementation (Milestone 13+).**

**CURRENT MILESTONE 12 STATUS**:

- **IMPLEMENTED**: CPU SIMD implementation (production-ready, high performance)

- **NOT IMPLEMENTED**: CUDA kernels, GPU acceleration, GPU-specific tuning parameters

**CURRENT PERFORMANCE TUNING**: Focus on batch sizes, CPU core count, and memory bandwidth optimization for CPU SIMD implementation. GPU-specific tuning will be relevant when CUDA kernels are implemented in Milestone 13+.

**This guide describes future GPU performance optimization.** For current CPU SIMD performance tuning, focus on workload batching and CPU resource allocation.

---

## Overview

This guide will help optimize GPU-accelerated Engram deployments once CUDA kernels are implemented. After following the GPU Deployment Guide and ensuring GPU acceleration is working, use this guide to maximize performance.

For architectural details, see the GPU Architecture Reference. For troubleshooting, see the GPU Troubleshooting Guide.

## Performance Tuning Workflow

```

1. Establish Baseline
   └─> Run benchmarks to measure current performance

2. Profile Workload
   └─> Identify bottlenecks (memory, compute, CPU, etc.)

3. Apply Optimizations
   └─> Tune configuration based on profiling data

4. Measure Impact
   └─> Re-run benchmarks to validate improvements

5. Iterate
   └─> Repeat for each optimization until targets met

```

## Step 1: Establish Baseline

### Run Performance Benchmarks

```bash
# Run comprehensive benchmark suite
cargo bench --bench gpu_performance_validation

# Save baseline results
cargo bench --bench gpu_performance_validation -- --save-baseline baseline_v1

# Key metrics to record:
# - CPU latency (P50, P99)
# - GPU latency (P50, P99)
# - Speedup ratio (GPU vs CPU)
# - Break-even batch size

```

**Example Baseline**:

```
Cosine Similarity (1024 vectors):
  CPU: 2.1 ms (P50), 2.3 ms (P99)
  GPU: 295 us (P50), 320 us (P99)
  Speedup: 7.1x
  Break-even: 64 vectors

Activation Spreading (1000 nodes):
  CPU: 850 us (P50), 920 us (P99)
  GPU: 120 us (P50), 135 us (P99)
  Speedup: 7.1x
  Break-even: 512 nodes

```

### Collect Production Metrics

```bash
# Export current metrics
curl http://localhost:8080/metrics > baseline_metrics.txt

# Key metrics:
grep -E "gpu_(launches|fallbacks|success_rate|speedup)" baseline_metrics.txt

# Workload characteristics:
grep batch_size_histogram baseline_metrics.txt

```

## Step 2: Profile Workload

### Identify Bottlenecks

**GPU Utilization Analysis**:

```bash
# Monitor GPU during production workload
nvidia-smi dmon -s puct -c 100 > gpu_utilization.log

# Analyze:
# sm% (GPU compute utilization):
#   - 90-100%: Compute bound (good)
#   - 50-90%: Partially utilized (can improve)
#   - < 50%: Memory bound or underutilized

# mem% (GPU memory utilization):
#   - High (>80%): Memory bound
#   - Low (<50%): Compute bound

# pwr (power draw):
#   - Near power limit: GPU working hard
#   - Well below limit: Not fully utilized

```

**Latency Breakdown**:

```bash
# Profile with NVIDIA Nsight Systems
nsys profile -o timeline.qdrep ./target/release/engram start

# Generate report
nsys stats --report cuda_gpu_kern_sum timeline.qdrep

# Look for:
# - Kernel execution time (actual compute)
# - Memory transfer time (CPU <-> GPU)
# - Launch overhead (gap between kernels)
# - Idle time (GPU waiting for work)

```

**Example Profiling Output**:

```
Total Time: 1000 ms
├─ Kernel Execution: 400 ms (40%)  <- Time spent computing
├─ Memory Transfer: 100 ms (10%)   <- Time copying data
├─ Launch Overhead: 50 ms (5%)     <- Time launching kernels
└─ Idle: 450 ms (45%)              <- GPU waiting for work

Bottleneck: GPU is idle 45% of time
→ Need to increase batch sizes or launch frequency

```

### Workload Classification

Determine your workload type:

**High-Throughput Workload**:

- Many concurrent queries (>100 QPS)

- Batch sizes naturally large (>256 vectors)

- Latency tolerance: P99 < 100ms

**Low-Latency Workload**:

- Few concurrent queries (<10 QPS)

- Small batch sizes (<64 vectors)

- Latency requirement: P99 < 10ms

**Mixed Workload**:

- Variable query rate

- Variable batch sizes

- Need both throughput and latency

Tuning strategy differs for each workload type.

## Step 3: Configuration Tuning (PLANNED for Milestone 13+)

**Note**: The configuration examples below use TOML format for readability, but actual configuration in Milestone 13+ will use the Rust `HybridConfig` API. TOML file loading is not yet implemented.

### Consumer GPU Tuning (RTX 3060, GTX 1660 Ti) - PLANNED

**High-Throughput Workload** (Planned configuration):

```toml
# PLANNED - Not yet functional in Milestone 12
# Future configuration matching HybridConfig struct

[gpu]
gpu_min_batch_size = 64          # Use GPU for 64+ vectors
force_cpu_mode = false

[gpu.thresholds]
gpu_speedup_threshold = 1.3      # Lower threshold for consumer GPUs
gpu_success_rate_threshold = 0.95

[gpu.telemetry]
telemetry_enabled = true
performance_window_size = 200    # Larger window for stable averages

# Note: vram_safety_margin not in HybridConfig yet

```

**Low-Latency Workload** (Planned configuration):

```toml
# PLANNED - Not yet functional
[gpu]
gpu_min_batch_size = 128         # Higher threshold (avoid launch overhead)
force_cpu_mode = false

[gpu.thresholds]
gpu_speedup_threshold = 2.0      # Only use GPU if significantly faster
gpu_success_rate_threshold = 0.98

[gpu.telemetry]
telemetry_enabled = true
performance_window_size = 100

```

**Mixed Workload** (Planned configuration):

```toml
# PLANNED - Not yet functional
[gpu]
gpu_min_batch_size = 96          # Middle ground
force_cpu_mode = false

[gpu.thresholds]
gpu_speedup_threshold = 1.5
gpu_success_rate_threshold = 0.95

[gpu.telemetry]
telemetry_enabled = true
performance_window_size = 150

```

### Datacenter GPU Tuning (A100, H100, V100) - PLANNED

**Note**: All datacenter GPU configurations below are planned for Milestone 13+. Parameter names match HybridConfig struct (see `engram-core/src/compute/cuda/hybrid.rs`).

**High-Throughput Workload** (Planned configuration):

```toml
# PLANNED - Not yet functional
[gpu]
gpu_min_batch_size = 32          # Lower threshold (A100 has lower overhead)
force_cpu_mode = false
vram_safety_margin = 0.7     # More aggressive (40GB VRAM)

[gpu.thresholds]
speedup_threshold = 1.2      # Lower threshold (A100 is much faster)
success_rate_threshold = 0.95

[gpu.telemetry]
enabled = true
performance_window = 500     # Higher query volume

```

**Low-Latency Workload**:

```toml
[gpu]
enabled = true
min_batch_size = 64          # Still benefit from GPU at smaller sizes
force_cpu_mode = false
vram_safety_margin = 0.8

[gpu.thresholds]
speedup_threshold = 1.5
success_rate_threshold = 0.98

[gpu.telemetry]
enabled = true
performance_window = 200

```

### Parameter Tuning Guide

#### min_batch_size

**Purpose**: Minimum batch size to use GPU (smaller batches use CPU)

**Tuning Strategy**:

```bash
# Run benchmark with different batch sizes
for size in 16 32 64 128 256 512; do
  cargo bench --bench gpu_performance_validation -- cosine_similarity_cpu_vs_gpu/$size
done

# Find crossover point where GPU becomes faster
# Set min_batch_size slightly below crossover

# Example results:
# 16: CPU faster (10us vs 30us) <- Launch overhead dominates
# 32: CPU faster (20us vs 35us)
# 64: GPU faster (40us vs 25us) <- Crossover
# 128: GPU much faster (80us vs 30us)

# Set min_batch_size = 64 (or 48 to be conservative)

```

**Recommendations**:

| GPU Type | High-Throughput | Low-Latency | Mixed |
|----------|-----------------|-------------|-------|
| Consumer (RTX 3060) | 64 | 128 | 96 |
| Datacenter (A100) | 32 | 64 | 48 |

#### vram_safety_margin

**Purpose**: Reserve fraction of VRAM to prevent OOM

**Tuning Strategy**:

```bash
# Monitor VRAM usage under load
nvidia-smi dmon -s m -c 100

# Note peak memory usage
# If peak is < 60% of total VRAM:
#   Can reduce safety margin to 0.7 (use more VRAM)
# If peak is > 90% of total VRAM:
#   Increase safety margin to 0.9 (prevent OOM)
# If OOM events occur:
#   Increase safety margin or reduce max batch size

```

**Recommendations**:

| VRAM | Safety Margin | Usable VRAM |
|------|---------------|-------------|
| 4GB  | 0.85 (conservative) | 3.4GB |
| 8GB  | 0.80 (balanced) | 6.4GB |
| 12GB+ | 0.75 (aggressive) | 9GB+ |

#### speedup_threshold

**Purpose**: Minimum speedup to prefer GPU over CPU

**Tuning Strategy**:

```bash
# Check actual speedup from metrics
curl http://localhost:8080/metrics | grep gpu_speedup_ratio

# If speedup_ratio is consistently > 5x:
#   Can lower threshold to 1.2 (use GPU more often)
# If speedup_ratio is < 2x:
#   Increase threshold to 2.0 (be more selective)

```

**Recommendations**:

| GPU Type | Speedup Threshold | Rationale |
|----------|-------------------|-----------|
| Consumer | 1.3-1.5 | Modest speedup, balance overhead |
| Datacenter | 1.2-1.3 | High speedup, aggressive GPU use |

#### success_rate_threshold

**Purpose**: Disable GPU if failure rate too high

**Tuning Strategy**:

```bash
# Check GPU success rate
curl http://localhost:8080/metrics | grep gpu_success_rate

# If success_rate > 0.99:
#   Can keep threshold at 0.95 (normal)
# If success_rate 0.95-0.98:
#   Investigate failures (OOM? CUDA errors?)
# If success_rate < 0.95:
#   GPU is unreliable, may auto-disable

```

**Recommendations**:

- Production: 0.95 (tolerate 5% failure, fallback to CPU)

- Strict SLA: 0.98 (tolerate only 2% failure)

- Development: 0.90 (more tolerant for debugging)

## Step 4: Batch Size Optimization

### Application-Level Batching

**Problem**: Small queries don't benefit from GPU

**Solution**: Batch multiple queries together

**Example** (gRPC streaming):

```rust
// Before: Process queries one-by-one
for query in queries {
    let results = spread_activation(&graph, &query);
    send_response(results);
}

// After: Batch queries
let batch_size = 64;
for chunk in queries.chunks(batch_size) {
    let batch_results = spread_activation_batch(&graph, chunk);
    for result in batch_results {
        send_response(result);
    }
}

```

**Tradeoffs**:

- Increases latency for individual queries (waiting for batch)

- Increases throughput overall (more efficient GPU use)

- Best for high-QPS services where latency tolerance allows

### Dynamic Batching

**Strategy**: Wait up to N milliseconds to accumulate batch, then process

```rust
// Pseudo-code for dynamic batching
let mut pending_queries = Vec::new();
let max_wait_ms = 10; // Maximum latency penalty
let target_batch_size = 128;

loop {
    // Collect queries for up to max_wait_ms
    let timeout = Instant::now() + Duration::from_millis(max_wait_ms);

    while Instant::now() < timeout && pending_queries.len() < target_batch_size {
        if let Some(query) = try_receive_query() {
            pending_queries.push(query);
        }
    }

    // Process batch
    if !pending_queries.is_empty() {
        process_batch(&pending_queries);
        pending_queries.clear();
    }
}

```

**Tuning Parameters**:

| Workload | Max Wait (ms) | Target Batch Size |
|----------|---------------|-------------------|
| Low-latency | 5 | 64 |
| Balanced | 10 | 128 |
| High-throughput | 20 | 256 |

## Step 5: GPU Hardware Optimization

### Power Limit Tuning

Increase GPU power limit for higher performance (if thermal headroom available):

```bash
# Check current power limit
nvidia-smi --query-gpu=power.limit,power.max_limit --format=csv

# Output:
# power.limit [W], power.max_limit [W]
# 170.00, 215.00
#        ^^^^ Can increase to this

# Increase power limit
sudo nvidia-smi -pl 200  # Set to 200W

# Monitor temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1

# If temperature > 85C, reduce power limit
# If temperature < 75C and not hitting power limit, can increase further

```

**Recommendations**:

| GPU | Stock Limit | Recommended | Max Safe |
|-----|-------------|-------------|----------|
| RTX 3060 | 170W | 190W | 200W |
| RTX 4070 | 200W | 220W | 240W |
| A100 (PCIe) | 250W | 280W | 300W |
| A100 (SXM) | 400W | 400W | 400W |

### GPU Clocks

Lock GPU clocks to prevent throttling and variance:

```bash
# Enable persistence mode (keeps driver loaded)
sudo nvidia-smi -pm 1

# Lock clocks to maximum (reduces latency variance)
# Query max clocks
nvidia-smi --query-gpu=clocks.max.sm,clocks.max.mem --format=csv

# Output:
# clocks.max.sm [MHz], clocks.max.mem [MHz]
# 1777, 7001

# Lock to max clocks
sudo nvidia-smi -lgc 1777  # Lock GPU clock
sudo nvidia-smi -lmc 7001  # Lock memory clock

# This increases power consumption but reduces P99 latency
# Best for production workloads with strict SLAs

```

**When to Lock Clocks**:

- Production with strict P99 latency SLAs

- Benchmarking (reduces variance)

**When NOT to Lock Clocks**:

- Development (wastes power)

- Multi-tenant systems (unfair to other users)

- Limited cooling (may overheat)

### Cooling Optimization

Better cooling allows higher sustained performance:

```bash
# Increase fan speed manually (for testing)
sudo nvidia-smi -i 0 --fan-speed=80  # 80% fan speed

# Monitor temperature improvement
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1

# If temperature drops significantly:
# Consider improving case airflow or adding GPU cooler

```

**Cooling Improvements**:

1. Clean dust from GPU fans and heatsink

2. Improve case airflow (add intake/exhaust fans)

3. Upgrade GPU cooler (for consumer cards)

4. Use server rack with proper ventilation (datacenter)

## Step 6: Multi-Tenant Optimization

### Multiple Engram Instances on One GPU

Use MPS (Multi-Process Service) to share GPU efficiently:

```bash
# Enable NVIDIA MPS
sudo nvidia-smi -c EXCLUSIVE_PROCESS  # Set compute mode
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d

# Now multiple Engram processes can share GPU efficiently

# Start multiple Engram instances
./target/release/engram start --port 8080 &
./target/release/engram start --port 8081 &
./target/release/engram start --port 8082 &

# Each instance shares GPU with lower overhead

```

**Benefits**:

- Lower overhead than time-slicing

- Better GPU utilization

- Fair resource sharing

**Drawbacks**:

- Requires exclusive compute mode

- More complex monitoring

- Failure in one process can affect others

### GPU Partitioning (A100 MIG)

NVIDIA A100/H100 support Multi-Instance GPU (MIG) for isolation:

```bash
# Enable MIG mode
sudo nvidia-smi -mig 1

# Create GPU instances (e.g., 3x 1g.10gb instances)
sudo nvidia-smi mig -cgi 1g.10gb -C

# List instances
nvidia-smi -L

# Output:
# GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-xxx)
#   MIG 1g.10gb Device 0: (UUID: MIG-xxx)
#   MIG 1g.10gb Device 1: (UUID: MIG-xxx)
#   MIG 1g.10gb Device 2: (UUID: MIG-xxx)

# Run Engram on specific MIG instance
export CUDA_VISIBLE_DEVICES=MIG-xxx
./target/release/engram start

```

**Use Cases**:

- Multi-tenant SaaS

- Isolated testing environments

- Guaranteed QoS per tenant

## Step 7: Monitoring and Continuous Optimization

### Set Up Performance Dashboards

**Grafana Dashboard Panels**:

1. **GPU Utilization**:

   ```promql
   # GPU compute utilization
   nvidia_smi_utilization_gpu_ratio{gpu="0"}

   # GPU memory utilization
   nvidia_smi_utilization_memory_ratio{gpu="0"}
   ```

2. **Engram GPU Metrics** (for future implementation):

   ```promql
   # GPU launch rate
   rate(engram_gpu_launch_total[5m])

   # GPU success rate (derived)
   (rate(engram_gpu_launch_total[5m]) - rate(engram_gpu_fallback_total[5m])) / rate(engram_gpu_launch_total[5m])

   # GPU speedup (will be derived from latency metrics when implemented)
   # TBD based on actual implementation
   ```

3. **Latency Percentiles**:

   ```promql
   # P50 latency
   histogram_quantile(0.50, rate(engram_spreading_latency_bucket[5m]))

   # P99 latency
   histogram_quantile(0.99, rate(engram_spreading_latency_bucket[5m]))
   ```

4. **Batch Size Distribution**:

   ```promql
   # Histogram of batch sizes
   rate(engram_batch_size_bucket[5m])
   ```

### Performance Regression Alerts

```yaml
# Alert if GPU speedup degrades (when GPU metrics implemented)

- alert: GPUSpeedupRegression
  # TBD: Will use derived speedup metric from latency data
  expr: engram_gpu_speedup_derived < 3.0
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "GPU speedup below target (3x)"
    description: "Current speedup: {{ $value }}, investigate performance"

# Alert if P99 latency increases

- alert: HighP99Latency
  expr: histogram_quantile(0.99, rate(engram_spreading_latency_bucket[5m])) > 100
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "P99 latency above 100ms"
    description: "P99: {{ $value }}ms, check GPU utilization and batch sizes"

# Alert if GPU utilization too low (for future GPU implementation)

- alert: LowGPUUtilization
  expr: nvidia_smi_utilization_gpu_ratio < 0.3 and rate(engram_gpu_launch_total[5m]) > 0
  for: 15m
  labels:
    severity: info
  annotations:
    summary: "GPU underutilized"
    description: "GPU utilization: {{ $value }}, consider increasing batch sizes"

```

### A/B Testing Configuration Changes

Test configuration changes safely:

```bash
# 1. Establish current baseline
cargo bench --bench gpu_performance_validation -- --save-baseline current

# 2. Make configuration change
# Edit /etc/engram/gpu.toml
# Change min_batch_size from 64 to 96

# 3. Restart and measure
systemctl restart engram
sleep 60  # Let metrics stabilize

# 4. Run benchmark against new config
cargo bench --bench gpu_performance_validation -- --save-baseline new_config

# 5. Compare baselines
cargo bench --bench gpu_performance_validation -- --baseline current --baseline new_config

# 6. Check production metrics
curl http://localhost:8080/metrics > new_metrics.txt
diff baseline_metrics.txt new_metrics.txt

# 7. Decide: keep change or revert
# If improvement > 10%: Keep
# If improvement < 5%: Revert
# If regression: Immediately revert

```

## Performance Optimization Checklist

Before considering performance "tuned", verify:

- [ ] Baseline benchmarks recorded

- [ ] Workload profiled (CPU/GPU/memory bottleneck identified)

- [ ] Configuration tuned for GPU type and workload

- [ ] Batch sizes optimized (>= min_batch_size for most queries)

- [ ] GPU utilization > 70% under load (if expecting GPU use)

- [ ] P99 latency meets SLA requirements

- [ ] GPU speedup > 3x vs CPU baseline

- [ ] OOM events < 1 per hour (target: 0)

- [ ] GPU success rate > 95%

- [ ] Power/thermal limits not throttling GPU

- [ ] PCIe bandwidth sufficient (Gen 3 x16 or better)

- [ ] Monitoring dashboards created

- [ ] Performance regression alerts configured

- [ ] A/B testing process established

## Common Performance Pitfalls

### Pitfall 1: Batch Sizes Too Small

**Symptom**: GPU available but not being used

**Diagnosis**:

```bash
curl http://localhost:8080/metrics | grep batch_size_histogram
# Shows most batches < min_batch_size

```

**Solution**:

- Lower `min_batch_size` in configuration

- Implement application-level batching

- Use dynamic batching with short timeout

### Pitfall 2: Too Aggressive min_batch_size

**Symptom**: High P99 latency, many CPU operations

**Diagnosis**:

```bash
# High proportion of operations use CPU despite GPU available
curl http://localhost:8080/metrics | grep gpu_launches_total
# Value doesn't increase much under load

```

**Solution**:

- Reduce `min_batch_size`

- Profile actual batch size distribution

- Set threshold at P25 of distribution

### Pitfall 3: GPU Throttling

**Symptom**: Performance degrades over time

**Diagnosis**:

```bash
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.max.sm --format=csv -l 1

# temperature > 80C and clocks.sm < clocks.max.sm
# GPU is thermal throttling

```

**Solution**:

- Improve cooling

- Reduce power limit

- Check case airflow

### Pitfall 4: CPU Bottleneck

**Symptom**: Low GPU utilization despite correct config

**Diagnosis**:

```bash
# GPU at 30%, CPU at 100%
top  # Shows high CPU usage
nvidia-smi dmon  # Shows low GPU usage

```

**Solution**:

- Increase CPU cores

- Optimize data preparation (reduce serialization overhead)

- Use more worker threads

- Profile CPU hotspots with `perf`

### Pitfall 5: Memory Transfer Overhead

**Symptom**: GPU utilization is bursty (spikes to 100%, then 0%)

**Diagnosis**:

```bash
# Nsight Systems shows alternating pattern:
# GPU compute, idle, memory transfer, idle, GPU compute
nsys profile -o timeline.qdrep ./target/release/engram start

```

**Solution**:

- Use CUDA Unified Memory (already default in Engram)

- Increase batch sizes to amortize transfer overhead

- Consider pinned host memory (future optimization)

## Advanced Optimizations

### Kernel Launch Overhead Reduction

**Future Feature**: Persistent kernels

Current overhead: 10-20us per kernel launch
With persistent kernels: <1us

This reduces break-even batch size from 64 to ~16 vectors.

### Mixed Precision (FP16)

**Future Feature**: Use FP16 for compute, FP32 for storage

Expected speedup on Ampere+: 1.5-2x

Requires CUDA compute capability 8.0+.

### Kernel Fusion

**Future Feature**: Fuse multiple operations into single kernel

Example: Cosine similarity + Top-K selection in one kernel
Reduces memory bandwidth by ~40%.

## Performance Tuning by GPU Model

### NVIDIA RTX 3060 (12GB)

```toml
[gpu]
min_batch_size = 64
vram_safety_margin = 0.75

[gpu.thresholds]
speedup_threshold = 1.4

```

**Expected Performance**:

- Cosine similarity: 7x speedup at 1K vectors

- Activation spreading: 7x speedup at 1K nodes

- Max throughput: ~400K vectors/sec

**Tuning Focus**:

- Batch size optimization critical

- Memory bandwidth limited (360 GB/s)

- Good for development and small production

### NVIDIA A100 (40GB)

```toml
[gpu]
min_batch_size = 32
vram_safety_margin = 0.70

[gpu.thresholds]
speedup_threshold = 1.2

```

**Expected Performance**:

- Cosine similarity: 26x speedup at 10K vectors

- Activation spreading: 19x speedup at 10K nodes

- Max throughput: ~10M vectors/sec

**Tuning Focus**:

- Use aggressive batching (>1K vectors)

- Leverage high memory bandwidth (1555 GB/s)

- Consider MIG for multi-tenancy

- Ideal for production at scale

### NVIDIA T4 (16GB)

```toml
[gpu]
min_batch_size = 80
vram_safety_margin = 0.80

[gpu.thresholds]
speedup_threshold = 1.5

```

**Expected Performance**:

- Cosine similarity: 5x speedup at 1K vectors

- Activation spreading: 5x speedup at 1K nodes

- Max throughput: ~300K vectors/sec

**Tuning Focus**:

- Good for inference workloads

- Lower power (70W) suitable for edge/cloud

- Higher launch overhead than A100

## References

- **Performance Report**: `/roadmap/milestone-12/performance_report.md`

- **Optimization Roadmap**: `/roadmap/milestone-12/optimization_roadmap.md`

- **GPU Architecture**: See GPU Architecture Reference

- **Deployment**: See GPU Deployment Guide

- **Troubleshooting**: See GPU Troubleshooting Guide

## Appendix: Performance Tuning Tools

### NVIDIA Nsight Systems

System-wide profiling for timeline analysis:

```bash
# Profile entire workflow
nsys profile -o timeline.qdrep ./target/release/engram start

# Focus on GPU/CUDA activity
nsys profile --trace=cuda,nvtx -o timeline.qdrep ./target/release/engram start

# Analyze interactively
nsys-ui timeline.qdrep

# Or generate text report
nsys stats --report cuda_gpu_kern_sum timeline.qdrep

```

**What to Look For**:

- Kernel execution time vs gaps

- Memory transfer overhead

- CPU/GPU utilization overlap

### NVIDIA Nsight Compute

Kernel-level profiling for optimization:

```bash
# Profile specific kernel with all metrics
ncu --set full -o kernel.ncu-rep ./target/release/engram start

# Focus on memory metrics
ncu --set memory -o mem.ncu-rep ./target/release/engram start

# Focus on compute metrics
ncu --set compute -o compute.ncu-rep ./target/release/engram start

# Analyze
ncu-ui kernel.ncu-rep

```

**What to Look For**:

- SM occupancy (target: >50%)

- Memory bandwidth utilization (target: >60%)

- Warp efficiency (target: >80%)

### Flamegraph

CPU-side profiling:

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Profile Engram
cargo flamegraph --bench gpu_performance_validation

# Open flamegraph.svg in browser

```

**What to Look For**:

- Hot paths in Rust code

- Serialization overhead

- Lock contention

### Custom Metrics

Add custom timing in Rust code:

```rust
use std::time::Instant;

let start = Instant::now();
let result = gpu_operation(&batch);
let elapsed = start.elapsed();

metrics.record_gpu_latency(elapsed.as_micros());

```

## Tuning Examples by Scenario

### Scenario 1: Real-Time Recommendation System

**Requirements**:

- P99 latency < 20ms

- 1000 QPS peak

- Batch sizes: 10-100 queries

**Tuning**:

```toml
[gpu]
min_batch_size = 96  # Above typical batch size
vram_safety_margin = 0.8

[gpu.thresholds]
speedup_threshold = 2.0  # Strict (only use GPU if much faster)
success_rate_threshold = 0.98

# Use dynamic batching with 5ms timeout
# This creates batches of ~50 queries at 1000 QPS

```

**Expected Result**:

- 60% of queries use GPU (batches >= 96)

- P99 latency: 12-15ms

- GPU utilization: 40-60%

### Scenario 2: Batch ETL Pipeline

**Requirements**:

- Throughput > 1M vectors/hour

- Latency not critical

- Large batches (1000+ vectors)

**Tuning**:

```toml
[gpu]
min_batch_size = 64  # Everything uses GPU
vram_safety_margin = 0.7  # Aggressive

[gpu.thresholds]
speedup_threshold = 1.2  # Always prefer GPU
success_rate_threshold = 0.95

# Increase batch size to 4096 in ETL code

```

**Expected Result**:

- 100% of operations use GPU

- Throughput: 3-5M vectors/hour (A100)

- GPU utilization: 90-100%

### Scenario 3: Multi-Tenant SaaS

**Requirements**:

- Isolation between tenants

- Fair resource sharing

- Variable workloads

**Tuning**:

```bash
# Use A100 with MIG (3 instances)
sudo nvidia-smi -mig 1
sudo nvidia-smi mig -cgi 1g.10gb,1g.10gb,1g.10gb -C

# Each Engram instance on separate MIG

```

```toml
[gpu]
min_batch_size = 64
vram_safety_margin = 0.85  # Conservative (shared resource)

[gpu.thresholds]
speedup_threshold = 1.5
success_rate_threshold = 0.95

```

**Expected Result**:

- Each tenant gets guaranteed GPU slice

- No interference between tenants

- Total throughput: 80% of full GPU (MIG overhead)
