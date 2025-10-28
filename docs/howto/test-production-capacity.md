# How to Test Production Capacity

Step-by-step guide to validate that your Engram deployment can handle expected production load.

## Context

You need to validate that your Engram deployment can handle expected production load of 15K operations/second with P99 latency under 10ms before going live.

## Prerequisites

- Engram deployed in production-like environment
- Load testing tool built: `cd tools/loadtest && cargo build --release`
- Access to production traffic patterns (optional but recommended)
- Monitoring infrastructure in place

## Steps

### Step 1: Establish Baseline

Run a short test to verify basic functionality:

```bash
# 5-minute smoke test at 5K ops/sec
./tools/loadtest/target/release/loadtest run \
  --scenario scenarios/mixed_balanced.toml \
  --target-rate 5000 \
  --duration 300 \
  --output results/baseline_smoke.json
```

**Verification**:
- [ ] Test completes without errors
- [ ] P99 latency < 10ms
- [ ] No connection errors
- [ ] Engram logs show no warnings

If this fails, fix basic issues before proceeding to capacity testing.

### Step 2: Capture Production Traffic Pattern (Optional)

If you have existing production traffic, capture a trace for realistic replay:

```bash
# Export 1 hour of production traffic
engram export-trace \
  --start "2025-10-27T00:00:00Z" \
  --duration 3600 \
  --output traces/prod_trace_2025-10-27.json
```

Skip this step if testing a new deployment.

### Step 3: Test at Target Load

Run sustained test at expected production rate:

```bash
# 1-hour test at 15K ops/sec
./tools/loadtest/target/release/loadtest run \
  --scenario scenarios/mixed_balanced.toml \
  --target-rate 15000 \
  --duration 3600 \
  --output results/target_load.json
```

**Monitor during test**:
- Watch real-time metrics in terminal
- Check system resources: `htop` or `top`
- Monitor Engram metrics dashboard (Grafana)
- Watch disk I/O: `iostat -x 1`

**Expected behavior**:
- Throughput stays at 15K ±5%
- P99 latency remains < 10ms
- Error rate < 0.1%
- CPU utilization < 80%
- Memory growth bounded (no leaks)

### Step 4: Replay Production Trace (If Available)

Test with actual production patterns:

```bash
# Replay at 100% production rate
./tools/loadtest/target/release/loadtest replay \
  --trace traces/prod_trace_2025-10-27.json \
  --rate-multiplier 1.0 \
  --output results/prod_replay_1x.json
```

### Step 5: Find Breaking Point

Incrementally increase load to find maximum capacity:

```bash
#!/bin/bash
# Script: find_capacity_limit.sh

for rate in 15000 20000 25000 30000 35000; do
  echo "Testing rate: ${rate} ops/sec"

  ./tools/loadtest/target/release/loadtest run \
    --scenario scenarios/mixed_balanced.toml \
    --target-rate $rate \
    --duration 600 \
    --output results/capacity_${rate}.json

  # Check if latency degraded
  p99=$(jq '.latency.overall.p99' results/capacity_${rate}.json)

  if (( $(echo "$p99 > 10.0" | bc -l) )); then
    echo "Breaking point found at ${rate} ops/sec (P99=${p99}ms)"
    break
  fi

  echo "P99: ${p99}ms - OK, continuing..."
  sleep 30  # Cool down between tests
done
```

Run:
```bash
chmod +x find_capacity_limit.sh
./find_capacity_limit.sh
```

### Step 6: Analyze Latency vs Throughput Curve

Generate capacity analysis chart:

```bash
python3 scripts/plot_capacity.py results/capacity_*.json \
  --output capacity_curve.png
```

Expected curve:
- Flat latency until saturation point
- Sharp increase in latency beyond capacity
- Identify "knee" of curve as safe operating point

### Step 7: Test with Burst Traffic

Validate handling of traffic spikes:

```bash
./tools/loadtest/target/release/loadtest run \
  --scenario scenarios/burst_traffic.toml \
  --duration 1800 \
  --output results/burst_test.json
```

**Scenario**: 2K ops/sec baseline, 30K ops/sec bursts every 5 minutes

**Verification**:
- [ ] System handles burst without errors
- [ ] Latency returns to normal after burst
- [ ] No cascading failures
- [ ] Queue depths remain bounded

### Step 8: Long-Duration Stability Test

Run extended test to check for memory leaks and degradation:

```bash
# 8-hour overnight test
nohup ./tools/loadtest/target/release/loadtest run \
  --scenario scenarios/mixed_balanced.toml \
  --target-rate 15000 \
  --duration 28800 \
  --output results/stability_8h.json \
  > stability_test.log 2>&1 &
```

**Monitor**:
- Check progress: `tail -f stability_test.log`
- Memory growth: `watch -n 60 "ps aux | grep engram | awk '{print \$6}'"`

**Next morning verification**:
- [ ] Test completed successfully
- [ ] Memory usage stable (no unbounded growth)
- [ ] P99 latency consistent throughout duration
- [ ] No error rate increase over time

## Verification Checklist

After completing all tests, verify:

**At 15K ops/sec target load**:
- [ ] P99 latency < 10ms in all 60-second windows
- [ ] Error rate < 0.1% (< 15 errors per 1000 operations)
- [ ] CPU utilization < 80% (20% headroom for spikes)
- [ ] Memory growth bounded (no leaks detected)
- [ ] No disk I/O bottlenecks (queue depth < 32)

**Burst traffic handling**:
- [ ] 30K ops/sec bursts handled without errors
- [ ] Latency recovers to baseline after burst
- [ ] No queue overflow or dropped requests

**Long-duration stability**:
- [ ] 8+ hour test completes successfully
- [ ] No memory leaks (RSS stable)
- [ ] No performance degradation over time
- [ ] Consolidation runs without impacting foreground latency

**System resources**:
- [ ] CPU: < 80% utilization at target load
- [ ] Memory: < 90% utilization, no swap
- [ ] Disk: IOPS within limits, < 1ms latency
- [ ] Network: < 50% bandwidth utilization

## Troubleshooting

### High Latency at Target Load

**Symptom**: P99 > 10ms at 15K ops/sec

**Root cause analysis**:

1. **Check CPU saturation**:
```bash
# CPU per core
mpstat -P ALL 1 10

# If any core is at 100%, add more cores or optimize hot paths
```

2. **Check for lock contention**:
```bash
# Profile with perf
perf record -g ./target/release/engram-server
perf report
```

3. **Check spreading activation depth**:
```bash
# Review spreading config
cat config/engram.toml | grep -A5 spreading

# Reduce max_iterations if > 5
```

**Solutions**:
- Add CPU cores (scale vertically)
- Optimize hot code paths (use flamegraph)
- Reduce spreading activation depth
- Enable SIMD optimizations
- Add more Engram instances (scale horizontally)

### Low Throughput

**Symptom**: Actual throughput < 80% of target

**Root cause analysis**:

1. **Check error rate**:
```bash
jq '.summary.error_rate' results/target_load.json
# If > 1%, system is overloaded
```

2. **Check network**:
```bash
# Network bandwidth
iftop

# Connection pool exhaustion
ss -s | grep tcp
```

3. **Check consolidation impact**:
```bash
# Disable consolidation temporarily
engram-cli config set consolidation.enabled=false
# Re-run test
```

**Solutions**:
- Increase connection pool size
- Add load balancer for connection distribution
- Schedule consolidation during low-traffic windows
- Optimize batch operations

### Memory Leaks

**Symptom**: Memory grows unbounded during 8-hour test

**Root cause analysis**:

1. **Track memory growth rate**:
```bash
# Sample every minute
while true; do
  ps aux | grep engram | awk '{print \$6}' >> memory_usage.txt
  sleep 60
done
```

2. **Enable heap profiling**:
```bash
cargo build --release --features profiling
HEAP_PROFILE=/tmp/engram_heap ./target/release/engram-server
```

3. **Review consolidation**:
```bash
# Check if consolidation is evicting old memories
engram-cli metrics | grep consolidation_evicted_count
```

**Solutions**:
- Update to latest version with leak fixes
- Enable aggressive consolidation
- Increase memory tier migration frequency
- Report issue to Engram team with heap profile

## Success Criteria

Your deployment is ready for production if:

1. **Performance**: Handles target load (15K ops/sec) with P99 < 10ms
2. **Capacity headroom**: Can sustain 2x target load (30K ops/sec) for burst handling
3. **Stability**: 8+ hour test shows no degradation
4. **Resource efficiency**: CPU < 80%, Memory < 90% at target load
5. **Resilience**: Recovers from burst traffic without cascading failures

## Next Steps

After successful capacity validation:

1. **Document results**: Save all test outputs to `results/production_validation/`
2. **Update runbook**: Record capacity limits in operations documentation
3. **Set up monitoring**: Configure alerts for latency and throughput
4. **Schedule re-testing**: Quarterly capacity re-validation
5. **Plan scaling**: Document when to add capacity (e.g., at 70% of max throughput)

## Related Documentation

- [Load Testing Guide](../operations/load-testing.md) - Detailed load testing methodology
- [Performance Tuning](../operations/performance-tuning.md) - Optimization techniques
- [Monitoring](../operations/monitoring.md) - Production observability setup
- [Alerting](../operations/alerting.md) - Alert configuration
