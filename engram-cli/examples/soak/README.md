# Soak Tests

Long-running stress tests for validating system stability, resource management, and performance under sustained load.

## Purpose

Soak tests validate:
- Memory leak detection (hours/days of operation)
- Resource pool behavior under continuous allocation/deallocation
- Metrics accuracy over long time periods
- System stability under sustained load
- Performance degradation over time

## Running Soak Tests

```bash
# Run memory pool soak test (600 iterations, ~6 seconds)
cargo run --example memory_pool_soak

# With custom output directory
cargo run --example memory_pool_soak -- /tmp/soak-results

# Long-running validation (custom iteration count via code modification)
# Edit the test to set ITERATIONS = 86400 for 24-hour run
```

## Available Soak Tests

### `memory_pool_soak.rs`
Validates activation record pool behavior over extended periods.

**What it tests:**
- Pool allocation/deallocation patterns
- Hit rate stability
- Utilization metrics accuracy
- Memory leak detection
- Streaming metrics snapshot consistency

**Outputs:**
- JSON snapshots at start/mid/end stages
- Streaming metrics export
- Console logs for monitoring

**Expected behavior:**
- Hit rate should stabilize > 95%
- Utilization should remain 40-60%
- No memory growth over time
- Consistent metrics accuracy

**Run time:** ~6 seconds (600 iterations) to 24 hours (configurable)

### `sse_streaming_soak.rs` (Planned - Priority 3 from SSE recovery plan)
Validates SSE event streaming under sustained load.

**What it tests:**
- Multiple concurrent SSE clients
- Event delivery under high throughput (100+ ops/sec)
- Lag detection and handling
- Connection stability over hours
- Subscriber lifecycle management

**Expected behavior:**
- Zero events dropped (monitoring metric)
- All clients receive all events
- Lag warnings only under extreme load
- Server remains stable after client disconnects

### `concurrent_operations_soak.rs` (Planned)
Multi-client stress test for concurrent operations.

**What it tests:**
- Concurrent read/write stability
- Lock contention under load
- Memory consistency under parallelism
- Performance under concurrent queries
- Resource cleanup after client disconnects

## Creating New Soak Tests

Soak tests should:

1. **Run for extended periods:**
   ```rust
   const ITERATIONS: usize = 86400; // 24 hours at 1/sec
   const SLEEP_INTERVAL: Duration = Duration::from_secs(1);
   ```

2. **Capture snapshots at key intervals:**
   ```rust
   const STAGES: [Stage; 3] = [
       ("start", 0),
       ("mid", ITERATIONS / 2),
       ("end", ITERATIONS - 1),
   ];
   ```

3. **Monitor resource usage:**
   ```rust
   // Check memory growth
   // Check CPU utilization
   // Check handle/file descriptor counts
   ```

4. **Output diagnostic artifacts:**
   ```rust
   fs::write(format!("{}/stage_{}.json", output_dir, label), snapshot);
   ```

5. **Log progression for monitoring:**
   ```rust
   if iteration % 100 == 0 {
       info!("Progress: {}/{} iterations", iteration, ITERATIONS);
   }
   ```

## Validation Criteria

All soak tests must verify:

1. **No memory leaks:**
   - Memory usage stable over time
   - Use `valgrind --leak-check=full` for verification

2. **Stable performance:**
   - Latency percentiles remain constant
   - No degradation over time
   - Throughput maintains baseline

3. **Resource cleanup:**
   - File descriptors released
   - Thread pools stable
   - Connection pools don't grow indefinitely

4. **Metrics accuracy:**
   - Recorded metrics match actual behavior
   - No counter overflow
   - No gauge drift

## Integration with Milestone Validation

Soak tests support multiple milestone requirements:

- **Milestone 4:** "Verify no memory leaks during long-running decay using valgrind"
- **Milestone 10:** "Benchmark showing sustained 100K observations/second with concurrent recalls"
- **Milestone 13:** "Jepsen-style testing for distributed consistency properties"

## Running Under Observability Tools

### Memory leak detection:
```bash
valgrind --leak-check=full --show-leak-kinds=all \
  cargo run --example memory_pool_soak
```

### CPU profiling:
```bash
cargo flamegraph --example memory_pool_soak
```

### Continuous monitoring:
```bash
# Terminal 1: Run soak test
cargo run --example memory_pool_soak

# Terminal 2: Monitor metrics
watch -n 5 'curl -s http://localhost:7432/metrics | jq .'
```

## Automation

For CI/CD integration, use shorter iterations in automated runs:

```bash
# Quick validation (1 minute)
SOAK_ITERATIONS=60 cargo run --example memory_pool_soak

# Nightly validation (1 hour)
SOAK_ITERATIONS=3600 cargo run --example memory_pool_soak

# Weekly validation (24 hours)
SOAK_ITERATIONS=86400 cargo run --example memory_pool_soak
```

Environment variable support should be added to soak tests for flexible configuration.
