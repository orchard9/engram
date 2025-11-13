# Task 005: Resource Exhaustion Graceful Degradation

## Objective
Validate Engram degrades gracefully under memory and disk pressure without crashes, data corruption, or complete unavailability. Tests admission control, backpressure, and emergency shedding mechanisms.

## Background
Resource exhaustion is a common production failure mode. Systems that crash under pressure experience cascading failures. Engram must:
- Detect resource pressure before OOM killer
- Apply backpressure to clients (reject new writes)
- Shed non-critical operations (pause consolidation)
- Maintain read availability under write pressure
- Recover automatically when resources available

## Requirements

### Functional Requirements
1. Detect memory pressure at 80% threshold
2. Enable admission control (reject writes, accept reads)
3. Pause consolidation during pressure
4. Resume normal operation when <70% memory
5. Zero crashes under sustained pressure

### Non-Functional Requirements
1. Pressure detection latency <1 second
2. Admission control overhead <5ms per request
3. Read availability >99% during write backpressure
4. Automatic recovery within 30s of pressure relief
5. Zero data loss during pressure events

## Technical Specification

### Resource Pressure Scenarios

**Scenario 1: Memory Exhaustion**
- Ingest large episodes until 90% RAM consumed
- Verify admission control rejects new writes
- Confirm reads continue serving from existing data
- Validate no OOM crashes

**Scenario 2: Disk Pressure**
- Fill disk to 95% capacity
- Verify WAL stops accepting writes
- Confirm consolidation paused
- Validate alerts triggered

**Scenario 3: Sustained Pressure**
- Maintain 85% memory for 10 minutes
- Verify backpressure sustained
- Confirm no memory leaks during pressure
- Validate graceful recovery

### Test Implementation

```rust
#[tokio::test]
#[ignore]
async fn test_memory_pressure_handling() -> Result<()> {
    let server = start_test_server(ResourceConfig {
        memory_pressure_threshold: 0.80,
        memory_critical_threshold: 0.90,
        enable_admission_control: true,
        enable_consolidation_shedding: true,
        ..Default::default()
    }).await?;

    server.create_memory_space("pressure_test").await?;

    // PHASE 1: Normal operation baseline
    let baseline_metrics = measure_baseline_performance(&server).await?;

    // PHASE 2: Gradual memory pressure
    println!("=== Applying Memory Pressure ===");
    let mut allocated = 0;
    let mut episodes = Vec::new();

    // Allocate until 80% memory consumed
    loop {
        // Create large episode (10MB)
        let large_episode = create_large_episode(10 * 1024 * 1024)?;

        match server.store("pressure_test", large_episode.clone()).await {
            Ok(_) => {
                episodes.push(large_episode);
                allocated += 10 * 1024 * 1024;
            }
            Err(AdmissionControlError::MemoryPressure) => {
                println!("Admission control engaged at {} MB allocated", allocated / 1024 / 1024);
                break;
            }
            Err(e) => return Err(e.into()),
        }

        // Check if we're at pressure threshold
        let memory_usage = server.get_memory_usage().await?;
        if memory_usage.percent > 0.80 {
            println!("Memory pressure threshold reached: {}%", memory_usage.percent * 100.0);
            break;
        }
    }

    // PHASE 3: Validate admission control active
    let write_result = server.store("pressure_test", create_test_episode()).await;
    assert!(
        matches!(write_result, Err(AdmissionControlError::MemoryPressure)),
        "Admission control should reject writes under pressure"
    );

    // PHASE 4: Validate reads still work
    let read_result = server.recall("pressure_test", create_test_cue()).await;
    assert!(read_result.is_ok(), "Reads should continue under write pressure");

    let read_latency = measure_read_latency(&server, 100).await?;
    assert!(
        read_latency.p99 < baseline_metrics.read_p99 * 1.5,
        "Read latency {} should not degrade >50%",
        read_latency.p99.as_millis()
    );

    // PHASE 5: Validate consolidation paused
    let consolidation_status = server.get_consolidation_status().await?;
    assert_eq!(
        consolidation_status.state,
        ConsolidationState::Paused,
        "Consolidation should pause under memory pressure"
    );

    // PHASE 6: Verify no crashes during sustained pressure
    println!("=== Sustaining Pressure for 5 minutes ===");
    let pressure_start = Instant::now();

    while pressure_start.elapsed() < Duration::from_secs(300) {
        // Attempt occasional writes (should be rejected)
        let _ = server.store("pressure_test", create_test_episode()).await;

        // Perform reads (should succeed)
        let _ = server.recall("pressure_test", create_test_cue()).await;

        sleep(Duration::from_secs(10)).await;

        // Verify server still responsive
        let health = server.health_check().await?;
        assert!(health.is_healthy, "Server crashed under pressure");
    }

    // PHASE 7: Release pressure and validate recovery
    println!("=== Releasing Pressure ===");
    drop(episodes); // Release allocated memory

    // Wait for GC
    sleep(Duration::from_secs(5)).await;

    let recovery_timeout = Duration::from_secs(60);
    let recovery_start = Instant::now();
    let mut recovered = false;

    while recovery_start.elapsed() < recovery_timeout {
        let memory_usage = server.get_memory_usage().await?;

        if memory_usage.percent < 0.70 {
            println!("Memory recovered to {}%", memory_usage.percent * 100.0);

            // Check if admission control lifted
            let write_result = server.store("pressure_test", create_test_episode()).await;

            if write_result.is_ok() {
                println!("Admission control lifted, writes accepted");
                recovered = true;
                break;
            }
        }

        sleep(Duration::from_millis(500)).await;
    }

    assert!(recovered, "System did not recover within 60s");

    // PHASE 8: Validate consolidation resumed
    let final_consolidation_status = server.get_consolidation_status().await?;
    assert_eq!(
        final_consolidation_status.state,
        ConsolidationState::Running,
        "Consolidation should resume after pressure relief"
    );

    Ok(())
}
```

### Observability Validation

**Metrics** (must be captured):
```
memory_usage_bytes{threshold="warning"} > 0.80
memory_usage_bytes{threshold="critical"} > 0.90
admission_control_rejections_total
consolidation_state{state="paused"}
read_requests_served_total (should not drop)
write_requests_rejected_total (should increase under pressure)
```

**Alerts** (must trigger):
```
MemoryPressureWarning: memory_usage > 80%
MemoryPressureCritical: memory_usage > 90%
AdmissionControlEngaged: rejections > 0
ConsolidationPaused: state = paused
```

**Logs** (must contain):
```
WARN: Memory usage 82.3%, enabling admission control
INFO: Consolidation paused due to memory pressure
INFO: Write request rejected: insufficient memory
INFO: Memory usage dropped to 68.1%, lifting admission control
INFO: Consolidation resumed
```

## Acceptance Criteria

### Pass Criteria
1. **Pressure Detection**: Memory pressure detected within 1s of threshold
2. **Admission Control**: Writes rejected, reads continue
3. **Consolidation Shedding**: Background operations paused
4. **Zero Crashes**: Server survives 5+ minutes of sustained pressure
5. **Automatic Recovery**: Normal operation resumed within 60s of relief
6. **Read Availability**: >99% read success rate during pressure
7. **Observability**: All metrics/logs/alerts captured correctly

### Fail Criteria
- Server crashes (OOM killer or panic)
- Data corruption detected post-pressure
- Admission control fails to engage (writes accepted under pressure)
- Recovery takes >60s or requires manual intervention
- Read availability <99% during write backpressure
- Missing telemetry for pressure events

## Performance Budget
Post-pressure recovery must meet baseline within 5%:
- Recall P99: <0.526ms
- Store P99: <0.526ms (after admission control lifted)
- Throughput: >949.9 ops/sec

## Acceptance Script

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Task 005: Resource Exhaustion Acceptance Test ==="

# Run memory pressure test
cargo test --release --features dual_memory_types \
    test_memory_pressure_handling -- --ignored --nocapture

# Run disk pressure test
cargo test --release --features dual_memory_types \
    test_disk_pressure_handling -- --ignored --nocapture

# Validate observability
./scripts/validate_pressure_telemetry.sh

if [ $? -eq 0 ]; then
    echo "PASS: Graceful degradation validated"
    exit 0
else
    echo "FAIL: Review test output"
    exit 1
fi
```

## Estimated Time
2 days:
- Day 1: Memory pressure implementation and testing
- Day 2: Disk pressure, observability validation

## Dependencies
- M17 dual memory implementation
- Admission control mechanism
- Consolidation scheduler with pause/resume

## References
- Admission Control: docs/explanation/admission-control.md
- Resource Management: docs/operations/capacity-planning.md
