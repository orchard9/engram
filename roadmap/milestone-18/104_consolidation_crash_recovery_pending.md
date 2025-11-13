# Task 004: Consolidation Process Crash Recovery

## Objective
Validate Engram recovers correctly from process crashes during consolidation with zero data loss and automatic resumption. Tests critical failure mode: crash during concept formation when episodes and concepts are partially written.

## Background
Consolidation is the highest-risk operation for data integrity:
- Reads episodes from storage (potential partial reads)
- Forms concepts via clustering (stateful computation)
- Writes concepts and bindings atomically (multi-step transaction)
- Updates consolidation scores (metadata mutation)

Crash during any step risks:
- Orphaned concepts (concept exists, no bindings)
- Partial bindings (episode→concept, but not concept→episode)
- Corrupted consolidation scores
- Lost episode data

This test validates WAL-based recovery prevents all these scenarios.

## Requirements

### Functional Requirements
1. Crash process during consolidation at 5 different phases
2. Restart process and verify automatic WAL recovery
3. Validate zero data loss (all episodes intact)
4. Confirm no orphaned concepts or partial bindings
5. Verify consolidation resumes correctly after recovery

### Non-Functional Requirements
1. Recovery time <15 minutes (RTO)
2. Zero data loss (RPO=0)
3. Automatic recovery (no manual intervention)
4. Post-recovery performance within 5% of baseline
5. Crash detection latency <30 seconds

## Technical Specification

### Crash Injection Points
1. **Pre-Clustering**: After episode scan, before clustering begins
2. **Mid-Clustering**: During k-means iteration (partial clusters)
3. **Pre-Concept-Write**: Clusters formed, before concept persistence
4. **Mid-Binding-Write**: Some bindings written, others pending
5. **Post-Consolidation**: After writes, before score updates

### Test Architecture
```rust
#[tokio::test]
#[ignore]
async fn test_consolidation_crash_recovery() -> Result<()> {
    for crash_point in CRASH_INJECTION_POINTS {
        println!("=== Testing crash at: {:?} ===", crash_point);

        // Setup: Fresh server with 10K episodes
        let mut server = start_test_server(TestConfig::default()).await?;
        ingest_test_data(&server, 10_000).await?;

        // Enable crash injection
        server.enable_crash_injection(crash_point).await?;

        // Trigger consolidation (will crash at injection point)
        let consolidation_handle = tokio::spawn(async move {
            server.trigger_consolidation("test_space").await
        });

        // Wait for crash detection
        let crash_detected = wait_for_crash(&server, Duration::from_secs(30)).await?;
        assert!(crash_detected, "Crash not detected within 30s");

        // Record pre-crash state
        let pre_crash_episodes = count_episodes_in_wal("test_space")?;
        let pre_crash_concepts = count_concepts_in_wal("test_space")?;

        // Restart server (triggers WAL recovery)
        let recovery_start = Instant::now();
        let recovered_server = restart_server().await?;
        let recovery_duration = recovery_start.elapsed();

        assert!(
            recovery_duration < Duration::from_secs(15 * 60),
            "Recovery took {} > 15 minutes",
            recovery_duration.as_secs()
        );

        // VALIDATION 1: Zero data loss
        let post_crash_episodes = recovered_server.count_episodes("test_space").await?;
        assert_eq!(
            post_crash_episodes, pre_crash_episodes,
            "Episode count mismatch: {} != {}",
            post_crash_episodes, pre_crash_episodes
        );

        // VALIDATION 2: No orphaned concepts
        let orphaned = recovered_server.find_orphaned_concepts("test_space").await?;
        assert_eq!(orphaned.len(), 0, "Found {} orphaned concepts", orphaned.len());

        // VALIDATION 3: No partial bindings
        let partial_bindings = recovered_server.find_partial_bindings("test_space").await?;
        assert_eq!(
            partial_bindings.len(), 0,
            "Found {} partial bindings",
            partial_bindings.len()
        );

        // VALIDATION 4: Consolidation resumes correctly
        let post_recovery_consolidation = recovered_server
            .trigger_consolidation("test_space")
            .await?;

        assert!(
            post_recovery_consolidation.success,
            "Post-recovery consolidation failed"
        );

        // VALIDATION 5: Performance regression <5%
        let baseline_latency = measure_recall_latency(&baseline_server).await?;
        let recovered_latency = measure_recall_latency(&recovered_server).await?;
        let regression = (recovered_latency - baseline_latency) / baseline_latency;

        assert!(
            regression < 0.05,
            "Performance regression {:.1}% > 5%",
            regression * 100.0
        );

        println!("✓ Crash at {:?} recovered successfully", crash_point);
    }

    Ok(())
}
```

### WAL Recovery Validation
```rust
// Verify WAL recovery completeness
fn validate_wal_recovery(space_id: &str) -> Result<()> {
    let wal_path = format!("data/wal/{}/", space_id);
    let wal_entries = parse_wal(&wal_path)?;

    // All entries must have monotonic sequence numbers
    let mut prev_seq = 0;
    for entry in &wal_entries {
        assert!(
            entry.sequence > prev_seq,
            "Non-monotonic WAL sequence: {} <= {}",
            entry.sequence, prev_seq
        );
        prev_seq = entry.sequence;
    }

    // All concept writes must have corresponding binding writes
    let concept_writes: HashSet<_> = wal_entries
        .iter()
        .filter_map(|e| match e.operation {
            WalOp::ConceptWrite(id) => Some(id),
            _ => None,
        })
        .collect();

    let binding_writes: HashSet<_> = wal_entries
        .iter()
        .filter_map(|e| match e.operation {
            WalOp::BindingWrite(concept_id, _) => Some(concept_id),
            _ => None,
        })
        .collect();

    for concept_id in &concept_writes {
        assert!(
            binding_writes.contains(concept_id),
            "Concept {} has no bindings in WAL",
            concept_id
        );
    }

    Ok(())
}
```

## Acceptance Criteria

### Pass Criteria (ALL must be met for ALL crash points)
1. **Zero Data Loss**: Episode count unchanged after recovery
2. **No Orphaned Concepts**: All concepts have ≥1 binding
3. **No Partial Bindings**: Episode↔Concept bidirectional consistency
4. **Automatic Recovery**: No manual intervention required
5. **RTO Met**: Recovery completes within 15 minutes
6. **Performance Preserved**: <5% regression post-recovery
7. **Resumption Success**: Post-recovery consolidation succeeds

### Fail Criteria (ANY triggers failure)
- Data loss detected (even 1 episode)
- Orphaned concepts or partial bindings found
- Recovery requires manual intervention
- RTO exceeded (>15 minutes)
- Performance regression ≥5%
- Post-recovery consolidation fails

## Observability Validation

During crash and recovery, verify telemetry captures:

**Metrics**:
```
crash_detected_total{crash_point="mid_clustering"}
wal_recovery_duration_seconds
wal_entries_replayed_total
orphaned_concepts_detected_total (should be 0)
partial_bindings_detected_total (should be 0)
```

**Logs** (must contain):
```
ERROR: Process crash detected at consolidation phase: mid_clustering
INFO: Starting WAL recovery for space: test_space
INFO: Replayed 1234 WAL entries in 23.4s
INFO: Recovery complete, resuming normal operation
```

**Alerts** (should trigger):
- `ProcessCrashDetected` - within 30s of crash
- `WalRecoveryInProgress` - immediately on restart
- `WalRecoveryComplete` - after successful recovery

## Testing Approach

### Crash Injection Mechanism
```rust
// Crash injection via conditional panic
pub struct CrashInjector {
    enabled: Arc<AtomicBool>,
    crash_point: Arc<RwLock<Option<CrashPoint>>>,
}

impl CrashInjector {
    pub fn maybe_crash(&self, point: CrashPoint) {
        if self.enabled.load(Ordering::Relaxed) {
            if let Some(target) = *self.crash_point.read().unwrap() {
                if target == point {
                    panic!("INJECTED CRASH: {:?}", point);
                }
            }
        }
    }
}

// In consolidation code:
crash_injector.maybe_crash(CrashPoint::PreClustering);
let clusters = perform_clustering(episodes)?;
crash_injector.maybe_crash(CrashPoint::MidClustering);
```

### Acceptance Script
**File**: `/Users/jordanwashburn/Workspace/orchard9/engram/scripts/acceptance/004_crash_recovery.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== Task 004: Consolidation Crash Recovery Acceptance Test ==="

# Requires crash injection feature
cargo test --release --features "dual_memory_types,crash_injection" \
    test_consolidation_crash_recovery -- --ignored --nocapture --test-threads=1

if [ $? -eq 0 ]; then
    echo "PASS: All crash points recovered successfully"
    echo "✓ Zero data loss across all scenarios"
    echo "✓ No orphaned concepts or partial bindings"
    echo "✓ Automatic recovery within RTO"
    exit 0
else
    echo "FAIL: Crash recovery validation failed"
    exit 1
fi
```

## Performance Budget
Post-recovery system must meet M17 baseline within 5% tolerance:
- Recall P99: <0.526ms
- Store P99: <0.526ms
- Throughput: >949.9 ops/sec

## Estimated Time
2 days:
- Day 1: Crash injection infrastructure, WAL validation
- Day 2: All 5 crash points tested, observability validation

## Dependencies
- M6 (Consolidation system with WAL)
- M17 Task 006 (Consolidation integration)

## Follow-Up Tasks
- If recovery >15 min: Optimize WAL replay (parallel recovery)
- If data loss detected: Critical bug, halt all other work
- If orphans found: Fix consolidation transaction atomicity

## References
- WAL Design: docs/explanation/write-ahead-log.md
- Consolidation: roadmap/milestone-6/
- Crash Recovery: docs/operations/disaster-recovery.md
