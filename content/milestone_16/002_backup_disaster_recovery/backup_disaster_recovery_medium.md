# Why Your Graph Database Backups Are Probably Broken (And How to Fix Them)

Here's an uncomfortable truth: Most teams discover their backups don't work when they desperately need them to work.

The backup script ran successfully for 6 months. The monitoring says green. Then disaster strikes. You restore the backup and discover the graph is corrupted, half the nodes are missing, or the indexes are broken. Your RTO was 30 minutes. Actual recovery time: 6 hours of panic and manual fixes.

Graph databases make backup particularly hard. Unlike relational databases with clean transaction boundaries, graphs have complex interconnected state spread across multiple storage tiers. A naive backup captures inconsistent snapshots that restore into corrupted graphs.

This article shows how to design bulletproof backup and disaster recovery for stateful graph systems, with specific focus on the hard parts traditional database backup guides skip over.

## The Multi-Tier Consistency Problem

Engram has three storage tiers, like many modern database systems:
- Fast tier: Active working set in RAM (10GB)
- Warm tier: Complete graph on SSD (100GB)
- Cold tier: Archival data in object storage (1TB)

Naive backup approach:

```bash
# DON'T DO THIS
backup_fast_tier   # T=0:00
backup_warm_tier   # T=5:00  (5 minutes later)
backup_cold_tier   # T=10:00 (10 minutes later)
```

Problem: Three snapshots at different timestamps. A memory activated at T=2:00 is in fast tier backup but not warm tier backup. Restore creates broken references.

**Correct approach:**

```rust
async fn backup_consistent_snapshot() -> Result<BackupManifest> {
    // Step 1: Acquire backup lock (prevents consolidation between tiers)
    let _lock = backup_coordinator.lock().await;

    // Step 2: Get wall-clock timestamp
    let snapshot_time = SystemTime::now();

    // Step 3: Snapshot all tiers at exact same logical time
    let (fast_snap, warm_snap, cold_snap) = tokio::join!(
        fast_tier.snapshot_at(snapshot_time),
        warm_tier.snapshot_at(snapshot_time),
        cold_tier.snapshot_at(snapshot_time)
    );

    // Step 4: Release lock immediately
    drop(_lock);

    // Step 5: Stream snapshots to backup storage (can be slow)
    let manifest = BackupManifest {
        timestamp: snapshot_time,
        tiers: vec![fast_snap?, warm_snap?, cold_snap?],
    };

    Ok(manifest)
}
```

Key insight: Coordination happens in milliseconds (acquire lock, record timestamp, initiate snapshots). The slow part (streaming data to storage) happens after lock release with consistent snapshots already captured.

## Full vs Incremental: The Tradeoff Matrix

**Full Backup:**
- Complete snapshot of all data
- Slowest to create (2-3 minutes quiesce time)
- Fastest to restore (single file)
- Largest storage footprint (100GB compressed)

**Incremental Backup:**
- Only operations since last backup
- Fastest to create (lock-free append)
- Slower to restore (full + all incrementals)
- Smallest storage footprint (10GB logs per day)

**Engram's Strategy:**

Full backup nightly at 2 AM:
```bash
#!/bin/bash
# Runs at 2:00 AM when traffic is lowest

# Quiesce writes for consistency
engram quiesce --timeout=3m

# Take snapshot
engram backup --full --output=/var/backups/full-$(date +%Y%m%d).tar.gz

# Resume writes
engram resume

# Upload to S3
aws s3 cp /var/backups/full-$(date +%Y%m%d).tar.gz \
  s3://engram-backups/ --storage-class STANDARD_IA

# Validate backup
engram validate-backup /var/backups/full-$(date +%Y%m%d).tar.gz
```

Incremental backup every 5 minutes:
```bash
#!/bin/bash
# Runs continuously via systemd timer

# No quiesce needed, lock-free operation log
engram backup --incremental --since=5m \
  --output=/var/backups/incremental-$(date +%Y%m%d-%H%M).log

# Incrementals are tiny, keep locally
# Only upload to S3 if remote RPO required
```

**Why this specific combination?**

Recovery scenarios:

1. **Last night's data:** Restore from full backup (10 minutes)
2. **Data from 2 hours ago:** Restore full + last 2 hours of incrementals (15 minutes)
3. **Data from 30 minutes ago:** Restore full + all incrementals (20 minutes)
4. **Data from 2 minutes ago:** Restore full + all incrementals (25 minutes)

All scenarios hit RTO <30 minutes target.

## Point-in-Time Recovery: The Time Machine

PITR enables restore to any moment, not just backup snapshots.

**Requirements:**
1. Base backup (full snapshot)
2. Operation log (every mutation since backup)
3. Replay capability (deterministic operation application)

**Operation Log Schema:**

```rust
#[derive(Serialize, Deserialize)]
struct Operation {
    timestamp: SystemTime,
    op_type: OpType,
    memory_id: Uuid,
    before_state: Option<Vec<u8>>,  // For rollback
    after_state: Vec<u8>,            // For replay
}

enum OpType {
    CreateMemory,
    UpdateStrength,
    DeleteMemory,
    CreateEdge,
    DeleteEdge,
}
```

**Replay Implementation:**

```rust
async fn restore_to_time(
    base_backup: PathBuf,
    operation_log: PathBuf,
    target_time: SystemTime
) -> Result<Graph> {
    // Step 1: Restore base backup
    let mut graph = Graph::restore_from(base_backup)?;
    info!("Base backup restored: {}", graph.node_count());

    // Step 2: Open operation log
    let log = OperationLog::open(operation_log)?;

    // Step 3: Replay operations up to target time
    let mut replayed = 0;
    for op in log.iter() {
        if op.timestamp > target_time {
            break;  // Stop at target time
        }

        graph.apply_operation(op).await?;
        replayed += 1;

        if replayed % 10000 == 0 {
            info!("Replayed {} operations", replayed);
        }
    }

    info!("PITR complete: {} operations replayed", replayed);
    Ok(graph)
}
```

**Use Case: Accidental Deletion**

```bash
# Operator accidentally deletes critical node at 14:32
DELETE FROM memories WHERE id='critical-node';

# Detected at 14:45, want to recover

# Restore to 14:30 (before deletion)
engram restore-pitr \
  --base-backup=/var/backups/full-20251024.tar.gz \
  --operation-log=/var/backups/oplog-20251024.log \
  --target-time="2025-10-24 14:30:00"

# Verify node exists
engram query --id='critical-node'  # Found

# Resume operations with recovered state
```

## The 3-2-1 Rule: Storage Strategy

Industry best practice: 3 copies, 2 media types, 1 offsite.

**Engram Implementation:**

**Copy 1: Production Data**
- Location: Local SSD (/var/lib/engram)
- Purpose: Live operational data
- Protection: RAID for disk failure

**Copy 2: Local Backup**
- Location: Different disk (/var/backups/engram)
- Purpose: Fast restore for local failures
- Protection: Separate physical disk

**Copy 3: Cloud Backup**
- Location: S3 with cross-region replication
- Purpose: Disaster recovery, offsite
- Protection: Geographic separation

**Retention Policy:**

```python
retention = {
    "hourly":  {"count": 24,  "keep": "1 day"},
    "daily":   {"count": 7,   "keep": "1 week"},
    "weekly":  {"count": 4,   "keep": "1 month"},
    "monthly": {"count": 12,  "keep": "1 year"},
}
```

**Pruning Script:**

```bash
#!/bin/bash
# Prune backups according to retention policy

# Keep last 24 hourly backups
ls -t /var/backups/hourly-*.tar.gz | tail -n +25 | xargs rm -f

# Keep last 7 daily backups
ls -t /var/backups/daily-*.tar.gz | tail -n +8 | xargs rm -f

# Keep last 4 weekly backups
ls -t /var/backups/weekly-*.tar.gz | tail -n +5 | xargs rm -f

# Keep last 12 monthly backups
ls -t /var/backups/monthly-*.tar.gz | tail -n +13 | xargs rm -f

echo "Backup pruning complete"
```

## Backup Validation: Trust But Verify

Backups you haven't tested are Schrödinger's backups. They're simultaneously working and broken until you need them.

**Automated Validation After Every Backup:**

```bash
#!/bin/bash
# /usr/local/bin/validate-backup.sh

BACKUP_FILE=$1
TEST_DIR=/tmp/restore-test-$(uuidgen)

echo "Validating backup: $BACKUP_FILE"

# Step 1: Checksum verification
EXPECTED=$(cat ${BACKUP_FILE}.sha256)
ACTUAL=$(sha256sum $BACKUP_FILE | awk '{print $1}')

if [ "$EXPECTED" != "$ACTUAL" ]; then
    echo "FAIL: Checksum mismatch"
    alert_oncall "Backup corrupted: $BACKUP_FILE"
    exit 1
fi

# Step 2: Restore to temporary location
engram restore --from=$BACKUP_FILE --to=$TEST_DIR || {
    echo "FAIL: Restore failed"
    alert_oncall "Backup restore failed: $BACKUP_FILE"
    exit 1
}

# Step 3: Integrity check
engram verify --data-dir=$TEST_DIR --check-all || {
    echo "FAIL: Graph integrity check failed"
    alert_oncall "Backup data corrupted: $BACKUP_FILE"
    exit 1
}

# Step 4: Performance test
LATENCY=$(engram benchmark --data-dir=$TEST_DIR --duration=10s | grep 'p50:' | awk '{print $2}')

if (( $(echo "$LATENCY > 10" | bc -l) )); then
    echo "WARNING: Performance degraded: ${LATENCY}ms"
fi

# Step 5: Cleanup
rm -rf $TEST_DIR

echo "PASS: Backup validated successfully"
```

**Run this after every backup. No exceptions.**

## Disaster Recovery Runbook

Abstract procedures don't work under pressure. Concrete steps do.

**Scenario 1: Complete Data Loss**

Context: Production server disk failed. All local data lost.
RTO: 30 minutes
RPO: 5 minutes (last incremental backup)

**Steps:**

```bash
# 1. Provision new server or repair existing (assume 5 minutes)

# 2. Download latest full backup from S3
aws s3 cp s3://engram-backups/full-latest.tar.gz /tmp/backup.tar.gz
# Time: 2 minutes (100GB over 400Mbps)

# 3. Download incremental logs since full backup
aws s3 sync s3://engram-backups/incremental/ /tmp/incrementals/
# Time: 30 seconds (10GB)

# 4. Restore full backup
engram restore --from=/tmp/backup.tar.gz --to=/var/lib/engram
# Time: 10 minutes (decompress + write to SSD)

# 5. Apply incremental logs
for log in /tmp/incrementals/*.log; do
    engram apply-log --log=$log
done
# Time: 5 minutes (replay operations)

# 6. Start server
systemctl start engram
# Time: 30 seconds

# 7. Wait for readiness
while ! curl -f http://localhost:8080/health/ready; do
    sleep 1
done
# Time: 2 minutes (warmup)

# 8. Smoke test
engram test --quick
# Time: 1 minute

# Total: 26 minutes (within 30-minute RTO)

echo "Disaster recovery complete"
```

**Scenario 2: Data Corruption Detected**

Context: Automated integrity check detected corruption. Need to restore to point before corruption.
RTO: 20 minutes
RPO: To point before corruption (PITR)

**Steps:**

```bash
# 1. Identify corruption time from logs
CORRUPTION_TIME=$(grep "integrity check failed" /var/log/engram.log | tail -1 | awk '{print $1}')
echo "Corruption detected at: $CORRUPTION_TIME"

# 2. Calculate restore target (5 minutes before corruption)
RESTORE_TIME=$(date -d "$CORRUPTION_TIME - 5 minutes" +"%Y-%m-%d %H:%M:%S")
echo "Restoring to: $RESTORE_TIME"

# 3. Stop production server
systemctl stop engram

# 4. Backup corrupted data (for forensics)
mv /var/lib/engram /var/lib/engram-corrupted-$(date +%s)

# 5. Restore to point before corruption
engram restore-pitr \
    --base-backup=/var/backups/full-latest.tar.gz \
    --operation-log=/var/log/engram-operations.log \
    --target-time="$RESTORE_TIME" \
    --output=/var/lib/engram

# Time: 15 minutes

# 6. Verify integrity
engram verify --data-dir=/var/lib/engram --check-all

# 7. Restart server
systemctl start engram

# 8. Monitor for recurrence
tail -f /var/log/engram.log | grep "integrity"

echo "Corruption recovery complete"
```

**Scenario 3: Accidental Deletion**

Context: Operator accidentally deleted 10,000 nodes. Need to recover them.
RTO: 15 minutes
RPO: Before deletion

[Similar detailed steps with specific commands and timings]

## Monthly Chaos Drills

Quarterly full disaster recovery exercise:

**Week 1: Deliberate Corruption**
```bash
# Corrupt 1% of random data
python3 /opt/engram/chaos/corrupt-data.py --percent=1
# Detect corruption: Should trigger alerts
# Restore from backup
# Verify complete recovery
```

**Week 2: Disk Full**
```bash
# Fill disk to 99%
# Trigger backup: Should fail gracefully
# Clean up and verify normal backup resumes
```

**Week 3: Network Partition**
```bash
# Block cloud backup upload mid-transfer
# Verify retry logic works
# Verify backup eventually succeeds
```

**Week 4: Full Datacenter Loss**
```bash
# Simulate: Delete all production data
# Restore from S3 backup only
# Measure actual RTO
# Update runbook with actual times
```

Document every drill. Fix every failure. Update the runbook.

## Monitoring and Alerting

What to monitor:

```yaml
alerts:
  - name: BackupFailed
    condition: backup_success_rate < 1.0
    severity: critical
    message: "Backup failed, data at risk"

  - name: BackupValidationFailed
    condition: backup_validation_success < 1.0
    severity: critical
    message: "Backup is corrupted"

  - name: BackupDurationHigh
    condition: backup_duration_seconds > 300
    severity: warning
    message: "Backup taking too long"

  - name: RestoreTestFailed
    condition: daily_restore_test_success < 1.0
    severity: critical
    message: "Daily restore test failed"

  - name: BackupStorageGrowth
    condition: rate(backup_storage_bytes[7d]) > threshold
    severity: warning
    message: "Backup storage growing faster than expected"
```

## The Uncomfortable Truth

Most backup failures are discovered during restore, when it's too late.

The only way to know your backups work: Test them continuously. Restore daily. Run chaos drills monthly. Measure actual RTO. Fix what breaks.

Your backup strategy should be boring and reliable, not clever and untested. Full backups at low-traffic times. Incremental backups continuously. Automated validation after every backup. Quarterly disaster drills with actual timings.

And most importantly: Actually restore from those backups regularly, even when you don't have to. Because the worst time to discover your backups don't work is when you desperately need them to work.
