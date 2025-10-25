# Backup and Disaster Recovery - Architectural Perspectives

## Systems Architecture Optimizer Perspective

### The Consistency Triangle: Fast, Consistent, or Always-On

You cannot have all three. Pick two.

**Fast Backups + Consistent State = Downtime**
- Quiesce writes, snapshot all tiers, resume
- Duration: 2-3 minutes
- Frequency: Nightly at 2 AM

**Fast Backups + Always-On = Potential Inconsistency**
- Online snapshots without coordination
- Risk: Fast tier at T=0, warm tier at T=5min
- Result: Mismatched state on restore

**Consistent State + Always-On = Slow Backups**
- MVCC-style copy-on-write
- Maintain shadow copies
- Overhead: 20-30% performance hit

**Engram's Choice:**
- Full backups: Fast + Consistent (accept 2min downtime at 2 AM)
- Incremental backups: Consistent + Always-On (use operation log)

### Tiered Backup Strategy Matching Storage Tiers

Graph memory has three tiers. Backups mirror this architecture.

**Fast Tier (Active Working Set):**
- Data: Recently activated patterns in RAM
- Backup: Operation log only (reconstruct on restore)
- Rationale: Fast tier is cache, can rebuild from warm tier + logs
- Size: 10GB operation logs vs 100GB RAM snapshot

**Warm Tier (Persistent Graph State):**
- Data: Complete graph structure on SSD
- Backup: Full snapshot + incrementals
- Rationale: Source of truth, must be backed up completely
- Size: 100GB compressed to 30GB

**Cold Tier (Archival Storage):**
- Data: Rarely accessed historical data
- Backup: Replicated to second region
- Rationale: Already on durable storage (S3), replication is backup
- Size: 1TB with cross-region replication

**Backup Bandwidth Allocation:**
- Fast tier: 10% (small, frequent)
- Warm tier: 80% (large, critical)
- Cold tier: 10% (already replicated)

### Lock-Free Incremental Backup

Traditional approach: Acquire write lock, copy data, release lock.

Problem: Blocks all writes during backup.

**Lock-Free Approach:**

```rust
// Operation log is append-only
struct OperationLog {
    entries: Arc<SegQueue<Operation>>,  // Lock-free queue
}

impl OperationLog {
    fn append(&self, op: Operation) {
        self.entries.push(op);  // No locks
    }

    fn snapshot(&self) -> Vec<Operation> {
        // Drain current entries without blocking appends
        let mut ops = Vec::new();
        while let Some(op) = self.entries.pop() {
            ops.push(op);
        }
        ops
    }
}
```

**Result:** Writes continue during backup. Zero blocking time.

**Tradeoff:** Operations during snapshot may or may not be included. Acceptable because next incremental catches them.

## Rust Graph Engine Architect Perspective

### Backup as Serialization Problem

Backup is just serialization. Restore is deserialization. Use Rust's type system to enforce correctness.

**Backup Manifest:**

```rust
#[derive(Serialize, Deserialize)]
struct BackupManifest {
    version: u32,
    timestamp: SystemTime,
    tiers: TierBackups,
    checksum: [u8; 32],
}

#[derive(Serialize, Deserialize)]
struct TierBackups {
    fast: Option<PathBuf>,   // Optional: Can rebuild from warm + logs
    warm: PathBuf,            // Required: Source of truth
    cold: Option<PathBuf>,   // Optional: Already replicated
}
```

**Type-Safe Restore:**

```rust
async fn restore(manifest: BackupManifest) -> Result<Graph> {
    // Compiler enforces warm tier is always present
    let warm_data = fs::read(&manifest.tiers.warm)?;

    let graph = Graph::from_backup(warm_data)?;

    // Fast tier is optional, rebuild if missing
    if let Some(fast_backup) = manifest.tiers.fast {
        graph.restore_fast_tier(fast_backup).await?;
    } else {
        graph.rebuild_fast_tier().await?;
    }

    Ok(graph)
}
```

**Versioning for Forward Compatibility:**

```rust
async fn restore(manifest: BackupManifest) -> Result<Graph> {
    match manifest.version {
        1 => restore_v1(manifest).await,
        2 => restore_v2(manifest).await,
        v => Err(format!("Unsupported backup version: {}", v).into()),
    }
}
```

**Why this matters:** Backups outlive code versions. Must be able to restore 6-month-old backup with current code.

### Streaming Backup for Large Graphs

Loading 100GB into memory for backup is wasteful. Stream directly to storage.

**Streaming Approach:**

```rust
use tokio::io::AsyncWriteExt;

async fn backup_streaming(
    graph: &Graph,
    output: &mut impl AsyncWrite
) -> Result<()> {
    // Write header
    output.write_u32(BACKUP_VERSION).await?;
    output.write_u64(graph.node_count()).await?;

    // Stream nodes in batches
    let batch_size = 10_000;
    let mut batch = Vec::with_capacity(batch_size);

    for node in graph.iter_nodes() {
        batch.push(node);

        if batch.len() == batch_size {
            serialize_batch(&batch, output).await?;
            batch.clear();
        }
    }

    // Write remaining nodes
    if !batch.is_empty() {
        serialize_batch(&batch, output).await?;
    }

    Ok(())
}
```

**Memory usage:** Constant 10MB buffer instead of 100GB full graph.

**Performance:** Stream compression (gzip) happens inline, no temporary files.

### Parallel Backup for Multi-Core

Modern servers have many cores. Use them.

**Parallel Backup:**

```rust
async fn backup_parallel(graph: &Graph, output_dir: PathBuf) -> Result<()> {
    let shard_count = num_cpus::get();
    let handles: Vec<_> = (0..shard_count)
        .map(|shard_id| {
            let graph = graph.clone();  // Arc clone, cheap
            let output = output_dir.join(format!("shard-{}.bin", shard_id));

            tokio::spawn(async move {
                backup_shard(&graph, shard_id, shard_count, output).await
            })
        })
        .collect();

    // Wait for all shards
    for handle in handles {
        handle.await??;
    }

    Ok(())
}

async fn backup_shard(
    graph: &Graph,
    shard_id: usize,
    shard_count: usize,
    output: PathBuf
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(output)?);

    for node in graph.iter_nodes() {
        // Partition by node ID hash
        if node.id().hash() % shard_count == shard_id {
            serialize_node(&node, &mut writer)?;
        }
    }

    writer.flush()?;
    Ok(())
}
```

**Result:** 8-core machine backs up 8x faster. Restores also parallelize.

## Verification Testing Lead Perspective

### Backup Testing as Continuous Validation

Untested backups are Schrödinger's backups. They're both working and broken until you try to restore.

**Automated Backup Verification:**

```bash
#!/bin/bash
# Run after every backup

BACKUP_FILE=$1
TEST_DIR=/tmp/restore-$(uuidgen)

# Restore to temporary directory
engram restore --from=$BACKUP_FILE --to=$TEST_DIR || {
    echo "CRITICAL: Backup restore failed"
    alert_oncall "Backup restoration failed: $BACKUP_FILE"
    exit 1
}

# Integrity check
engram verify --data-dir=$TEST_DIR --full || {
    echo "CRITICAL: Backup data corrupted"
    alert_oncall "Backup integrity check failed: $BACKUP_FILE"
    exit 1
}

# Performance test (should be near production speed)
LATENCY=$(engram benchmark --data-dir=$TEST_DIR --duration=10s --metric=p50)

if (( $(echo "$LATENCY > 10" | bc -l) )); then
    echo "WARNING: Restored backup has degraded performance: ${LATENCY}ms"
    # Don't fail, but log warning
fi

# Cleanup
rm -rf $TEST_DIR

echo "Backup verification passed: $BACKUP_FILE"
```

**Run this after every backup. No exceptions.**

### Chaos Testing: Deliberate Failures

Don't wait for real disasters. Simulate them.

**Monthly Chaos Scenarios:**

1. **Random Node Deletion:**
   ```bash
   # Delete 10% of random nodes
   engram chaos delete-nodes --percent=10
   # Restore from backup
   engram restore --from=latest
   # Verify missing nodes recovered
   ```

2. **Corruption Injection:**
   ```bash
   # Corrupt random bytes in warm tier
   dd if=/dev/urandom of=/var/lib/engram/warm/graph.bin \
     bs=1 count=1024 seek=$RANDOM conv=notrunc
   # Should detect corruption
   engram verify  # Should fail
   # Restore from backup
   engram restore --from=latest
   ```

3. **Disk Full:**
   ```bash
   # Fill disk to 99%
   fallocate -l $(df --output=avail /var/lib/engram | tail -1)K \
     /var/lib/engram/fill.bin
   # Trigger backup (should fail gracefully)
   engram backup  # Should detect and abort
   # Cleanup
   rm /var/lib/engram/fill.bin
   ```

4. **Network Partition During Backup:**
   ```bash
   # Start backup upload to S3
   engram backup --upload &
   BACKUP_PID=$!
   sleep 5
   # Block S3 traffic
   iptables -A OUTPUT -d s3.amazonaws.com -j DROP
   # Backup should retry and eventually fail
   wait $BACKUP_PID
   # Restore connectivity
   iptables -D OUTPUT -d s3.amazonaws.com -j DROP
   ```

**Document every failure. Fix every failure. Repeat monthly.**

### Restore Time Objective Testing

RTO <30 minutes is a promise. Measure it.

**Automated RTO Test:**

```bash
#!/bin/bash
# Simulate complete cluster loss

# Record start time
START=$(date +%s)

# 1. Delete all data
rm -rf /var/lib/engram/*

# 2. Download latest backup
aws s3 cp s3://engram-backups/latest.tar.gz /tmp/backup.tar.gz

# 3. Restore
engram restore --from=/tmp/backup.tar.gz --to=/var/lib/engram

# 4. Start server
systemctl start engram

# 5. Wait for readiness
while ! curl -f http://localhost:8080/health/ready; do
    sleep 1
done

# Record end time
END=$(date +%s)
DURATION=$((END - START))

# Verify RTO
if [ $DURATION -gt 1800 ]; then  # 30 minutes
    echo "FAIL: RTO exceeded: ${DURATION}s > 1800s"
    exit 1
else
    echo "PASS: RTO met: ${DURATION}s < 1800s"
fi
```

**Run this quarterly. Document actual time. Optimize until consistent <30min.**

## Cognitive Architecture Designer Perspective

### Memory Consolidation and Backup Synchronization

Biological memory consolidation happens during sleep. Digital backup consolidation should too.

**Nightly Backup as Sleep Cycle:**

```rust
async fn nightly_maintenance() {
    // 2:00 AM - Low traffic window
    info!("Starting nightly maintenance cycle");

    // Step 1: Consolidate memories (hippocampus → neocortex analog)
    graph.consolidate_fast_to_warm().await;

    // Step 2: Optimize indexes (synaptic pruning analog)
    graph.rebuild_indexes().await;

    // Step 3: Snapshot consistent state
    let backup = graph.backup_full().await;

    // Step 4: Upload to archival storage
    cloud_storage.upload(backup).await;

    info!("Nightly maintenance complete");
}
```

**Why this works:**
- Consolidation ensures fast tier is flushed
- Backup captures consolidated state
- Restore has optimized indexes, no rebuild needed

### Recovery Point Objective as Memory Fidelity

RPO <5 minutes means we can lose up to 5 minutes of memories.

For a cognitive system, this is acceptable:
- Short-term working memory loss is tolerable
- Long-term consolidated memories are preserved
- System can relearn recent patterns quickly

**Analogy:** Like minor head trauma. Recent events fuzzy, long-term memories intact.

**Implementation:**

```rust
// Incremental backup every 5 minutes
async fn incremental_backup_loop() {
    let mut interval = tokio::time::interval(Duration::from_secs(300));

    loop {
        interval.tick().await;

        // Capture operations from last 5 minutes
        let ops = operation_log.since(now() - 5.minutes()).await;

        // Append to incremental backup
        backup_storage.append_incremental(ops).await;

        info!("Incremental backup complete: {} operations", ops.len());
    }
}
```

## Synthesis: Production Backup Strategy

**Full Backup (Nightly at 2 AM):**
- Quiesce writes for 2-3 minutes
- Snapshot all tiers at consistent timestamp
- Compress and encrypt
- Upload to S3 with cross-region replication
- Validate by test restore
- Retention: 7 daily, 4 weekly, 12 monthly

**Incremental Backup (Every 5 Minutes):**
- Stream operation log to backup storage
- No write blocking (lock-free append)
- Minimal overhead (<2% CPU)
- Retention: Keep all for 24 hours

**Disaster Recovery:**
1. Download latest full backup (2 minutes)
2. Download incremental logs since backup (1 minute)
3. Restore full backup (10 minutes)
4. Replay incremental logs (5 minutes)
5. Warmup and validation (10 minutes)
6. Total: 28 minutes (within 30-minute RTO)

**Monthly Validation:**
- Chaos testing: Deliberate corruption and restore
- RTO measurement: Full cluster rebuild
- RPO verification: Incremental restore accuracy
- Documentation update: Actual times vs targets

**Monitoring:**
- Backup success/failure rate
- Backup duration trends
- Restore test results
- Storage usage and growth rate

No backup strategy survives first contact with disaster. Test it. Measure it. Fix it. Repeat.
