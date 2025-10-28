# Backup & Restore

Comprehensive backup and restore procedures for Engram with RTO <30min and RPO <5min.

## Overview

Engram provides multi-tier backup capabilities designed for cognitive memory systems with point-in-time recovery (PITR) support.

### Key Metrics

- **RTO (Recovery Time Objective)**: <30 minutes

- **RPO (Recovery Point Objective)**: <5 minutes (with incremental backups)

- **Compression**: zstd level 3 (>5:1 ratio for embeddings)

- **Verification**: 4-level integrity checking (L1-L4)

### Backup Types

1. **Full Backup**: Complete snapshot of all tiers (hot/warm/cold) and WAL

2. **Incremental Backup**: WAL-based changes since last checkpoint

3. **Differential Backup**: Tier-aware changes (80-90% size reduction)

## Storage Architecture

Engram's tiered storage system is designed for efficient backup:

```
/var/lib/engram/
├── spaces/              # Per-space data directories
│   ├── {space_id}/
│   │   ├── wal/        # Write-ahead logs
│   │   ├── hot/        # Hot tier (DashMap snapshots)
│   │   ├── warm_tier.dat  # Warm tier (mmap files)
│   │   └── cold/       # Cold tier (columnar storage)
└── backups/
    ├── full/           # Full backups
    ├── incremental/    # WAL archives
    └── manifests/      # Backup manifests with checksums

```

## Full Backup

Create complete snapshot of all memory spaces.

### Basic Usage

```bash
# Backup default space
/scripts/backup_full.sh

# Backup all spaces
ENGRAM_SPACE_ID=all /scripts/backup_full.sh

# Backup specific space
ENGRAM_SPACE_ID=production /scripts/backup_full.sh

# Custom backup location
BACKUP_DIR=/mnt/backups /scripts/backup_full.sh

```

### Performance Characteristics

- **1GB database**: <5 minutes

- **10GB database**: <15 minutes

- **100GB database**: <60 minutes

- **Parallel compression**: 4 threads (configurable via BACKUP_PARALLEL)

### What's Included

Full backups capture:

- Hot tier DashMap exports (in-memory data)

- Warm tier memory-mapped files (recently accessed)

- Cold tier columnar storage (long-term storage)

- All WAL segments (for PITR capability)

- Per-file SHA256 checksums

- Backup manifest with metadata

### Output

```
[backup] Starting full backup at 2024-01-15T10:30:00Z
[backup] Signaling WAL quiescence for space: default
[backup] Creating tier snapshots...
[backup] Calculating checksums...
[backup] Creating compressed archive...
[backup] Generating manifest...
[backup] Full backup completed in 42s
[backup] Location: /var/backups/engram/full/engram-full-default-20240115T103000Z-hostname.tar.zst
[backup] Size: 2.3 GiB
[backup] SHA256: abc123...

```

## Incremental Backup

WAL-based incremental backups for minimal RPO.

### Basic Usage

```bash
# Incremental backup for default space
/scripts/backup_incremental.sh

# Incremental for specific space
ENGRAM_SPACE_ID=production /scripts/backup_incremental.sh

```

### How It Works

1. Reads last checkpoint from `.checkpoints/{space_id}.json`

2. Finds new WAL segments since checkpoint

3. Creates compressed archive of WAL files

4. Updates checkpoint with new sequence number

5. Triggers compaction if >100 segments

### Performance

- **Throughput**: <100MB/s (fsync-bounded)

- **Duration**: <30 seconds for typical workload

- **Size**: 10-100MB per increment

### Sequence Tracking

Incremental backups track WAL sequence ranges:

```json
{
  "version": "1.0",
  "type": "incremental",
  "space_id": "default",
  "sequence_range": [1000, 2000],
  "wal_count": 50,
  "timestamp": "20240115T103045Z"
}

```

## Point-in-Time Recovery (PITR)

Restore database to exact timestamp with nanosecond precision.

### Basic Usage

```bash
# Restore to specific timestamp
/scripts/restore_pitr.sh "2024-01-15T10:30:45Z"

# Use specific base backup
/scripts/restore_pitr.sh "2024-01-15T10:30:45Z" /var/backups/engram/full/backup.tar.zst

```

### Recovery Process

1. **Find Base Backup**: Selects most recent full backup before target time

2. **Restore Base**: Extracts and restores full backup

3. **WAL Replay**: Replays WAL entries up to target timestamp

4. **Validation**: Verifies tier coherence and data integrity

### Timestamp Formats

Supported formats:

- ISO 8601: `2024-01-15T10:30:45Z`

- With nanoseconds: `2024-01-15T10:30:45.123456789Z`

- Date only: `2024-01-15` (defaults to midnight UTC)

### Recovery Precision

- **Timestamp precision**: Nanosecond-accurate

- **WAL replay**: >1000 entries/second

- **Recovery time**: <20 minutes for 1000 WAL segments

## Restore Procedures

### Full Restore

Restore complete backup to data directory.

```bash
# Basic restore
/scripts/restore.sh /var/backups/engram/full/backup.tar.zst

# Custom restore location
/scripts/restore.sh /var/backups/engram/full/backup.tar.zst /var/lib/engram

# Verify-only mode (no actual restore)
/scripts/restore.sh /var/backups/engram/full/backup.tar.zst /var/lib/engram verify-only

```

### Safety Features

The restore script automatically:

- Stops Engram service before restore

- Creates safety backup with hard links (space-efficient)

- Verifies manifest checksums before extraction

- Sets proper file permissions (700 for data, 600 for WAL)

- Validates restored directory structure

### Incremental Restore

Apply WAL changes from incremental backup:

```bash
# Apply incremental backup
/scripts/restore.sh /var/backups/engram/incremental/incr-backup.tar.zst /var/lib/engram incremental

```

## Backup Verification

Multi-level integrity checking for confidence in backups.

### Verification Levels

#### L1: Quick Manifest Check

Fastest verification (<1 second):

```bash
/scripts/verify_backup.sh /var/backups/engram/full/backup.tar.zst L1

```

Checks:

- Manifest exists and is valid JSON

- File size matches manifest

- No structural issues

#### L2: Checksum Verification

Standard verification (<30 seconds/GB):

```bash
/scripts/verify_backup.sh /var/backups/engram/full/backup.tar.zst L2

```

Checks:

- All L1 checks

- SHA256 hash matches manifest

- Archive compression integrity

#### L3: Deep Structure Validation

Thorough validation (<2 minutes/GB):

```bash
/scripts/verify_backup.sh /var/backups/engram/full/backup.tar.zst L3

```

Checks:

- All L2 checks

- Extracts and validates directory structure

- Verifies tier presence (hot/warm/cold/wal)

- Checks per-file checksums

- Validates WAL segment count

#### L4: Full Restore Test

Complete validation (<5 minutes/GB):

```bash
/scripts/verify_backup.sh /var/backups/engram/full/backup.tar.zst L4

```

Checks:

- All L3 checks

- Performs actual restore to temp directory

- Validates restored structure

- Confirms data accessibility

### Automated Verification Schedule

Recommended schedule:

- **Every backup**: L1 verification (immediate)

- **Hourly**: L2 on latest backup

- **Daily**: L3 on random selection (10% of backups)

- **Weekly**: L4 on oldest retained backup

## Retention Policies

Grandfather-Father-Son (GFS) rotation scheme.

### Default Retention

- **Daily backups**: Keep 7 days

- **Weekly backups**: Keep 4 weeks (Sunday backups)

- **Monthly backups**: Keep 12 months (1st of month backups)

### Configuration

```bash
# Set custom retention
export BACKUP_RETENTION_DAILY=14
export BACKUP_RETENTION_WEEKLY=8
export BACKUP_RETENTION_MONTHLY=24

# Run pruning (dry run first)
DRY_RUN=true /scripts/prune_backups.sh

# Apply pruning
/scripts/prune_backups.sh

```

### Safety Features

- Always keeps at least 2 full backups

- Logs all deletions for audit trail

- Maintains backup catalog in `.catalog`

- Supports dry-run mode for testing

### Manual Pruning

```bash
# Preview what would be deleted
DRY_RUN=true /scripts/prune_backups.sh

# Check backup catalog
cat /var/backups/engram/.catalog

# List all backups
ls -lh /var/backups/engram/full/
ls -lh /var/backups/engram/incremental/

```

## WAL Compaction

Prevent unbounded WAL growth through compaction.

### Automatic Compaction

WAL compaction triggers automatically:

- After 100 WAL segments accumulate

- Keeps 50 most recent segments

- Archives old segments for 1 hour (verification period)

### Manual Compaction

```bash
# Compact specific space
/scripts/wal_compact.sh default

# Compact with custom threshold
WAL_COMPACT_THRESHOLD=200 /scripts/wal_compact.sh production

```

### Compaction Process

1. Counts WAL segments in space

2. If above threshold (default 100):
   - Keeps 50 most recent segments
   - Archives older segments to `.archive-{timestamp}/`
   - Archives deleted after 1 hour verification period

3. Reports space savings

## Automated Backups

### Kubernetes CronJob

Deploy automated backups in Kubernetes:

```bash
# Deploy backup CronJob
kubectl apply -f deployments/kubernetes/backup-cronjob.yaml

# Check backup jobs
kubectl get cronjobs
kubectl get jobs -l app=engram,component=backup

# View backup logs
kubectl logs -l app=engram,component=backup --tail=100

```

Configuration:

- Runs every 4 hours (RPO 4h)

- Uses PersistentVolumeClaim for backup storage

- Resource limits: 2Gi memory, 2 CPU cores

- Mounts data directory as read-only

### Systemd Timers

Deploy automated backups with systemd:

```bash
# Install service files
sudo cp deployments/systemd/engram-backup.* /etc/systemd/system/
sudo cp deployments/systemd/engram-backup-incremental.* /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable timers
sudo systemctl enable engram-backup.timer
sudo systemctl enable engram-backup-incremental.timer

# Start timers
sudo systemctl start engram-backup.timer
sudo systemctl start engram-backup-incremental.timer

# Check status
sudo systemctl status engram-backup.timer
sudo systemctl list-timers engram-backup*

```

Timer schedules:

- **Full backup**: Every 4 hours (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)

- **Incremental backup**: Every 15 minutes

- Randomized delay: Up to 15 minutes (prevents load spikes)

## Remote Replication

### S3 Replication

Replicate backups to S3 for disaster recovery:

```bash
# Upload to S3
aws s3 cp /var/backups/engram/full/backup.tar.zst \
  s3://engram-backups/ \
  --storage-class INTELLIGENT_TIERING \
  --metadata "checksum=sha256:abc123..."

# Sync entire backup directory
aws s3 sync /var/backups/engram/ s3://engram-backups/ \
  --storage-class INTELLIGENT_TIERING \
  --exclude "tmp/*"

```

### Rsync Replication

Replicate to remote server:

```bash
# Initial sync
rsync -avz --inplace --partial \
  --bwlimit=10000 \
  --checksum \
  /var/backups/engram/ \
  remote:/backups/engram/

# Incremental sync (faster)
rsync -avz --inplace --partial \
  --bwlimit=10000 \
  /var/backups/engram/ \
  remote:/backups/engram/

```

### Replication Strategy

1. **Local NVMe** (Primary): Immediate write-through

2. **Regional Object Storage** (Secondary): <1 day lag

3. **Cross-region Archive** (Tertiary): <7 day lag

## Troubleshooting

### Backup Fails with "WAL quiescence timeout"

**Symptom**: Backup script hangs or times out waiting for WAL to quiesce.

**Solution**:

1. Check Engram is running: `systemctl status engram`

2. Verify WAL directory exists and is writable

3. Check for disk space: `df -h /var/lib/engram`

4. Review Engram logs for errors

### Restore Fails with "Checksum mismatch"

**Symptom**: Restore script reports SHA256 mismatch.

**Solution**:

1. Verify backup file is not corrupted: `zstd -t backup.tar.zst`

2. Try alternate backup if available

3. Check manifest file exists and is valid JSON

4. Re-download backup if from remote storage

### PITR Can't Find Base Backup

**Symptom**: `restore_pitr.sh` reports "No suitable base backup found"

**Solution**:

1. Verify full backup exists before target timestamp

2. Check manifest files in `/var/backups/engram/manifests/`

3. Manually specify base backup as second argument

4. Ensure backup timestamps are in correct format

### Verification Level L4 Fails

**Symptom**: Full restore test fails during L4 verification.

**Solution**:

1. Check available disk space in `/tmp`

2. Verify Engram binary is in PATH

3. Review extracted files in temp directory

4. Use L3 verification as alternative

## Best Practices

### Daily Operations

```bash
# Morning: Full backup of all spaces
ENGRAM_SPACE_ID=all /scripts/backup_full.sh

# Hourly: Incremental WAL backup
/scripts/backup_incremental.sh

# Evening: Verify latest backup
latest=/var/backups/engram/full/$(ls -t /var/backups/engram/full/ | head -1)
/scripts/verify_backup.sh "$latest" L2

# Weekly: Deep verification
random_backup=$(ls /var/backups/engram/full/*.tar.zst | shuf -n 1)
/scripts/verify_backup.sh "$random_backup" L3

# Monthly: Prune old backups
DRY_RUN=true /scripts/prune_backups.sh  # Preview first
/scripts/prune_backups.sh

```

### Testing Recovery Procedures

Regularly test restore procedures:

```bash
# Test restore to temporary location
test_dir=/tmp/engram-restore-test
/scripts/restore.sh /var/backups/engram/full/latest.tar.zst "$test_dir"

# Verify restored data
ls -la "$test_dir/spaces/"

# Cleanup
rm -rf "$test_dir"

```

### Monitoring

Monitor backup operations:

- Check backup logs: `journalctl -u engram-backup -n 100`

- Verify backup sizes are consistent

- Alert on backup failures

- Monitor backup storage usage

- Track backup duration trends

### Security

Protect backup data:

- Encrypt backups before remote storage

- Use separate credentials for backup access

- Restrict backup directory permissions (700)

- Store manifests separately for integrity verification

- Implement backup retention policies

- Test restore procedures regularly

## Related Documentation

- [Disaster Recovery](disaster-recovery.md) - DR runbooks and incident response

- [Monitoring](monitoring.md) - Backup metrics and alerting

- [Troubleshooting](troubleshooting.md) - Common backup issues

- [Production Deployment](production-deployment.md) - Production setup guide
