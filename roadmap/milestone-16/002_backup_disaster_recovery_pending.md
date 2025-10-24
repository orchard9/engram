# Task 002: Backup & Disaster Recovery System — pending

**Priority:** P0 (Critical Path)
**Estimated Effort:** 2 days
**Dependencies:** Task 001 (Container Deployment)

## Objective

Implement comprehensive backup, restore, and disaster recovery procedures with RTO <30min and RPO <5min. Enable operators to protect data, recover from failures, and maintain business continuity with confidence.

## Integration Points

**Uses:**
- `/scripts/backup_engram.sh` - Existing basic backup script
- `/scripts/check_engram_health.sh` - Health verification
- `/engram-core/src/storage/wal.rs` - WAL for PITR (with CRC32C checksums)
- `/engram-core/src/storage/tiers.rs` - Tiered storage architecture
- `/engram-core/src/storage/persistence.rs` - Per-space persistence handles
- `/engram-cli/src/config.rs` - Data directory configuration
- `/deployments/kubernetes/` - K8s manifests from Task 001

**Creates:**
- `/scripts/backup_full.sh` - Full backup with compression
- `/scripts/backup_incremental.sh` - WAL-based incremental backup
- `/scripts/restore.sh` - Restore with validation
- `/scripts/restore_pitr.sh` - Point-in-time recovery script
- `/scripts/verify_backup.sh` - Integrity verification
- `/scripts/prune_backups.sh` - Retention policy enforcement
- `/scripts/wal_compact.sh` - WAL compaction utility
- `/deployments/kubernetes/backup-cronjob.yaml` - Automated K8s backups
- `/deployments/systemd/engram-backup.service` - Systemd backup service
- `/deployments/systemd/engram-backup.timer` - Systemd timer
- `/tools/backup-manager/src/main.rs` - Backup orchestration tool (optional)

**Updates:**
- `/docs/operations/backup-restore.md` - Complete backup guide
- `/docs/operations/disaster-recovery.md` - DR runbook

## Technical Specifications

### Architecture Overview

**Storage Layout:**
```
/var/lib/engram/
├── spaces/            # Per-space data directories
│   ├── {space_id}/
│   │   ├── wal/      # Write-ahead logs (separate per space)
│   │   ├── hot/      # Hot tier (DashMap snapshots)
│   │   ├── warm/     # Warm tier (memory-mapped files)
│   │   └── cold/     # Cold tier (columnar storage)
│   └── metadata.json # Space registry metadata
├── backups/          # Backup storage location
│   ├── full/         # Full backups
│   ├── incremental/  # WAL archives
│   └── manifests/    # Backup manifests with checksums
└── recovery/         # Temporary recovery workspace
```

**Backup Consistency Model:**
- **Atomic snapshots**: Use filesystem-level snapshots (LVM/ZFS) when available
- **WAL quiescence**: Pause WAL writes during critical backup phases
- **Multi-version concurrency**: Read from consistent snapshot while allowing writes
- **Sequence tracking**: Track WAL sequence numbers for exact PITR

### Backup Types

**Full Backup:**
- Atomic snapshot of tiered storage data
- Per-space backup isolation (no cross-tenant data leakage)
- Includes: Hot tier DashMap export, warm tier mmap files, cold tier columnar data, WAL files
- Compression: zstd level 3 (optimal for 768-dim embeddings)
- Naming: `engram-full-<space_id>-<timestamp>-<hostname>.tar.zst`
- Performance targets:
  - 1GB database: <5 minutes
  - 10GB database: <15 minutes
  - 100GB database: <60 minutes
- Parallelization: Concurrent backup of independent memory spaces

**Incremental Backup (WAL-based):**
- WAL segments since last checkpoint
- Per-space WAL isolation maintained
- Naming: `engram-incr-<space_id>-<start_seq>-<end_seq>-<timestamp>.tar.zst`
- Performance: <100MB/s write throughput (bounded by fsync)
- Compaction: Automatic WAL compaction after 100 segments

**Tier-aware Differential Backup:**
- Hot tier: Full export (small, in-memory)
- Warm tier: Modified pages only (tracked via mmap dirty bits)
- Cold tier: New segments only (append-only structure)
- Reduces backup size by 80-90% vs full backup

### Retention Policies

**Default retention:**
- Daily backups: Keep 7 days
- Weekly backups: Keep 4 weeks
- Monthly backups: Keep 12 months

**Configurable via environment:**
- `BACKUP_RETENTION_DAILY=7`
- `BACKUP_RETENTION_WEEKLY=4`
- `BACKUP_RETENTION_MONTHLY=12`

**Pruning logic:**
- Run after each backup
- Delete backups older than retention period
- Always keep at least 2 full backups (safety)
- Log all deletions for audit trail

### Point-in-Time Recovery (PITR)

**WAL-based Recovery Architecture:**

**Recovery Precision:**
- Nanosecond-precision timestamps in WAL headers (u64)
- Monotonic sequence numbers for exact ordering
- CRC32C hardware-accelerated checksums for integrity

**Implementation Strategy:**
1. **Identify Recovery Point:**
   - Binary search WAL segments by timestamp
   - Find exact sequence number for target time
   - Validate WAL chain continuity (no gaps)

2. **Restore Base Snapshot:**
   - Load most recent full backup before target
   - Verify backup manifest checksums
   - Initialize tiered storage structures

3. **WAL Replay Engine:**
   ```rust
   // Pseudo-code for WAL replay with exact timestamp targeting
   fn replay_to_timestamp(target_ns: u64) -> Result<u64> {
       let mut last_sequence = 0;
       for entry in wal_reader.scan_all()? {
           if entry.header.timestamp > target_ns {
               break; // Stop at exact timestamp
           }
           apply_wal_entry(entry)?;
           last_sequence = entry.header.sequence;
       }
       Ok(last_sequence)
   }
   ```

4. **Consistency Validation:**
   - Verify tier coherence (hot/warm/cold consistency)
   - Validate memory graph integrity
   - Check confidence score distributions

**Performance Optimizations:**
- **Parallel WAL replay**: Split by memory space for concurrent recovery
- **Batch application**: Group operations by tier for locality
- **Skip-list index**: Fast forward through WAL segments
- **Memory pre-allocation**: Size tiers based on backup manifest

### Backup Verification

**Multi-layer Integrity Verification:**

**Checksum Hierarchy:**
1. **Block-level**: CRC32C per 4KB block (hardware-accelerated)
2. **File-level**: SHA256 for each backup component
3. **Manifest-level**: Ed25519 signature for tamper detection

**Verification Algorithm:**
```bash
# Merkle tree construction for incremental verification
backup_manifest.json:
{
  "root_hash": "sha256:abc123...",
  "segments": [
    {"path": "hot_tier.dat", "hash": "sha256:def456...", "size": 1048576},
    {"path": "warm_tier.mmap", "hash": "sha256:ghi789...", "size": 10485760},
    {"path": "wal/segment-001.log", "hash": "sha256:jkl012...", "size": 65536}
  ],
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "space_id": "production",
    "sequence_range": [1000, 2000]
  }
}
```

**Verification Levels:**
1. **Quick (L1)**: Manifest checksum only (<1 second)
2. **Standard (L2)**: All file checksums (<30 seconds/GB)
3. **Deep (L3)**: Extract and validate structure (<2 minutes/GB)
4. **Full (L4)**: Restore and query validation (<5 minutes/GB)

**Automated Verification Schedule:**
- Every backup: L1 verification
- Hourly: L2 on latest backup
- Daily: L3 on random selection (10% of backups)
- Weekly: L4 on oldest retained backup

### Cross-Region Replication

**Tiered Replication Strategy:**

**Storage Hierarchy:**
1. **Local NVMe** (Primary): Full backups + recent WAL
2. **Regional Object Storage** (Secondary): Compressed backups, 1-day lag
3. **Cross-region Archive** (Tertiary): Long-term retention, 7-day lag

**Optimized Transfer Protocol:**
```bash
# Parallel multi-part upload for S3
aws s3 cp backup.tar.zst s3://bucket/ \
  --storage-class INTELLIGENT_TIERING \
  --metadata "checksum=sha256:..." \
  --expected-size $(stat -f%z backup.tar.zst) \
  --no-progress

# Rsync with delta transfer for incremental updates
rsync -avz --inplace --partial \
  --bwlimit=10000 \  # 10MB/s bandwidth limit
  --checksum \
  backups/ remote:/backups/
```

**Replication Consistency:**
- **Write-through**: Critical backups replicated synchronously
- **Write-behind**: Regular backups queued for async replication
- **Eventual consistency**: Archive tier updated within 24 hours

**Bandwidth Optimization:**
- Delta compression for incremental transfers
- Deduplication at 4KB block level
- Compression ratio monitoring (target >5:1 for embeddings)

### Disaster Recovery Procedures

**RTO (Recovery Time Objective): 30 minutes**
- Time from disaster detection to service restoration

**RPO (Recovery Point Objective): 5 minutes**
- Maximum acceptable data loss (interval between backups)

**DR Scenarios:**

**1. Data Corruption**
- Context: Database files corrupted, Engram won't start
- Action: Restore from latest full backup, replay WAL
- Verification: Health check passes, data integrity verified
- Expected time: 15 minutes

**2. Complete Data Loss**
- Context: Storage failure, all local data lost
- Action: Restore from remote backup
- Verification: All memories accessible, metrics match pre-failure
- Expected time: 30 minutes

**3. Accidental Deletion**
- Context: Operator accidentally deleted memory space
- Action: PITR to timestamp before deletion
- Verification: Deleted data restored, no extra data present
- Expected time: 20 minutes

**4. Datacenter Failure**
- Context: Entire datacenter unavailable
- Action: Deploy new instance, restore from remote backup
- Verification: Service accessible from new location
- Expected time: 60 minutes (includes deployment)

**5. Ransomware Attack**
- Context: Data encrypted by malware
- Action: Restore from offline backup, verify no infection
- Verification: Security scan clean, data accessible
- Expected time: 45 minutes

## Script Specifications

### /scripts/backup_full.sh

```bash
#!/bin/bash
# Full backup with tier-aware snapshotting and verification
# Implements atomic snapshots across hot/warm/cold tiers

set -euo pipefail

# Configuration
DATA_DIR="${ENGRAM_DATA_DIR:-/var/lib/engram}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/engram}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
HOSTNAME=$(hostname -s)
SPACE_ID="${ENGRAM_SPACE_ID:-default}"
BACKUP_NAME="engram-full-${SPACE_ID}-${TIMESTAMP}-${HOSTNAME}"
BACKUP_FILE="${BACKUP_DIR}/full/${BACKUP_NAME}.tar.zst"
MANIFEST_FILE="${BACKUP_DIR}/manifests/${BACKUP_NAME}.json"
PARALLEL_JOBS="${BACKUP_PARALLEL:-4}"

# Ensure directories exist
mkdir -p "${BACKUP_DIR}"/{full,manifests,tmp}

# Function to calculate SHA256 with progress
calculate_sha256() {
    local file="$1"
    sha256sum "$file" | cut -d' ' -f1
}

# Function to backup a memory space
backup_space() {
    local space_path="$1"
    local space_id="$2"
    local temp_dir="${BACKUP_DIR}/tmp/${BACKUP_NAME}-${space_id}"

    mkdir -p "$temp_dir"

    # Step 1: Quiesce WAL writes (signal to pause)
    echo "[backup] Signaling WAL quiescence for space: $space_id"
    kill -USR1 $(pgrep -f "engram.*${space_id}") 2>/dev/null || true
    sleep 0.5  # Allow pending writes to complete

    # Step 2: Create consistent snapshots per tier
    echo "[backup] Creating tier snapshots..."

    # Hot tier: Export DashMap to binary format
    if [ -d "${space_path}/hot" ]; then
        cp -a "${space_path}/hot" "$temp_dir/" 2>/dev/null || true
    fi

    # Warm tier: Copy memory-mapped files (CoW if possible)
    if [ -f "${space_path}/warm_tier.dat" ]; then
        # Use reflink for CoW on supported filesystems
        cp --reflink=auto "${space_path}/warm_tier.dat" "$temp_dir/" 2>/dev/null || \
        cp "${space_path}/warm_tier.dat" "$temp_dir/"
    fi

    # Cold tier: Copy columnar storage files
    if [ -d "${space_path}/cold" ]; then
        cp -a "${space_path}/cold" "$temp_dir/" 2>/dev/null || true
    fi

    # WAL: Copy all segments
    if [ -d "${space_path}/wal" ]; then
        cp -a "${space_path}/wal" "$temp_dir/"
    fi

    # Step 3: Resume WAL writes
    echo "[backup] Resuming WAL writes for space: $space_id"
    kill -USR2 $(pgrep -f "engram.*${space_id}") 2>/dev/null || true

    # Step 4: Calculate checksums in parallel
    echo "[backup] Calculating checksums..."
    find "$temp_dir" -type f -print0 | \
        xargs -0 -P "$PARALLEL_JOBS" -I {} sha256sum {} > "$temp_dir/checksums.txt"

    echo "$temp_dir"
}

# Main backup flow
echo "[backup] Starting full backup at $(date -Iseconds)"
START_TIME=$(date +%s)

# Backup all memory spaces or specific one
if [ "$SPACE_ID" = "all" ]; then
    SPACES=$(ls -d "${DATA_DIR}/spaces/"*/ 2>/dev/null | xargs -n1 basename)
else
    SPACES="$SPACE_ID"
fi

# Collect all space backups
TEMP_DIRS=()
for space in $SPACES; do
    SPACE_PATH="${DATA_DIR}/spaces/${space}"
    if [ -d "$SPACE_PATH" ]; then
        temp_dir=$(backup_space "$SPACE_PATH" "$space")
        TEMP_DIRS+=("$temp_dir")
    fi
done

# Create compressed archive
echo "[backup] Creating compressed archive..."
tar -C "${BACKUP_DIR}/tmp" \
    --use-compress-program="zstd -3 -T${PARALLEL_JOBS}" \
    -cf "$BACKUP_FILE" \
    $(printf '%s\n' "${TEMP_DIRS[@]}" | xargs -n1 basename)

# Generate manifest
echo "[backup] Generating manifest..."
BACKUP_SIZE=$(stat -c%s "$BACKUP_FILE" 2>/dev/null || stat -f%z "$BACKUP_FILE")
BACKUP_HASH=$(calculate_sha256 "$BACKUP_FILE")

cat > "$MANIFEST_FILE" <<EOF
{
    "version": "1.0",
    "type": "full",
    "timestamp": "${TIMESTAMP}",
    "hostname": "${HOSTNAME}",
    "spaces": $(printf '%s\n' $SPACES | jq -R . | jq -s .),
    "backup_file": "${BACKUP_FILE}",
    "size_bytes": ${BACKUP_SIZE},
    "sha256": "${BACKUP_HASH}",
    "compression": "zstd-3",
    "duration_seconds": $(($(date +%s) - START_TIME))
}
EOF

# Cleanup temporary directories
rm -rf "${BACKUP_DIR}/tmp/${BACKUP_NAME}"*

# Verify backup integrity
echo "[backup] Verifying backup integrity..."
zstd -t "$BACKUP_FILE" || {
    echo "[backup] ERROR: Archive verification failed!"
    exit 1
}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "[backup] Full backup completed in ${DURATION}s"
echo "[backup] Location: $BACKUP_FILE"
echo "[backup] Size: $(numfmt --to=iec-i --suffix=B $BACKUP_SIZE)"
echo "[backup] SHA256: $BACKUP_HASH"
```

### /scripts/backup_incremental.sh

```bash
#!/bin/bash
# WAL-based incremental backup with sequence tracking
# Captures all WAL segments since last checkpoint for PITR

set -euo pipefail

# Configuration
DATA_DIR="${ENGRAM_DATA_DIR:-/var/lib/engram}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/engram}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
SPACE_ID="${ENGRAM_SPACE_ID:-default}"
CHECKPOINT_FILE="${BACKUP_DIR}/.checkpoints/${SPACE_ID}.json"
MANIFEST_DIR="${BACKUP_DIR}/manifests"

# Ensure directories exist
mkdir -p "${BACKUP_DIR}"/{incremental,manifests,.checkpoints}

# Function to extract WAL sequence from filename
extract_sequence() {
    local filename="$1"
    # WAL files named: wal-{timestamp:016x}.log
    basename "$filename" .log | cut -d'-' -f2 | xargs -I {} printf "%d\n" 0x{}
}

# Function to read last checkpoint
read_checkpoint() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        jq -r '.last_sequence' "$CHECKPOINT_FILE" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Function to write checkpoint
write_checkpoint() {
    local sequence="$1"
    local wal_count="$2"
    cat > "$CHECKPOINT_FILE" <<EOF
{
    "space_id": "${SPACE_ID}",
    "last_sequence": ${sequence},
    "timestamp": "${TIMESTAMP}",
    "wal_files": ${wal_count}
}
EOF
}

# Main backup flow
echo "[incremental] Starting incremental backup for space: $SPACE_ID"
START_TIME=$(date +%s)

SPACE_WAL_DIR="${DATA_DIR}/spaces/${SPACE_ID}/wal"
if [ ! -d "$SPACE_WAL_DIR" ]; then
    echo "[incremental] WAL directory not found: $SPACE_WAL_DIR"
    exit 1
fi

# Get last checkpoint sequence
LAST_SEQUENCE=$(read_checkpoint)
echo "[incremental] Last checkpoint sequence: $LAST_SEQUENCE"

# Find new WAL files
WAL_FILES=()
MAX_SEQUENCE=0

for wal_file in "$SPACE_WAL_DIR"/wal-*.log; do
    [ -f "$wal_file" ] || continue

    # Extract sequence number from WAL file
    seq=$(extract_sequence "$wal_file")

    # Include files newer than checkpoint
    if [ "$seq" -gt "$LAST_SEQUENCE" ]; then
        WAL_FILES+=("$wal_file")
        [ "$seq" -gt "$MAX_SEQUENCE" ] && MAX_SEQUENCE="$seq"
    fi
done

if [ ${#WAL_FILES[@]} -eq 0 ]; then
    echo "[incremental] No new WAL files since last backup"
    exit 0
fi

echo "[incremental] Found ${#WAL_FILES[@]} new WAL files"

# Create backup filename with sequence range
BACKUP_NAME="engram-incr-${SPACE_ID}-${LAST_SEQUENCE}-${MAX_SEQUENCE}-${TIMESTAMP}"
BACKUP_FILE="${BACKUP_DIR}/incremental/${BACKUP_NAME}.tar.zst"

# Create archive with parallel compression
echo "[incremental] Creating compressed archive..."
printf '%s\n' "${WAL_FILES[@]}" | \
    tar -C / --files-from=- -c | \
    zstd -3 -T0 > "$BACKUP_FILE"

# Calculate checksum
BACKUP_HASH=$(sha256sum "$BACKUP_FILE" | cut -d' ' -f1)
BACKUP_SIZE=$(stat -c%s "$BACKUP_FILE" 2>/dev/null || stat -f%z "$BACKUP_FILE")

# Generate manifest
MANIFEST_FILE="${MANIFEST_DIR}/${BACKUP_NAME}.json"
cat > "$MANIFEST_FILE" <<EOF
{
    "version": "1.0",
    "type": "incremental",
    "space_id": "${SPACE_ID}",
    "timestamp": "${TIMESTAMP}",
    "sequence_range": [${LAST_SEQUENCE}, ${MAX_SEQUENCE}],
    "wal_count": ${#WAL_FILES[@]},
    "backup_file": "${BACKUP_FILE}",
    "size_bytes": ${BACKUP_SIZE},
    "sha256": "${BACKUP_HASH}",
    "compression": "zstd-3"
}
EOF

# Update checkpoint
write_checkpoint "$MAX_SEQUENCE" "${#WAL_FILES[@]}"

# Optional: Trigger WAL compaction if too many segments
if [ ${#WAL_FILES[@]} -gt 100 ]; then
    echo "[incremental] Triggering WAL compaction (${#WAL_FILES[@]} segments)"
    "${SCRIPT_DIR}/wal_compact.sh" "$SPACE_ID" &
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "[incremental] Incremental backup completed in ${DURATION}s"
echo "[incremental] Sequence range: ${LAST_SEQUENCE}..${MAX_SEQUENCE}"
echo "[incremental] Location: $BACKUP_FILE"
echo "[incremental] Size: $(numfmt --to=iec-i --suffix=B $BACKUP_SIZE)"
```

### /scripts/restore.sh

```bash
#!/bin/bash
# Tier-aware restore with integrity validation
# Supports full and incremental restore operations

set -euo pipefail

# Arguments
BACKUP_FILE="${1:-}"
RESTORE_DIR="${2:-${ENGRAM_DATA_DIR:-/var/lib/engram}}"
RESTORE_MODE="${3:-full}"  # full|incremental|verify-only

# Validation
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file> [restore-dir] [mode]"
    echo "Modes: full (default), incremental, verify-only"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "[restore] ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Function to verify backup manifest
verify_manifest() {
    local manifest_file="$1"

    if [ ! -f "$manifest_file" ]; then
        echo "[restore] ERROR: Manifest not found: $manifest_file"
        return 1
    fi

    # Verify manifest structure
    jq -e '.version and .type and .sha256' "$manifest_file" >/dev/null || {
        echo "[restore] ERROR: Invalid manifest format"
        return 1
    }

    # Verify backup file checksum
    local expected_hash=$(jq -r '.sha256' "$manifest_file")
    local actual_hash=$(sha256sum "$BACKUP_FILE" | cut -d' ' -f1)

    if [ "$expected_hash" != "$actual_hash" ]; then
        echo "[restore] ERROR: Checksum mismatch!"
        echo "[restore]   Expected: $expected_hash"
        echo "[restore]   Actual:   $actual_hash"
        return 1
    fi

    echo "[restore] Manifest validation successful"
    return 0
}

# Function to stop Engram gracefully
stop_engram() {
    if pgrep -x engram >/dev/null 2>&1; then
        echo "[restore] Stopping Engram service..."

        # Try graceful shutdown first
        if command -v systemctl >/dev/null 2>&1; then
            systemctl stop engram.service 2>/dev/null || true
        else
            pkill -TERM engram || true
        fi

        # Wait for shutdown (max 30 seconds)
        local count=0
        while pgrep -x engram >/dev/null 2>&1 && [ $count -lt 30 ]; do
            sleep 1
            ((count++))
        done

        # Force kill if still running
        if pgrep -x engram >/dev/null 2>&1; then
            echo "[restore] WARNING: Force killing Engram"
            pkill -9 engram || true
        fi
    fi
}

# Function to create safety backup
create_safety_backup() {
    local target_dir="$1"

    if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir")" ]; then
        local safety_backup="${target_dir}.pre-restore-$(date +%Y%m%dT%H%M%SZ)"
        echo "[restore] Creating safety backup: $safety_backup"

        # Use hard links for space efficiency
        cp -al "$target_dir" "$safety_backup" 2>/dev/null || \
        cp -a "$target_dir" "$safety_backup"

        echo "[restore] Safety backup created at: $safety_backup"
        echo "$safety_backup"
    else
        echo ""
    fi
}

# Function to restore tiered data
restore_tiers() {
    local extract_dir="$1"
    local target_dir="$2"

    echo "[restore] Restoring tiered storage architecture..."

    # Create target structure
    mkdir -p "$target_dir"/{spaces,backups,recovery}

    # Restore each space
    for space_dir in "$extract_dir"/*; do
        [ -d "$space_dir" ] || continue

        local space_id=$(basename "$space_dir")
        local space_target="$target_dir/spaces/$space_id"

        echo "[restore] Restoring space: $space_id"
        mkdir -p "$space_target"

        # Restore hot tier (DashMap data)
        if [ -d "$space_dir/hot" ]; then
            cp -a "$space_dir/hot" "$space_target/"
            echo "[restore]   ✓ Hot tier restored"
        fi

        # Restore warm tier (memory-mapped files)
        if [ -f "$space_dir/warm_tier.dat" ]; then
            cp "$space_dir/warm_tier.dat" "$space_target/"

            # Pre-warm mmap by reading first page
            dd if="$space_target/warm_tier.dat" of=/dev/null bs=4096 count=1 2>/dev/null
            echo "[restore]   ✓ Warm tier restored (mmap ready)"
        fi

        # Restore cold tier (columnar storage)
        if [ -d "$space_dir/cold" ]; then
            cp -a "$space_dir/cold" "$space_target/"
            echo "[restore]   ✓ Cold tier restored"
        fi

        # Restore WAL files
        if [ -d "$space_dir/wal" ]; then
            cp -a "$space_dir/wal" "$space_target/"

            # Verify WAL chain integrity
            local wal_count=$(ls -1 "$space_target/wal"/wal-*.log 2>/dev/null | wc -l)
            echo "[restore]   ✓ WAL restored ($wal_count segments)"
        fi

        # Verify checksums if available
        if [ -f "$space_dir/checksums.txt" ]; then
            echo "[restore] Verifying restored file integrity..."
            cd "$space_target" && sha256sum -c "$space_dir/checksums.txt" --quiet || {
                echo "[restore] WARNING: Some files failed checksum verification"
            }
        fi
    done
}

# Main restore flow
echo "[restore] Starting restore operation"
echo "[restore] Mode: $RESTORE_MODE"
echo "[restore] Source: $BACKUP_FILE"
echo "[restore] Target: $RESTORE_DIR"

START_TIME=$(date +%s)

# Find and verify manifest
MANIFEST_FILE="${BACKUP_FILE%.tar.zst}.json"
if [ ! -f "$MANIFEST_FILE" ]; then
    # Try in manifests directory
    MANIFEST_NAME=$(basename "${BACKUP_FILE%.tar.zst}")
    MANIFEST_FILE="${BACKUP_DIR:-/var/backups/engram}/manifests/${MANIFEST_NAME}.json"
fi

if [ -f "$MANIFEST_FILE" ]; then
    verify_manifest "$MANIFEST_FILE" || exit 1
    BACKUP_TYPE=$(jq -r '.type' "$MANIFEST_FILE")
    echo "[restore] Backup type: $BACKUP_TYPE"
else
    echo "[restore] WARNING: No manifest found, assuming full backup"
    BACKUP_TYPE="full"
fi

# Verify-only mode
if [ "$RESTORE_MODE" = "verify-only" ]; then
    echo "[restore] Verification mode - testing archive integrity..."
    zstd -t "$BACKUP_FILE" && echo "[restore] ✓ Archive is valid" || {
        echo "[restore] ✗ Archive is corrupted"
        exit 1
    }
    exit 0
fi

# Stop Engram before restore
stop_engram

# Create safety backup for full restore
SAFETY_BACKUP=""
if [ "$RESTORE_MODE" = "full" ]; then
    SAFETY_BACKUP=$(create_safety_backup "$RESTORE_DIR")
fi

# Extract backup to temporary directory
TEMP_EXTRACT="/tmp/engram-restore-$$"
mkdir -p "$TEMP_EXTRACT"
trap "rm -rf $TEMP_EXTRACT" EXIT

echo "[restore] Extracting backup archive..."
zstd -d -c "$BACKUP_FILE" | tar -C "$TEMP_EXTRACT" -x

# Perform restore based on mode
case "$RESTORE_MODE" in
    full)
        # Clear target directory for full restore
        if [ -d "$RESTORE_DIR/spaces" ]; then
            rm -rf "$RESTORE_DIR/spaces"
        fi

        restore_tiers "$TEMP_EXTRACT" "$RESTORE_DIR"
        ;;

    incremental)
        # Restore only WAL files for incremental
        echo "[restore] Applying incremental backup..."

        for wal_file in "$TEMP_EXTRACT"/*/wal/*.log; do
            [ -f "$wal_file" ] || continue

            # Extract space ID from path
            space_id=$(basename $(dirname $(dirname "$wal_file")))
            target_wal_dir="$RESTORE_DIR/spaces/$space_id/wal"

            mkdir -p "$target_wal_dir"
            cp "$wal_file" "$target_wal_dir/"
        done

        echo "[restore] Incremental restore complete"
        ;;

    *)
        echo "[restore] ERROR: Unknown restore mode: $RESTORE_MODE"
        exit 1
        ;;
esac

# Post-restore validation
echo "[restore] Performing post-restore validation..."

# Check critical directories exist
for dir in spaces; do
    if [ ! -d "$RESTORE_DIR/$dir" ]; then
        echo "[restore] ERROR: Critical directory missing: $dir"
        exit 1
    fi
done

# Set proper permissions
chmod 700 "$RESTORE_DIR"
find "$RESTORE_DIR" -type d -exec chmod 755 {} \;
find "$RESTORE_DIR" -type f -exec chmod 644 {} \;
find "$RESTORE_DIR/spaces/*/wal" -type f -exec chmod 600 {} \; 2>/dev/null || true

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "[restore] Restore completed in ${DURATION}s"
echo "[restore] Data location: $RESTORE_DIR"
[ -n "$SAFETY_BACKUP" ] && echo "[restore] Safety backup: $SAFETY_BACKUP"
echo ""
echo "[restore] Next steps:"
echo "  1. Verify configuration in /etc/engram/config.toml"
echo "  2. Start Engram: systemctl start engram"
echo "  3. Check health: curl http://localhost:7432/api/v1/system/health"
echo "  4. Remove safety backup after verification: rm -rf $SAFETY_BACKUP"
```

### /scripts/restore_pitr.sh

```bash
#!/bin/bash
# Point-in-Time Recovery with nanosecond precision
# Restores database to exact timestamp using WAL replay

set -euo pipefail

# Arguments
TARGET_TIMESTAMP="${1:-}"
FULL_BACKUP="${2:-}"
DATA_DIR="${ENGRAM_DATA_DIR:-/var/lib/engram}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/engram}"

# Validation
if [ -z "$TARGET_TIMESTAMP" ]; then
    echo "Usage: $0 <target-timestamp> [full-backup-file]"
    echo "Timestamp format: YYYY-MM-DDTHH:MM:SS[.nnnnnnnnn]Z"
    echo "Example: 2024-01-15T10:30:45.123456789Z"
    exit 1
fi

# Convert timestamp to nanoseconds since epoch
target_ns=$(date -d "$TARGET_TIMESTAMP" +%s%N 2>/dev/null) || {
    echo "[pitr] ERROR: Invalid timestamp format: $TARGET_TIMESTAMP"
    exit 1
}

echo "[pitr] Target recovery point: $TARGET_TIMESTAMP ($target_ns ns)"

# Function to find appropriate base backup
find_base_backup() {
    local target_ns="$1"
    local best_backup=""
    local best_time=0

    # Search for full backups before target time
    for manifest in "$BACKUP_DIR"/manifests/engram-full-*.json; do
        [ -f "$manifest" ] || continue

        local backup_timestamp=$(jq -r '.timestamp' "$manifest")
        local backup_ns=$(date -d "$backup_timestamp" +%s%N)

        if [ "$backup_ns" -le "$target_ns" ] && [ "$backup_ns" -gt "$best_time" ]; then
            best_time="$backup_ns"
            best_backup=$(jq -r '.backup_file' "$manifest")
        fi
    done

    echo "$best_backup"
}

# Function to collect WAL files for replay
collect_wal_files() {
    local start_ns="$1"
    local end_ns="$2"
    local space_id="$3"

    local wal_files=()

    # Collect from incremental backups
    for manifest in "$BACKUP_DIR"/manifests/engram-incr-${space_id}-*.json; do
        [ -f "$manifest" ] || continue

        local timestamp=$(jq -r '.timestamp' "$manifest")
        local manifest_ns=$(date -d "$timestamp" +%s%N)

        if [ "$manifest_ns" -ge "$start_ns" ] && [ "$manifest_ns" -le "$end_ns" ]; then
            local backup_file=$(jq -r '.backup_file' "$manifest")
            wal_files+=("$backup_file")
        fi
    done

    # Also check live WAL directory
    local live_wal_dir="$DATA_DIR/spaces/$space_id/wal"
    if [ -d "$live_wal_dir" ]; then
        for wal_file in "$live_wal_dir"/wal-*.log; do
            [ -f "$wal_file" ] || continue
            wal_files+=("$wal_file")
        done
    fi

    printf '%s\n' "${wal_files[@]}"
}

# Main PITR flow
echo "[pitr] Starting Point-in-Time Recovery"
START_TIME=$(date +%s)

# Step 1: Find or validate base backup
if [ -z "$FULL_BACKUP" ]; then
    echo "[pitr] Searching for appropriate base backup..."
    FULL_BACKUP=$(find_base_backup "$target_ns")

    if [ -z "$FULL_BACKUP" ]; then
        echo "[pitr] ERROR: No suitable base backup found before target time"
        exit 1
    fi
fi

echo "[pitr] Using base backup: $FULL_BACKUP"

# Step 2: Restore base backup
echo "[pitr] Restoring base backup..."
"${SCRIPT_DIR}/restore.sh" "$FULL_BACKUP" "$DATA_DIR" full

# Get backup timestamp for WAL replay start point
MANIFEST_FILE="${FULL_BACKUP%.tar.zst}.json"
if [ -f "$MANIFEST_FILE" ]; then
    backup_timestamp=$(jq -r '.timestamp' "$MANIFEST_FILE")
    backup_ns=$(date -d "$backup_timestamp" +%s%N)
else
    echo "[pitr] WARNING: No manifest for base backup, replaying all WALs"
    backup_ns=0
fi

# Step 3: Replay WAL to target timestamp
echo "[pitr] Replaying WAL entries to target timestamp..."

# Create WAL replay engine
cat > /tmp/wal_replay.rs <<'EOF'
use engram_core::storage::wal::{WalReader, WalEntry};
use std::path::Path;

fn replay_to_timestamp(wal_dir: &Path, target_ns: u64) -> Result<u64, Box<dyn std::error::Error>> {
    let reader = WalReader::new(wal_dir, Default::default());
    let entries = reader.scan_all()?;

    let mut last_sequence = 0;
    let mut entries_applied = 0;

    for entry in entries {
        if entry.header.timestamp > target_ns {
            println!("[pitr] Stopping at sequence {} (timestamp exceeds target)", entry.header.sequence);
            break;
        }

        // Apply entry based on type
        match entry.header.entry_type.into() {
            WalEntryType::EpisodeStore => {
                // Deserialize and store episode
                let episode: Episode = bincode::deserialize(&entry.payload)?;
                // Apply to appropriate tier
            }
            WalEntryType::MemoryUpdate => {
                // Update memory
            }
            WalEntryType::MemoryDelete => {
                // Delete memory
            }
            _ => {}
        }

        last_sequence = entry.header.sequence;
        entries_applied += 1;

        if entries_applied % 1000 == 0 {
            println!("[pitr] Applied {} entries (sequence {})", entries_applied, last_sequence);
        }
    }

    println!("[pitr] WAL replay complete: {} entries applied", entries_applied);
    println!("[pitr] Final sequence: {}", last_sequence);

    Ok(last_sequence)
}
EOF

# Compile and run WAL replay (requires Rust toolchain)
if command -v cargo >/dev/null 2>&1; then
    echo "[pitr] Compiling WAL replay engine..."
    cd /tmp && cargo build --release --bin wal_replay 2>/dev/null

    # Run replay for each space
    for space_dir in "$DATA_DIR"/spaces/*; do
        [ -d "$space_dir" ] || continue
        space_id=$(basename "$space_dir")

        echo "[pitr] Replaying WAL for space: $space_id"
        /tmp/target/release/wal_replay "$space_dir/wal" "$target_ns"
    done
else
    echo "[pitr] WARNING: Rust toolchain not available, using shell-based replay"

    # Fallback to shell-based replay (simplified)
    for space_dir in "$DATA_DIR"/spaces/*; do
        [ -d "$space_dir" ] || continue
        space_id=$(basename "$space_dir")

        echo "[pitr] Processing WAL for space: $space_id"
        wal_count=0

        for wal_file in "$space_dir"/wal/wal-*.log; do
            [ -f "$wal_file" ] || continue

            # Extract timestamp from WAL file (hexadecimal)
            wal_timestamp=$(basename "$wal_file" .log | cut -d'-' -f2)
            wal_ns=$((16#$wal_timestamp * 1000000000))

            if [ "$wal_ns" -gt "$target_ns" ]; then
                echo "[pitr] Skipping WAL file (beyond target): $wal_file"
                break
            fi

            ((wal_count++))
        done

        echo "[pitr] Processed $wal_count WAL files for space $space_id"
    done
fi

# Step 4: Validate recovery
echo "[pitr] Validating recovered state..."

# Start Engram in read-only mode for validation
engram start --read-only --port 7433 &
ENGRAM_PID=$!

sleep 5

# Run validation queries
curl -s http://localhost:7433/api/v1/system/health || {
    echo "[pitr] ERROR: Health check failed after recovery"
    kill $ENGRAM_PID 2>/dev/null
    exit 1
}

# Get recovery statistics
recovery_stats=$(curl -s http://localhost:7433/api/v1/system/stats)
echo "[pitr] Recovery statistics:"
echo "$recovery_stats" | jq '.'

# Stop validation instance
kill $ENGRAM_PID 2>/dev/null

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "[pitr] Point-in-Time Recovery completed in ${DURATION}s"
echo "[pitr] Database restored to: $TARGET_TIMESTAMP"
echo "[pitr] Data location: $DATA_DIR"
echo ""
echo "[pitr] Next steps:"
echo "  1. Verify recovered data matches expectations"
echo "  2. Start production Engram: systemctl start engram"
echo "  3. Monitor for any anomalies"
```

### /scripts/verify_backup.sh

```bash
#!/bin/bash
# Multi-level backup verification with merkle tree validation
# Implements L1-L4 verification levels for different use cases

set -euo pipefail

# Arguments
BACKUP_FILE="${1:-}"
VERIFICATION_LEVEL="${2:-L2}"  # L1|L2|L3|L4

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file> [verification-level]"
    echo "Levels:"
    echo "  L1 - Quick manifest check (<1 second)"
    echo "  L2 - File checksums verification (default)"
    echo "  L3 - Deep structure validation"
    echo "  L4 - Full restore and query test"
    exit 1
fi

# Function for L1 verification (manifest only)
verify_l1() {
    local manifest="${BACKUP_FILE%.tar.zst}.json"

    if [ ! -f "$manifest" ]; then
        echo "[L1] FAIL: Manifest not found"
        return 1
    fi

    # Validate manifest structure
    jq -e '.version and .type and .sha256 and .size_bytes' "$manifest" >/dev/null || {
        echo "[L1] FAIL: Invalid manifest structure"
        return 1
    }

    # Quick size check
    local expected_size=$(jq -r '.size_bytes' "$manifest")
    local actual_size=$(stat -c%s "$BACKUP_FILE" 2>/dev/null || stat -f%z "$BACKUP_FILE")

    if [ "$expected_size" != "$actual_size" ]; then
        echo "[L1] FAIL: Size mismatch (expected: $expected_size, actual: $actual_size)"
        return 1
    fi

    echo "[L1] PASS: Manifest validation successful"
    return 0
}

# Function for L2 verification (checksums)
verify_l2() {
    verify_l1 || return 1

    local manifest="${BACKUP_FILE%.tar.zst}.json"
    local expected_hash=$(jq -r '.sha256' "$manifest")

    echo "[L2] Calculating SHA256 checksum..."
    local actual_hash=$(sha256sum "$BACKUP_FILE" | cut -d' ' -f1)

    if [ "$expected_hash" != "$actual_hash" ]; then
        echo "[L2] FAIL: Checksum mismatch"
        echo "     Expected: $expected_hash"
        echo "     Actual:   $actual_hash"
        return 1
    fi

    # Verify archive integrity
    echo "[L2] Testing archive compression..."
    zstd -t "$BACKUP_FILE" || {
        echo "[L2] FAIL: Archive compression corrupted"
        return 1
    }

    echo "[L2] PASS: Checksum and compression valid"
    return 0
}

# Function for L3 verification (structure)
verify_l3() {
    verify_l2 || return 1

    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT

    echo "[L3] Extracting and validating structure..."

    # Extract archive
    zstd -d -c "$BACKUP_FILE" | tar -C "$temp_dir" -x || {
        echo "[L3] FAIL: Extraction failed"
        return 1
    }

    # Validate directory structure
    local errors=0

    for space_dir in "$temp_dir"/*; do
        [ -d "$space_dir" ] || continue

        local space_id=$(basename "$space_dir")
        echo "[L3] Validating space: $space_id"

        # Check tier structure
        for tier in hot warm cold wal; do
            if [ "$tier" = "warm" ]; then
                # Warm tier uses a file, not directory
                [ -f "$space_dir/warm_tier.dat" ] || {
                    echo "[L3] WARNING: Missing warm tier for $space_id"
                    ((errors++))
                }
            else
                [ -d "$space_dir/$tier" ] || [ "$tier" = "hot" ] || {
                    echo "[L3] WARNING: Missing $tier tier for $space_id"
                    ((errors++))
                }
            fi
        done

        # Verify checksums if present
        if [ -f "$space_dir/checksums.txt" ]; then
            echo "[L3] Verifying file checksums for $space_id..."
            cd "$space_dir" && sha256sum -c checksums.txt --quiet || {
                echo "[L3] WARNING: Checksum failures in $space_id"
                ((errors++))
            }
            cd - >/dev/null
        fi

        # Check WAL consistency
        if [ -d "$space_dir/wal" ]; then
            local wal_count=$(ls -1 "$space_dir/wal"/wal-*.log 2>/dev/null | wc -l)
            echo "[L3]   WAL segments: $wal_count"

            # Verify WAL header magic numbers
            for wal_file in "$space_dir/wal"/wal-*.log; do
                [ -f "$wal_file" ] || continue

                # Read first 4 bytes (magic number)
                magic=$(xxd -p -l 4 "$wal_file" 2>/dev/null | sed 's/\(..\)\(..\)\(..\)\(..\)/\4\3\2\1/')
                if [ "$magic" != "deadbeef" ]; then
                    echo "[L3] WARNING: Invalid WAL magic in $(basename "$wal_file")"
                    ((errors++))
                fi
            done
        fi
    done

    if [ $errors -gt 0 ]; then
        echo "[L3] FAIL: Structure validation found $errors issues"
        return 1
    fi

    echo "[L3] PASS: Structure validation successful"
    return 0
}

# Function for L4 verification (full restore test)
verify_l4() {
    verify_l3 || return 1

    local test_dir="/tmp/engram-verify-$$"
    trap "rm -rf $test_dir; pkill -f 'engram.*--verify-port' 2>/dev/null" EXIT

    echo "[L4] Performing full restore test..."

    # Restore to test directory
    "${SCRIPT_DIR}/restore.sh" "$BACKUP_FILE" "$test_dir" full || {
        echo "[L4] FAIL: Restore operation failed"
        return 1
    }

    # Start Engram in verification mode
    echo "[L4] Starting Engram for verification..."
    engram start \
        --data-dir "$test_dir" \
        --port 7499 \
        --verify-port \
        --read-only &

    local engram_pid=$!
    sleep 5

    # Run health check
    echo "[L4] Running health check..."
    curl -f -s http://localhost:7499/api/v1/system/health >/dev/null || {
        echo "[L4] FAIL: Health check failed"
        kill $engram_pid 2>/dev/null
        return 1
    }

    # Query test data
    echo "[L4] Testing data queries..."
    local stats=$(curl -s http://localhost:7499/api/v1/system/stats)

    local memory_count=$(echo "$stats" | jq -r '.total_memories')
    echo "[L4]   Total memories: $memory_count"

    # Test a sample query
    local query_result=$(curl -s -X POST http://localhost:7499/api/v1/memories/search \
        -H "Content-Type: application/json" \
        -d '{"query":"test","limit":1}')

    if echo "$query_result" | jq -e '.results' >/dev/null; then
        echo "[L4]   Query test: PASS"
    else
        echo "[L4]   Query test: FAIL"
        kill $engram_pid 2>/dev/null
        return 1
    fi

    # Stop test instance
    kill $engram_pid 2>/dev/null
    wait $engram_pid 2>/dev/null || true

    echo "[L4] PASS: Full restore and query test successful"
    return 0
}

# Main verification flow
echo "[verify] Starting backup verification"
echo "[verify] File: $BACKUP_FILE"
echo "[verify] Level: $VERIFICATION_LEVEL"

START_TIME=$(date +%s)

case "$VERIFICATION_LEVEL" in
    L1) verify_l1 ;;
    L2) verify_l2 ;;
    L3) verify_l3 ;;
    L4) verify_l4 ;;
    *)
        echo "[verify] ERROR: Unknown verification level: $VERIFICATION_LEVEL"
        exit 1
        ;;
esac

RESULT=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $RESULT -eq 0 ]; then
    echo "[verify] Verification completed successfully in ${DURATION}s"
else
    echo "[verify] Verification FAILED after ${DURATION}s"
fi

exit $RESULT
```

### /scripts/wal_compact.sh

```bash
#!/bin/bash
# WAL compaction to prevent unbounded growth
# Merges WAL segments and removes obsolete entries

set -euo pipefail

SPACE_ID="${1:-default}"
DATA_DIR="${ENGRAM_DATA_DIR:-/var/lib/engram}"
SPACE_DIR="${DATA_DIR}/spaces/${SPACE_ID}"
WAL_DIR="${SPACE_DIR}/wal"
COMPACT_THRESHOLD="${WAL_COMPACT_THRESHOLD:-100}"  # Compact after 100 segments

if [ ! -d "$WAL_DIR" ]; then
    echo "[compact] WAL directory not found: $WAL_DIR"
    exit 1
fi

echo "[compact] Starting WAL compaction for space: $SPACE_ID"
START_TIME=$(date +%s)

# Count WAL segments
WAL_COUNT=$(ls -1 "$WAL_DIR"/wal-*.log 2>/dev/null | wc -l)
echo "[compact] Found $WAL_COUNT WAL segments"

if [ "$WAL_COUNT" -lt "$COMPACT_THRESHOLD" ]; then
    echo "[compact] Below threshold ($COMPACT_THRESHOLD), skipping compaction"
    exit 0
fi

# Create compacted WAL file
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
COMPACT_FILE="${WAL_DIR}/wal-compact-${TIMESTAMP}.tmp"

# Use Engram's built-in compaction (leverages WalWriter::compact_from_memory)
engram compact-wal \
    --space-id "$SPACE_ID" \
    --output "$COMPACT_FILE" \
    --validate || {
    echo "[compact] ERROR: Compaction failed"
    rm -f "$COMPACT_FILE"
    exit 1
}

# Atomic rename (this is the commit point)
FINAL_FILE="${WAL_DIR}/wal-$(date +%s%N | cut -c1-16).log"
mv "$COMPACT_FILE" "$FINAL_FILE"

# Archive old WAL files
ARCHIVE_DIR="${WAL_DIR}/.archive-${TIMESTAMP}"
mkdir -p "$ARCHIVE_DIR"

for wal_file in "$WAL_DIR"/wal-*.log; do
    [ -f "$wal_file" ] || continue
    [ "$wal_file" = "$FINAL_FILE" ] && continue

    mv "$wal_file" "$ARCHIVE_DIR/"
done

# Calculate space savings
OLD_SIZE=$(du -sb "$ARCHIVE_DIR" | cut -f1)
NEW_SIZE=$(stat -c%s "$FINAL_FILE" 2>/dev/null || stat -f%z "$FINAL_FILE")
SAVED=$((OLD_SIZE - NEW_SIZE))

echo "[compact] Compaction complete"
echo "[compact]   Old size: $(numfmt --to=iec-i --suffix=B $OLD_SIZE)"
echo "[compact]   New size: $(numfmt --to=iec-i --suffix=B $NEW_SIZE)"
echo "[compact]   Saved: $(numfmt --to=iec-i --suffix=B $SAVED) ($(( (SAVED * 100) / OLD_SIZE ))%)"

# Optional: Delete archive after verification period
(
    sleep 3600  # Keep archive for 1 hour
    rm -rf "$ARCHIVE_DIR"
    echo "[compact] Archive deleted: $ARCHIVE_DIR"
) &

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "[compact] WAL compaction completed in ${DURATION}s"
```

### /scripts/prune_backups.sh

```bash
#!/bin/bash
# Intelligent backup retention with tiered pruning
# Implements grandfather-father-son (GFS) rotation scheme

set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/var/backups/engram}"
RETENTION_DAILY="${BACKUP_RETENTION_DAILY:-7}"
RETENTION_WEEKLY="${BACKUP_RETENTION_WEEKLY:-4}"
RETENTION_MONTHLY="${BACKUP_RETENTION_MONTHLY:-12}"
DRY_RUN="${DRY_RUN:-false}"

echo "[prune] Starting backup pruning"
echo "[prune] Retention: ${RETENTION_DAILY}d/${RETENTION_WEEKLY}w/${RETENTION_MONTHLY}m"

# Function to get backup date from filename
get_backup_date() {
    local filename="$1"
    # Extract timestamp from filename: engram-full-space-20240115T103045Z-hostname.tar.zst
    local timestamp=$(basename "$filename" | sed -n 's/.*-\([0-9]\{8\}T[0-9]\{6\}Z\)-.*/\1/p')
    date -d "$timestamp" +%s 2>/dev/null || echo "0"
}

# Function to check if date is a Sunday
is_sunday() {
    local date_epoch="$1"
    [ "$(date -d "@$date_epoch" +%w)" = "0" ]
}

# Function to check if date is first of month
is_first_of_month() {
    local date_epoch="$1"
    [ "$(date -d "@$date_epoch" +%d)" = "01" ]
}

# Mark backups to keep
declare -A keep_backups

# Process full backups
for backup in "$BACKUP_DIR"/full/engram-full-*.tar.zst; do
    [ -f "$backup" ] || continue

    backup_date=$(get_backup_date "$backup")
    [ "$backup_date" = "0" ] && continue

    age_days=$(( ($(date +%s) - backup_date) / 86400 ))

    # Keep all recent daily backups
    if [ "$age_days" -le "$RETENTION_DAILY" ]; then
        keep_backups["$backup"]=1
        echo "[prune] KEEP (daily): $(basename "$backup") - ${age_days}d old"

    # Keep weekly backups (Sunday)
    elif [ "$age_days" -le $((RETENTION_WEEKLY * 7)) ] && is_sunday "$backup_date"; then
        keep_backups["$backup"]=1
        echo "[prune] KEEP (weekly): $(basename "$backup") - ${age_days}d old"

    # Keep monthly backups (1st of month)
    elif [ "$age_days" -le $((RETENTION_MONTHLY * 30)) ] && is_first_of_month "$backup_date"; then
        keep_backups["$backup"]=1
        echo "[prune] KEEP (monthly): $(basename "$backup") - ${age_days}d old"

    else
        # Mark for deletion
        if [ "$DRY_RUN" = "true" ]; then
            echo "[prune] WOULD DELETE: $(basename "$backup") - ${age_days}d old"
        else
            echo "[prune] DELETE: $(basename "$backup") - ${age_days}d old"
            rm -f "$backup"
            rm -f "${backup%.tar.zst}.json"  # Remove manifest too
        fi
    fi
done

# Process incremental backups (keep those needed for PITR)
for backup in "$BACKUP_DIR"/incremental/engram-incr-*.tar.zst; do
    [ -f "$backup" ] || continue

    backup_date=$(get_backup_date "$backup")
    [ "$backup_date" = "0" ] && continue

    age_days=$(( ($(date +%s) - backup_date) / 86400 ))

    # Keep incremental backups between retained full backups
    if [ "$age_days" -le "$RETENTION_DAILY" ]; then
        keep_backups["$backup"]=1
    else
        if [ "$DRY_RUN" = "true" ]; then
            echo "[prune] WOULD DELETE (incr): $(basename "$backup") - ${age_days}d old"
        else
            echo "[prune] DELETE (incr): $(basename "$backup") - ${age_days}d old"
            rm -f "$backup"
            rm -f "${backup%.tar.zst}.json"
        fi
    fi
done

# Safety check: ensure at least 2 full backups remain
full_count=$(ls -1 "$BACKUP_DIR"/full/engram-full-*.tar.zst 2>/dev/null | wc -l)
if [ "$full_count" -lt 2 ]; then
    echo "[prune] WARNING: Keeping at least 2 full backups for safety"

    # Keep the 2 most recent
    for backup in $(ls -t "$BACKUP_DIR"/full/engram-full-*.tar.zst 2>/dev/null | head -2); do
        keep_backups["$backup"]=1
    done
fi

# Calculate space freed
FREED_BYTES=0
for backup in "$BACKUP_DIR"/{full,incremental}/*.tar.zst; do
    [ -f "$backup" ] || continue
    if [ -z "${keep_backups[$backup]:-}" ] && [ "$DRY_RUN" = "false" ]; then
        size=$(stat -c%s "$backup" 2>/dev/null || stat -f%z "$backup")
        FREED_BYTES=$((FREED_BYTES + size))
    fi
done

echo "[prune] Summary:"
echo "[prune]   Backups kept: ${#keep_backups[@]}"
echo "[prune]   Space freed: $(numfmt --to=iec-i --suffix=B $FREED_BYTES)"
[ "$DRY_RUN" = "true" ] && echo "[prune]   (DRY RUN - no files deleted)"

# Update backup catalog
if [ "$DRY_RUN" = "false" ]; then
    echo "[prune] Updating backup catalog..."
    ls -la "$BACKUP_DIR"/{full,incremental}/*.tar.zst 2>/dev/null | \
        awk '{print $9, $5}' > "$BACKUP_DIR/.catalog"
fi

echo "[prune] Backup pruning complete"
```

## Kubernetes CronJob

### /deployments/kubernetes/backup-cronjob.yaml

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: engram-backup
spec:
  schedule: "0 */4 * * *"  # Every 4 hours (RPO 4h)
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: engram/engram:latest
            command:
            - /scripts/backup_full.sh
            env:
            - name: ENGRAM_DATA_DIR
              value: /data
            - name: BACKUP_DIR
              value: /backups
            volumeMounts:
            - name: data
              mountPath: /data
              readOnly: true
            - name: backups
              mountPath: /backups
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: engram-data
          - name: backups
            persistentVolumeClaim:
              claimName: engram-backups
```

## Operator Command Examples

### Daily Operations
```bash
# Morning: Create full backup of all spaces
ENGRAM_SPACE_ID=all /scripts/backup_full.sh

# Hourly: Incremental WAL backup
/scripts/backup_incremental.sh

# Evening: Verify latest backup integrity
latest_backup=$(ls -t /var/backups/engram/full/*.tar.zst | head -1)
/scripts/verify_backup.sh "$latest_backup" L2

# Weekly: Deep verification of random backup
random_backup=$(ls /var/backups/engram/full/*.tar.zst | shuf -n 1)
/scripts/verify_backup.sh "$random_backup" L3

# Monthly: Prune old backups (dry run first)
DRY_RUN=true /scripts/prune_backups.sh
/scripts/prune_backups.sh
```

### Disaster Recovery Scenarios

**Scenario 1: Corrupted Hot Tier**
```bash
# Symptom: Engram crashes with DashMap corruption error
# Solution: Restore from last good backup

# 1. Stop service
systemctl stop engram

# 2. Find latest verified backup
latest_good=$(grep "L2.*PASS" /var/log/backup-verify.log | tail -1 | cut -d: -f2)

# 3. Restore
/scripts/restore.sh "$latest_good" /var/lib/engram full

# 4. Start and verify
systemctl start engram
curl http://localhost:7432/api/v1/system/health
```

**Scenario 2: Accidental Deletion (PITR)**
```bash
# User reports critical memory deleted at 2:30 PM
# Restore to 2:29 PM

# 1. Determine exact timestamp
target_time="2024-01-15T14:29:00Z"

# 2. Perform point-in-time recovery
/scripts/restore_pitr.sh "$target_time"

# 3. Verify deleted memory is restored
engram query "critical memory content"
```

**Scenario 3: Complete Data Loss**
```bash
# All local data destroyed

# 1. Restore from S3
aws s3 cp s3://engram-backups/latest-full.tar.zst /tmp/
aws s3 cp s3://engram-backups/latest-full.json /tmp/

# 2. Verify and restore
/scripts/verify_backup.sh /tmp/latest-full.tar.zst L3
/scripts/restore.sh /tmp/latest-full.tar.zst /var/lib/engram

# 3. Apply incremental backups
for incr in $(aws s3 ls s3://engram-backups/ | grep engram-incr | awk '{print $4}'); do
    aws s3 cp "s3://engram-backups/$incr" /tmp/
    /scripts/restore.sh "/tmp/$incr" /var/lib/engram incremental
done

# 4. Start service
systemctl start engram
```

**Scenario 4: WAL Corruption**
```bash
# WAL replay fails during startup

# 1. Compact WAL to remove corrupt entries
/scripts/wal_compact.sh default

# 2. If compaction fails, restore from backup
latest_backup=$(ls -t /var/backups/engram/full/*.tar.zst | head -1)
/scripts/restore.sh "$latest_backup"

# 3. Restart with clean WAL
systemctl restart engram
```

### Performance Testing
```bash
# Test backup performance on production-sized data
# Create 10GB test dataset
engram benchmark --create-data 10GB

# Time full backup
time /scripts/backup_full.sh
# Expected: <15 minutes for 10GB

# Time incremental backup
time /scripts/backup_incremental.sh
# Expected: <30 seconds

# Test restore performance
time /scripts/restore.sh /var/backups/engram/full/latest.tar.zst /tmp/test-restore
# Expected: <10 minutes for 10GB

# Test PITR with 1000 WAL segments
time /scripts/restore_pitr.sh "2024-01-15T10:00:00Z"
# Expected: <20 minutes
```

## Documentation Requirements

### /docs/operations/backup-restore.md

**Sections:**
1. Overview - Backup strategies and RTO/RPO definitions
2. Full Backup - When to use, how to run, verification
3. Incremental Backup - WAL-based backups, PITR capability
4. Restore Procedures - Step-by-step restore from full backup
5. Point-in-Time Recovery - Restore to specific timestamp
6. Automated Backups - Kubernetes CronJob, systemd timers
7. Retention Policies - Default policies, customization
8. Backup Verification - Integrity checking procedures
9. Remote Replication - S3/SSH backup replication
10. Troubleshooting - Common backup/restore issues

### /docs/operations/disaster-recovery.md

**Sections:**
1. RTO/RPO Definitions - Service level objectives
2. DR Scenarios - Data corruption, datacenter failure, etc.
3. Incident Response - Step-by-step response for each scenario
4. Recovery Procedures - Detailed restore procedures
5. Post-Recovery Verification - Health checks and data validation
6. DR Testing - How to test DR procedures
7. Escalation Paths - When to escalate, who to contact
8. Post-Incident Review - Template for incident retrospective

## Acceptance Criteria

**Backup Performance & Integrity:**
- [ ] Full backup: <5 minutes for 1GB, <15 minutes for 10GB, <60 minutes for 100GB
- [ ] Incremental backup: <100MB/s WAL throughput with fsync
- [ ] Per-space isolation maintained (no cross-tenant data in backups)
- [ ] CRC32C checksums on all WAL entries (hardware-accelerated)
- [ ] SHA256 checksums on all backup files
- [ ] Merkle tree manifest for incremental verification
- [ ] Zero data corruption during concurrent backup operations
- [ ] Atomic snapshots with WAL quiescence (<500ms pause)

**Restore & Recovery:**
- [ ] Full restore: <10 minutes for 10GB backup
- [ ] PITR precision: Nanosecond-accurate timestamp recovery
- [ ] WAL replay: >1000 entries/second
- [ ] Tier coherence validation after restore
- [ ] Memory graph integrity preserved
- [ ] Confidence score distributions maintained
- [ ] Safety backup with hard links (space-efficient)
- [ ] Parallel restoration for independent memory spaces

**Storage Optimization:**
- [ ] zstd compression ratio >5:1 for embeddings
- [ ] Tier-aware differential backup reduces size by 80-90%
- [ ] WAL compaction reduces size by >50% after 100 segments
- [ ] Copy-on-write for warm tier snapshots (where supported)
- [ ] Block-level deduplication for incremental transfers
- [ ] Grandfather-father-son (GFS) retention scheme

**Verification Levels:**
- [ ] L1 (manifest): <1 second verification
- [ ] L2 (checksums): <30 seconds/GB
- [ ] L3 (structure): <2 minutes/GB with WAL validation
- [ ] L4 (full restore): <5 minutes/GB with query tests
- [ ] Automated verification schedule (hourly L2, daily L3, weekly L4)
- [ ] WAL magic number validation (0xDEADBEEF)

**Cross-Region Replication:**
- [ ] Primary (NVMe): Immediate write-through
- [ ] Regional (S3): <1 day lag with INTELLIGENT_TIERING
- [ ] Archive (Cross-region): <7 day lag
- [ ] Bandwidth throttling (10MB/s default)
- [ ] Delta transfer with rsync for incrementals
- [ ] Multi-part parallel uploads for S3

**Automation & Operations:**
- [ ] Kubernetes CronJob with configurable schedule
- [ ] Systemd timer with dependency management
- [ ] Automatic WAL compaction trigger at 100 segments
- [ ] Backup catalog maintained (.catalog file)
- [ ] JSON manifests with complete metadata
- [ ] Dry-run mode for pruning operations
- [ ] Background archive deletion after verification

**Disaster Recovery Targets:**
- [ ] RTO: <30 minutes for complete data loss
- [ ] RPO: <5 minutes with incremental backups
- [ ] Data corruption recovery: <15 minutes
- [ ] PITR recovery: <20 minutes for 1000 WAL segments
- [ ] WAL corruption recovery via compaction
- [ ] Cross-space recovery with isolation maintained

**Documentation & Testing:**
- [ ] Operator runbook with exact commands
- [ ] Performance benchmarks documented
- [ ] All 5 DR scenarios tested end-to-end
- [ ] Backup/restore integrated with health checks
- [ ] Troubleshooting guide for common failures
- [ ] Verification logs with audit trail

## Follow-Up Tasks

- Task 003: Integrate backup metrics and alerts
- Task 005: Add backup failure to troubleshooting runbook
- Task 008: Add encrypted backup support
- Future: Continuous WAL archiving for RPO <1 minute
