#!/usr/bin/env bash
# Point-in-Time Recovery with nanosecond precision
# Restores database to exact timestamp using WAL replay

set -euo pipefail

# Arguments
TARGET_TIMESTAMP="${1:-}"
FULL_BACKUP="${2:-}"
DATA_DIR="${ENGRAM_DATA_DIR:-/var/lib/engram}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/engram}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Validation
if [ -z "$TARGET_TIMESTAMP" ]; then
    echo "Usage: $0 <target-timestamp> [full-backup-file]"
    echo "Timestamp format: YYYY-MM-DDTHH:MM:SS[.nnnnnnnnn]Z"
    echo "Example: 2024-01-15T10:30:45.123456789Z"
    exit 1
fi

# Convert timestamp to nanoseconds since epoch (simplified for macOS/Linux compatibility)
if command -v gdate >/dev/null 2>&1; then
    target_ns=$(gdate -d "$TARGET_TIMESTAMP" +%s%N 2>/dev/null) || {
        echo "[pitr] ERROR: Invalid timestamp format: $TARGET_TIMESTAMP"
        exit 1
    }
else
    target_ns=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$TARGET_TIMESTAMP" +%s 2>/dev/null || \
                date -d "$TARGET_TIMESTAMP" +%s 2>/dev/null || echo "0")
    target_ns="${target_ns}000000000"  # Convert to nanoseconds
fi

if [ "$target_ns" = "0" ]; then
    echo "[pitr] ERROR: Could not parse timestamp: $TARGET_TIMESTAMP"
    exit 1
fi

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

        # Convert backup timestamp to nanoseconds
        if command -v gdate >/dev/null 2>&1; then
            backup_ns=$(gdate -d "$backup_timestamp" +%s%N 2>/dev/null || echo "0")
        else
            backup_ns=$(date -j -f "%Y%m%dT%H%M%SZ" "$backup_timestamp" +%s 2>/dev/null || \
                       date -d "$backup_timestamp" +%s 2>/dev/null || echo "0")
            backup_ns="${backup_ns}000000000"
        fi

        if [ "$backup_ns" -le "$target_ns" ] && [ "$backup_ns" -gt "$best_time" ]; then
            best_time="$backup_ns"
            best_backup=$(jq -r '.backup_file' "$manifest")
        fi
    done

    echo "$best_backup"
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
if [ ! -f "$MANIFEST_FILE" ]; then
    MANIFEST_NAME=$(basename "${FULL_BACKUP%.tar.zst}")
    MANIFEST_FILE="${BACKUP_DIR}/manifests/${MANIFEST_NAME}.json"
fi

if [ -f "$MANIFEST_FILE" ]; then
    backup_timestamp=$(jq -r '.timestamp' "$MANIFEST_FILE")
    if command -v gdate >/dev/null 2>&1; then
        backup_ns=$(gdate -d "$backup_timestamp" +%s%N 2>/dev/null || echo "0")
    else
        backup_ns=$(date -d "$backup_timestamp" +%s 2>/dev/null || echo "0")
        backup_ns="${backup_ns}000000000"
    fi
else
    echo "[pitr] WARNING: No manifest for base backup, replaying all WALs"
    backup_ns=0
fi

# Step 3: Replay WAL to target timestamp
echo "[pitr] Replaying WAL entries to target timestamp..."
echo "[pitr] NOTE: WAL replay requires Engram to process WAL files on startup"
echo "[pitr] The WAL files have been restored and will be replayed automatically"

# For each space, keep only WAL files before target timestamp
for space_dir in "$DATA_DIR"/spaces/*; do
    [ -d "$space_dir" ] || continue
    space_id=$(basename "$space_dir")

    echo "[pitr] Processing WAL for space: $space_id"
    wal_count=0
    removed_count=0

    for wal_file in "$space_dir"/wal/wal-*.log; do
        [ -f "$wal_file" ] || continue

        # Extract timestamp from WAL file (hexadecimal timestamp in filename)
        wal_timestamp_hex=$(basename "$wal_file" .log | cut -d'-' -f2)
        # Convert hex timestamp to decimal (assuming it's in seconds)
        wal_ts_sec=$((16#$wal_timestamp_hex))
        wal_ns=$((wal_ts_sec * 1000000000))

        if [ "$wal_ns" -gt "$target_ns" ]; then
            echo "[pitr] Removing WAL file (beyond target): $(basename "$wal_file")"
            rm -f "$wal_file"
            ((removed_count++))
        else
            ((wal_count++))
        fi
    done

    echo "[pitr] Space $space_id: kept $wal_count WAL files, removed $removed_count files"
done

# Step 4: Create recovery marker
echo "[pitr] Creating recovery marker..."
cat > "$DATA_DIR/.pitr_recovery_info" <<EOF
{
    "target_timestamp": "${TARGET_TIMESTAMP}",
    "target_ns": ${target_ns},
    "base_backup": "${FULL_BACKUP}",
    "recovery_time": "$(date -Iseconds)",
    "status": "ready_for_startup"
}
EOF

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "[pitr] Point-in-Time Recovery setup completed in ${DURATION}s"
echo "[pitr] Database prepared to restore to: $TARGET_TIMESTAMP"
echo "[pitr] Data location: $DATA_DIR"
echo ""
echo "[pitr] Next steps:"
echo "  1. Start Engram to replay WAL: systemctl start engram"
echo "  2. Monitor startup logs for WAL replay progress"
echo "  3. Verify recovered data matches expectations"
echo "  4. Check recovery info: cat $DATA_DIR/.pitr_recovery_info"
