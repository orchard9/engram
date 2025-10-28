#!/usr/bin/env bash
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure directories exist
mkdir -p "${BACKUP_DIR}"/{incremental,manifests,.checkpoints}

# Function to extract WAL sequence from filename
extract_sequence() {
    local filename="$1"
    # WAL files named: wal-{timestamp:016x}.log
    basename "$filename" .log | cut -d'-' -f2 | xargs printf "%d\n" 2>/dev/null || echo "0"
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
tar -cf - -C / "${WAL_FILES[@]}" | zstd -3 -T0 > "$BACKUP_FILE"

# Calculate checksum
BACKUP_HASH=$(sha256sum "$BACKUP_FILE" | cut -d' ' -f1)
BACKUP_SIZE=$(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE")

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
if [ ${#WAL_FILES[@]} -gt 100 ] && [ -x "${SCRIPT_DIR}/wal_compact.sh" ]; then
    echo "[incremental] Triggering WAL compaction (${#WAL_FILES[@]} segments)"
    "${SCRIPT_DIR}/wal_compact.sh" "$SPACE_ID" &
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "[incremental] Incremental backup completed in ${DURATION}s"
echo "[incremental] Sequence range: ${LAST_SEQUENCE}..${MAX_SEQUENCE}"
echo "[incremental] Location: $BACKUP_FILE"
echo "[incremental] Size: $(numfmt --to=iec-i --suffix=B $BACKUP_SIZE 2>/dev/null || echo "${BACKUP_SIZE} bytes")"
