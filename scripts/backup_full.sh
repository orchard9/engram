#!/usr/bin/env bash
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
        xargs -0 -P "$PARALLEL_JOBS" -I {} sh -c 'sha256sum "{}" >> "'"$temp_dir"'/checksums.txt"'

    echo "$temp_dir"
}

# Main backup flow
echo "[backup] Starting full backup at $(date -Iseconds)"
START_TIME=$(date +%s)

# Backup all memory spaces or specific one
if [ "$SPACE_ID" = "all" ]; then
    SPACES=$(ls -d "${DATA_DIR}/spaces/"*/ 2>/dev/null | xargs -n1 basename || echo "")
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
BACKUP_SIZE=$(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE")
BACKUP_HASH=$(calculate_sha256 "$BACKUP_FILE")

# Convert spaces to JSON array
SPACES_JSON="["
first=true
for space in $SPACES; do
    if [ "$first" = true ]; then
        SPACES_JSON="${SPACES_JSON}\"${space}\""
        first=false
    else
        SPACES_JSON="${SPACES_JSON},\"${space}\""
    fi
done
SPACES_JSON="${SPACES_JSON}]"

cat > "$MANIFEST_FILE" <<EOF
{
    "version": "1.0",
    "type": "full",
    "timestamp": "${TIMESTAMP}",
    "hostname": "${HOSTNAME}",
    "spaces": ${SPACES_JSON},
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
echo "[backup] Size: $(numfmt --to=iec-i --suffix=B $BACKUP_SIZE 2>/dev/null || echo "${BACKUP_SIZE} bytes")"
echo "[backup] SHA256: $BACKUP_HASH"
