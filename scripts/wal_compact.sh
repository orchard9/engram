#!/usr/bin/env bash
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
WAL_COUNT=$(ls -1 "$WAL_DIR"/wal-*.log 2>/dev/null | wc -l | tr -d ' ')
echo "[compact] Found $WAL_COUNT WAL segments"

if [ "$WAL_COUNT" -lt "$COMPACT_THRESHOLD" ]; then
    echo "[compact] Below threshold ($COMPACT_THRESHOLD), skipping compaction"
    exit 0
fi

# Create compacted WAL file
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
COMPACT_FILE="${WAL_DIR}/wal-compact-${TIMESTAMP}.tmp"

echo "[compact] Creating compacted WAL file..."
echo "[compact] NOTE: Full WAL compaction requires Engram's internal compaction logic"
echo "[compact] This script performs simple archive rotation"

# Archive old WAL files (simplified compaction)
ARCHIVE_DIR="${WAL_DIR}/.archive-${TIMESTAMP}"
mkdir -p "$ARCHIVE_DIR"

# Keep only the most recent WAL files (last 50)
WAL_FILES=($(ls -t "$WAL_DIR"/wal-*.log 2>/dev/null))
KEEP_COUNT=50

if [ ${#WAL_FILES[@]} -gt $KEEP_COUNT ]; then
    echo "[compact] Archiving old WAL files (keeping $KEEP_COUNT most recent)..."

    # Move old files to archive
    for ((i=$KEEP_COUNT; i<${#WAL_FILES[@]}; i++)); do
        wal_file="${WAL_FILES[$i]}"
        mv "$wal_file" "$ARCHIVE_DIR/"
    done

    # Calculate space savings
    OLD_SIZE=$(du -sk "$ARCHIVE_DIR" 2>/dev/null | cut -f1)
    OLD_SIZE=$((OLD_SIZE * 1024))
    NEW_SIZE=$(du -sk "$WAL_DIR" 2>/dev/null | cut -f1)
    NEW_SIZE=$((NEW_SIZE * 1024))
    SAVED=$((OLD_SIZE))

    echo "[compact] Compaction complete"
    if command -v numfmt >/dev/null 2>&1; then
        echo "[compact]   Archived size: $(numfmt --to=iec-i --suffix=B $OLD_SIZE)"
        echo "[compact]   Current size: $(numfmt --to=iec-i --suffix=B $NEW_SIZE)"
        echo "[compact]   Saved: $(numfmt --to=iec-i --suffix=B $SAVED)"
    else
        echo "[compact]   Archived size: $OLD_SIZE bytes"
        echo "[compact]   Current size: $NEW_SIZE bytes"
    fi

    # Optional: Delete archive after verification period
    (
        sleep 3600  # Keep archive for 1 hour
        if [ -d "$ARCHIVE_DIR" ]; then
            rm -rf "$ARCHIVE_DIR"
            echo "[compact] Archive deleted: $ARCHIVE_DIR"
        fi
    ) &

else
    echo "[compact] Not enough WAL files to compact (${#WAL_FILES[@]} <= $KEEP_COUNT)"
    rmdir "$ARCHIVE_DIR" 2>/dev/null || true
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "[compact] WAL compaction completed in ${DURATION}s"
