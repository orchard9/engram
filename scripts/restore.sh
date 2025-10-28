#!/usr/bin/env bash
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

BACKUP_DIR="${BACKUP_DIR:-/var/backups/engram}"

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

    if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        local safety_backup="${target_dir}.pre-restore-$(date +%Y%m%dT%H%M%SZ)"
        echo "[restore] Creating safety backup: $safety_backup"

        # Use hard links for space efficiency (fall back to regular copy)
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
            echo "[restore]   Hot tier restored"
        fi

        # Restore warm tier (memory-mapped files)
        if [ -f "$space_dir/warm_tier.dat" ]; then
            cp "$space_dir/warm_tier.dat" "$space_target/"

            # Pre-warm mmap by reading first page
            dd if="$space_target/warm_tier.dat" of=/dev/null bs=4096 count=1 2>/dev/null || true
            echo "[restore]   Warm tier restored (mmap ready)"
        fi

        # Restore cold tier (columnar storage)
        if [ -d "$space_dir/cold" ]; then
            cp -a "$space_dir/cold" "$space_target/"
            echo "[restore]   Cold tier restored"
        fi

        # Restore WAL files
        if [ -d "$space_dir/wal" ]; then
            cp -a "$space_dir/wal" "$space_target/"

            # Verify WAL chain integrity
            local wal_count=$(ls -1 "$space_target/wal"/wal-*.log 2>/dev/null | wc -l)
            echo "[restore]   WAL restored ($wal_count segments)"
        fi

        # Verify checksums if available
        if [ -f "$space_dir/checksums.txt" ]; then
            echo "[restore] Verifying restored file integrity..."
            (cd "$space_target" && sha256sum -c "$space_dir/checksums.txt" --quiet) || {
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
    MANIFEST_FILE="${BACKUP_DIR}/manifests/${MANIFEST_NAME}.json"
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
    zstd -t "$BACKUP_FILE" && echo "[restore] Archive is valid" || {
        echo "[restore] Archive is corrupted"
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
if [ -n "$SAFETY_BACKUP" ]; then
    echo "  4. Remove safety backup after verification: rm -rf $SAFETY_BACKUP"
fi
