#!/usr/bin/env bash
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

# Ensure backup directories exist
mkdir -p "$BACKUP_DIR"/{full,incremental,manifests}

# Function to get backup date from filename
get_backup_date() {
    local filename="$1"
    # Extract timestamp from filename: engram-full-space-20240115T103045Z-hostname.tar.zst
    local timestamp=$(basename "$filename" | grep -oE '[0-9]{8}T[0-9]{6}Z' | head -1)
    if [ -z "$timestamp" ]; then
        echo "0"
        return
    fi

    # Convert to epoch - macOS compatible
    if command -v gdate >/dev/null 2>&1; then
        gdate -d "$timestamp" +%s 2>/dev/null || echo "0"
    else
        date -j -f "%Y%m%dT%H%M%SZ" "$timestamp" +%s 2>/dev/null || echo "0"
    fi
}

# Function to check if date is a Sunday
is_sunday() {
    local date_epoch="$1"
    local day_of_week=$(date -r "$date_epoch" +%w 2>/dev/null || date -d "@$date_epoch" +%w 2>/dev/null)
    [ "$day_of_week" = "0" ]
}

# Function to check if date is first of month
is_first_of_month() {
    local date_epoch="$1"
    local day_of_month=$(date -r "$date_epoch" +%d 2>/dev/null || date -d "@$date_epoch" +%d 2>/dev/null)
    [ "$day_of_month" = "01" ]
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
full_count=$(ls -1 "$BACKUP_DIR"/full/engram-full-*.tar.zst 2>/dev/null | wc -l | tr -d ' ')
if [ "$full_count" -lt 2 ]; then
    echo "[prune] WARNING: Keeping at least 2 full backups for safety"

    # Keep the 2 most recent
    for backup in $(ls -t "$BACKUP_DIR"/full/engram-full-*.tar.zst 2>/dev/null | head -2); do
        keep_backups["$backup"]=1
    done
fi

# Calculate space freed
FREED_BYTES=0
for dir in full incremental; do
    if [ -d "$BACKUP_DIR/$dir" ]; then
        for backup in "$BACKUP_DIR/$dir"/*.tar.zst; do
            [ -f "$backup" ] || continue
            if [ -z "${keep_backups[$backup]:-}" ] && [ "$DRY_RUN" = "false" ]; then
                size=$(stat -f%z "$backup" 2>/dev/null || stat -c%s "$backup" 2>/dev/null || echo "0")
                FREED_BYTES=$((FREED_BYTES + size))
            fi
        done
    fi
done

echo "[prune] Summary:"
echo "[prune]   Backups kept: ${#keep_backups[@]}"
if command -v numfmt >/dev/null 2>&1; then
    echo "[prune]   Space freed: $(numfmt --to=iec-i --suffix=B $FREED_BYTES)"
else
    echo "[prune]   Space freed: $FREED_BYTES bytes"
fi
[ "$DRY_RUN" = "true" ] && echo "[prune]   (DRY RUN - no files deleted)"

# Update backup catalog
if [ "$DRY_RUN" = "false" ]; then
    echo "[prune] Updating backup catalog..."
    {
        ls -lh "$BACKUP_DIR"/full/*.tar.zst 2>/dev/null | awk '{print $9, $5}' || true
        ls -lh "$BACKUP_DIR"/incremental/*.tar.zst 2>/dev/null | awk '{print $9, $5}' || true
    } > "$BACKUP_DIR/.catalog"
fi

echo "[prune] Backup pruning complete"
