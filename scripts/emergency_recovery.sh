#!/usr/bin/env bash
# Emergency recovery procedures for critical production failures
# Use with extreme caution - always backup first

set -euo pipefail

ENGRAM_URL="${ENGRAM_URL:-http://localhost:7432}"
DATA_DIR="${ENGRAM_DATA_DIR:-./data}"
BACKUP_DIR="${ENGRAM_BACKUP_DIR:-./backups}"
CONFIG_FILE="${ENGRAM_CONFIG:-~/.config/engram/config.toml}"

usage() {
  cat <<EOF
Emergency Recovery Tool for Engram

Usage: $0 [OPTIONS] [MODE]

Modes:
  --sanitize-nan          Remove NaN/Infinity values from data
  --fix-wal-corruption    Move corrupted WAL entries aside
  --rebuild-indices       Rebuild all indices from scratch
  --reset-space <id>      Reset a specific memory space to empty state
  --restore-latest        Restore from most recent backup
  --readonly-mode         Start in read-only mode (no writes)

Options:
  --dry-run              Show what would be done without making changes
  --space <id>           Target specific space (default: all)
  --backup-first         Create backup before any changes (recommended)

Examples:
  # Safe NaN cleanup with dry-run first
  $0 --sanitize-nan --dry-run
  $0 --sanitize-nan --backup-first

  # Fix WAL corruption for specific space
  $0 --fix-wal-corruption --space tenant_a --backup-first

  # Emergency restore from backup
  $0 --restore-latest --space default

EOF
  exit 1
}

# Parse arguments
MODE=""
DRY_RUN=false
SPACE="all"
BACKUP_FIRST=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --sanitize-nan)
      MODE="sanitize-nan"
      shift
      ;;
    --fix-wal-corruption)
      MODE="fix-wal"
      shift
      ;;
    --rebuild-indices)
      MODE="rebuild-indices"
      shift
      ;;
    --reset-space)
      MODE="reset-space"
      SPACE="${2:-}"
      if [ -z "$SPACE" ] || [ "$SPACE" = "all" ]; then
        echo "ERROR: --reset-space requires a specific space ID (not 'all')"
        exit 1
      fi
      shift 2
      ;;
    --restore-latest)
      MODE="restore-latest"
      shift
      ;;
    --readonly-mode)
      MODE="readonly"
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --space)
      SPACE="${2:-all}"
      shift 2
      ;;
    --backup-first)
      BACKUP_FIRST=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

if [ -z "$MODE" ]; then
  usage
fi

echo "========================================"
echo "   Engram Emergency Recovery Tool      "
echo "========================================"
echo "Mode: $MODE"
echo "Target: $SPACE"
echo "Dry Run: $DRY_RUN"
echo "Backup First: $BACKUP_FIRST"
echo ""

# Confirm destructive operations
if [ "$DRY_RUN" = false ]; then
  echo "WARNING: This operation will modify production data."
  echo "Press Ctrl+C now to abort, or type 'yes' to continue."
  read -r -p "Continue? (yes/no): " CONFIRM
  if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
  fi
  echo ""
fi

# Create backup if requested
if [ "$BACKUP_FIRST" = true ] && [ "$DRY_RUN" = false ]; then
  echo "Creating backup before proceeding..."
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  if [ -f "$SCRIPT_DIR/backup_full.sh" ]; then
    if [ "$SPACE" = "all" ]; then
      "$SCRIPT_DIR/backup_full.sh" "$BACKUP_DIR/emergency-pre-$MODE-$TIMESTAMP.tar.zst" || {
        echo "Backup failed. Aborting recovery."
        exit 1
      }
    else
      "$SCRIPT_DIR/backup_full.sh" "$BACKUP_DIR/emergency-pre-$MODE-$SPACE-$TIMESTAMP.tar.zst" --space "$SPACE" || {
        echo "Backup failed. Aborting recovery."
        exit 1
      }
    fi
    echo "Backup created successfully."
  else
    echo "WARNING: Backup script not found at $SCRIPT_DIR/backup_full.sh"
    echo "Proceeding without backup as requested."
  fi
  echo ""
fi

# Execute recovery mode
case $MODE in
  sanitize-nan)
    echo "Scanning for NaN/Infinity values..."
    echo ""

    if [ "$DRY_RUN" = true ]; then
      echo "[DRY RUN] Would perform the following actions:"
      echo ""
      echo "1. Check for NaN/Infinity in confidence scores via API"
      if command -v curl > /dev/null && command -v jq > /dev/null; then
        echo "   Checking memories with invalid confidence..."
        INVALID_COUNT=$(curl -s "$ENGRAM_URL/api/v1/memories" 2>/dev/null | \
          jq '[.[] | select(.confidence | (isnan or isinfinite))] | length' 2>/dev/null || echo "0")
        echo "   Found: $INVALID_COUNT memories with invalid confidence"
      else
        echo "   (curl or jq not available for checking)"
      fi
      echo ""
      echo "2. Enable validation in config:"
      echo "   [validation]"
      echo "   check_finite_embeddings = true"
      echo "   check_finite_confidence = true"
      echo "   clamp_invalid_values = true"
      echo "   replace_nan_with_zero = true"
      echo ""
      echo "3. Restart Engram to apply sanitization on load"
      echo ""
      echo "[DRY RUN] No changes made. Run without --dry-run to execute."
    else
      echo "Enabling validation in config..."

      # Ensure config file exists
      mkdir -p "$(dirname "$CONFIG_FILE")"
      touch "$CONFIG_FILE"

      # Check if validation section already exists
      if grep -q "\[validation\]" "$CONFIG_FILE" 2>/dev/null; then
        echo "Validation section already exists in config."
        echo "Please manually verify these settings:"
      else
        cat >> "$CONFIG_FILE" <<EOF

# Added by emergency recovery at $(date)
[validation]
check_finite_embeddings = true
check_finite_confidence = true
clamp_invalid_values = true
replace_nan_with_zero = true
EOF
        echo "Validation settings added to config."
      fi

      echo ""
      echo "Configuration updated: $CONFIG_FILE"
      echo ""
      echo "To complete recovery:"
      echo "  1. Stop Engram: systemctl stop engram (or kill process)"
      echo "  2. Start Engram: systemctl start engram (or restart manually)"
      echo "  3. Verify: curl $ENGRAM_URL/api/v1/system/health"
      echo ""
      echo "The validation will sanitize data on load."
    fi
    ;;

  fix-wal)
    echo "Scanning for corrupted WAL entries..."
    echo ""

    if [ "$SPACE" = "all" ]; then
      if [ -d "$DATA_DIR" ]; then
        SPACES=$(ls -1 "$DATA_DIR" 2>/dev/null | grep -v "\.log" || echo "")
      else
        SPACES=""
      fi
    else
      SPACES="$SPACE"
    fi

    if [ -z "$SPACES" ]; then
      echo "No spaces found in $DATA_DIR"
      exit 1
    fi

    for space in $SPACES; do
      WAL_DIR="$DATA_DIR/$space/wal"
      if [ ! -d "$WAL_DIR" ]; then
        echo "Skipping $space (no WAL directory)"
        continue
      fi

      echo "Checking WAL for space: $space"
      CORRUPT_DIR="$DATA_DIR/$space/wal-corrupt-$(date +%Y%m%d_%H%M%S)"

      if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would create: $CORRUPT_DIR"
        WAL_COUNT=$(find "$WAL_DIR" -name "*.log" 2>/dev/null | wc -l | tr -d ' ')
        echo "[DRY RUN] Would scan $WAL_COUNT WAL files for corruption"
        echo "[DRY RUN] Would move corrupted files to quarantine"
        echo ""
      else
        mkdir -p "$CORRUPT_DIR"

        # Look for recent deserialization errors in logs
        CORRUPT_FILES=0
        if command -v journalctl > /dev/null 2>&1; then
          echo "Checking logs for WAL errors..."
          journalctl -u engram --since "1 hour ago" --no-pager 2>/dev/null | \
            grep "Failed to deserialize WAL\|WAL corruption\|WAL.*error" | \
            head -20 | \
            while read -r line; do
              echo "  Found error: ${line:0:100}..."
              CORRUPT_FILES=$((CORRUPT_FILES + 1))
            done || true
        fi

        # Move any .corrupted or .tmp files
        while IFS= read -r file; do
          if [ -n "$file" ]; then
            echo "Moving $(basename "$file") to quarantine"
            mv "$file" "$CORRUPT_DIR/" 2>/dev/null || true
          fi
        done < <(find "$WAL_DIR" -name "*.corrupted" -o -name "*.tmp" 2>/dev/null || true)

        if [ "$CORRUPT_FILES" -gt 0 ]; then
          echo ""
          echo "Moved $CORRUPT_FILES corrupted files to: $CORRUPT_DIR"
          echo "Review files before deleting. Restart Engram to continue."
        else
          echo "No obviously corrupted WAL files found."
          echo "If issues persist, check logs for specific file offsets."
        fi
        echo ""
      fi
    done
    ;;

  rebuild-indices)
    echo "Rebuilding indices..."
    echo ""

    if [ "$DRY_RUN" = true ]; then
      echo "[DRY RUN] Would trigger index rebuild for spaces: $SPACE"
      if [ "$SPACE" = "all" ]; then
        echo "[DRY RUN] Would POST to: $ENGRAM_URL/api/v1/system/rebuild-indices"
      else
        echo "[DRY RUN] Would POST to: $ENGRAM_URL/api/v1/system/rebuild-indices?space=$SPACE"
      fi
      echo ""
      echo "[DRY RUN] No changes made. Run without --dry-run to execute."
    else
      if command -v curl > /dev/null; then
        if [ "$SPACE" = "all" ]; then
          echo "Triggering index rebuild for all spaces..."
          curl -X POST "$ENGRAM_URL/api/v1/system/rebuild-indices" 2>&1 || {
            echo "API call failed. Engram may not be running or endpoint not available."
            exit 1
          }
        else
          echo "Triggering index rebuild for space: $SPACE"
          curl -X POST "$ENGRAM_URL/api/v1/system/rebuild-indices?space=$SPACE" 2>&1 || {
            echo "API call failed. Engram may not be running or endpoint not available."
            exit 1
          }
        fi

        echo ""
        echo "Index rebuild triggered successfully."
        echo ""
        echo "Monitor progress with:"
        echo "  watch 'curl -s $ENGRAM_URL/api/v1/system/health | jq .indices'"
      else
        echo "ERROR: curl not available. Cannot trigger rebuild."
        exit 1
      fi
    fi
    ;;

  reset-space)
    echo "DANGER: Resetting space '$SPACE' to empty state"
    echo ""

    if [ "$DRY_RUN" = true ]; then
      SPACE_DIR="$DATA_DIR/$SPACE"
      echo "[DRY RUN] Would delete all data in: $SPACE_DIR"
      echo "[DRY RUN] Would remove:"
      [ -d "$SPACE_DIR/wal" ] && echo "  - WAL files: $(find "$SPACE_DIR/wal" -name "*.log" 2>/dev/null | wc -l | tr -d ' ') files"
      [ -d "$SPACE_DIR/hot" ] && echo "  - Hot tier: $(du -sh "$SPACE_DIR/hot" 2>/dev/null | awk '{print $1}')"
      [ -d "$SPACE_DIR/warm" ] && echo "  - Warm tier: $(du -sh "$SPACE_DIR/warm" 2>/dev/null | awk '{print $1}')"
      [ -d "$SPACE_DIR/cold" ] && echo "  - Cold tier: $(du -sh "$SPACE_DIR/cold" 2>/dev/null | awk '{print $1}')"
      [ -d "$SPACE_DIR/indices" ] && echo "  - Indices: $(du -sh "$SPACE_DIR/indices" 2>/dev/null | awk '{print $1}')"
      echo "[DRY RUN] Would preserve directory structure"
      echo ""
      echo "[DRY RUN] No changes made. Run without --dry-run to execute."
    else
      SPACE_DIR="$DATA_DIR/$SPACE"

      if [ ! -d "$SPACE_DIR" ]; then
        echo "ERROR: Space directory not found: $SPACE_DIR"
        exit 1
      fi

      echo "Deleting data files (preserving structure)..."

      # Delete data but preserve structure
      [ -d "$SPACE_DIR/wal" ] && rm -rf "${SPACE_DIR:?}/wal/"*.log 2>/dev/null && echo "  Cleared WAL"
      [ -d "$SPACE_DIR/hot" ] && rm -rf "${SPACE_DIR:?}/hot/"* 2>/dev/null && echo "  Cleared hot tier"
      [ -d "$SPACE_DIR/warm" ] && rm -rf "${SPACE_DIR:?}/warm/"* 2>/dev/null && echo "  Cleared warm tier"
      [ -d "$SPACE_DIR/cold" ] && rm -rf "${SPACE_DIR:?}/cold/"* 2>/dev/null && echo "  Cleared cold tier"
      [ -d "$SPACE_DIR/indices" ] && rm -rf "${SPACE_DIR:?}/indices/"* 2>/dev/null && echo "  Cleared indices"

      echo ""
      echo "Space '$SPACE' has been reset to empty state."
      echo "Restart Engram to reinitialize."
    fi
    ;;

  restore-latest)
    echo "Restoring from latest backup..."
    echo ""

    if [ "$SPACE" = "all" ]; then
      LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/latest-full-*.tar.zst 2>/dev/null | head -1 || echo "")
      if [ -z "$LATEST_BACKUP" ]; then
        LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/*.tar.zst 2>/dev/null | head -1 || echo "")
      fi
    else
      LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/*"$SPACE"*.tar.zst 2>/dev/null | head -1 || echo "")
    fi

    if [ -z "$LATEST_BACKUP" ]; then
      echo "ERROR: No backup found for space: $SPACE"
      echo "Searched in: $BACKUP_DIR"
      exit 1
    fi

    echo "Latest backup: $LATEST_BACKUP"
    echo "Size: $(du -h "$LATEST_BACKUP" | awk '{print $1}')"
    echo ""

    if [ "$DRY_RUN" = true ]; then
      echo "[DRY RUN] Would restore from: $LATEST_BACKUP"
      echo "[DRY RUN] Would execute:"
      echo "  1. Stop Engram"
      echo "  2. Extract backup to: $DATA_DIR"
      echo "  3. Restart Engram"
      echo ""
      echo "[DRY RUN] No changes made. Run without --dry-run to execute."
    else
      SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
      if [ -f "$SCRIPT_DIR/restore.sh" ]; then
        echo "Executing restore script..."
        "$SCRIPT_DIR/restore.sh" "$LATEST_BACKUP" || {
          echo "Restore failed."
          exit 1
        }
        echo ""
        echo "Restore complete. Engram should be restarted."
      else
        echo "ERROR: Restore script not found at $SCRIPT_DIR/restore.sh"
        echo "Manual restore required:"
        echo "  1. Stop Engram"
        echo "  2. tar -xzf $LATEST_BACKUP -C /"
        echo "  3. Start Engram"
        exit 1
      fi
    fi
    ;;

  readonly)
    echo "Starting in read-only mode..."
    echo ""

    if [ "$DRY_RUN" = true ]; then
      echo "[DRY RUN] Would set ENGRAM_READONLY=true"
      echo "[DRY RUN] Would add to: /etc/default/engram or systemd config"
      echo "[DRY RUN] Would restart Engram"
      echo ""
      echo "[DRY RUN] No changes made. Run without --dry-run to execute."
    else
      export ENGRAM_READONLY=true

      # Try to make it persistent
      if [ -f /etc/default/engram ]; then
        if ! grep -q "ENGRAM_READONLY" /etc/default/engram; then
          echo "ENGRAM_READONLY=true" | sudo tee -a /etc/default/engram > /dev/null
          echo "Added ENGRAM_READONLY=true to /etc/default/engram"
        fi
      else
        echo "Note: /etc/default/engram not found."
        echo "Add ENGRAM_READONLY=true to your systemd service or startup script."
      fi

      echo ""
      echo "Restart Engram to enable read-only mode:"
      echo "  systemctl restart engram"
      echo ""
      echo "In read-only mode, Engram will:"
      echo "  - Accept read operations (queries, activations)"
      echo "  - Reject write operations (store, update, delete)"
      echo "  - Not write to WAL or tiers"
    fi
    ;;

  *)
    echo "Unknown mode: $MODE"
    usage
    ;;
esac

echo ""
echo "========================================"
echo "Recovery operation complete"
echo ""

if [ "$DRY_RUN" = true ]; then
  echo "This was a DRY RUN - no changes were made."
  echo "Run again without --dry-run to execute."
else
  echo "Changes have been applied."
  echo "Review logs and verify system health before resuming normal operations."
  echo ""
  echo "Recommended next steps:"
  echo "  1. Run health check: ./scripts/diagnose_health.sh"
  echo "  2. Check logs: journalctl -u engram -n 100"
  echo "  3. Verify operations: curl $ENGRAM_URL/api/v1/system/health"
fi

echo ""
exit 0
