#!/usr/bin/env bash
# Inject controlled WAL corruption for chaos engineering testing
#
# WARNING: This script intentionally corrupts data for testing purposes.
# Only use in test environments!
#
# Usage: ./inject_wal_corruption.sh [WAL_DIR]

set -euo pipefail

WAL_DIR="${1:-data/wal}"

echo "================================"
echo "Chaos: WAL Corruption Injection"
echo "================================"
echo ""
echo "WAL directory: ${WAL_DIR}"
echo ""

if [ ! -d "$WAL_DIR" ]; then
    echo "Error: WAL directory not found: $WAL_DIR"
    echo ""
    echo "Specify the correct WAL directory as the first argument"
    exit 1
fi

echo "⚠️  WARNING: This will intentionally corrupt WAL files!"
echo ""
read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# Find WAL files
WAL_FILES=$(find "$WAL_DIR" -type f -name "*.wal" | head -5)

if [ -z "$WAL_FILES" ]; then
    echo "No WAL files found in $WAL_DIR"
    exit 1
fi

echo ""
echo "Found WAL files:"
echo "$WAL_FILES"
echo ""

# Create backup
BACKUP_DIR="${WAL_DIR}_backup_$(date +%s)"
echo "Creating backup at: $BACKUP_DIR"
cp -r "$WAL_DIR" "$BACKUP_DIR"

echo ""
echo "Injecting corruption types:"
echo ""

CORRUPTION_COUNT=0

for WAL_FILE in $WAL_FILES; do
    if [ ! -f "$WAL_FILE" ]; then
        continue
    fi

    FILE_SIZE=$(stat -f%z "$WAL_FILE" 2>/dev/null || stat -c%s "$WAL_FILE" 2>/dev/null)

    if [ "$FILE_SIZE" -lt 1024 ]; then
        echo "  Skipping $WAL_FILE (too small)"
        continue
    fi

    # Choose corruption type
    CORRUPTION_TYPE=$((CORRUPTION_COUNT % 4))

    case $CORRUPTION_TYPE in
        0)
            # Truncate file
            NEW_SIZE=$((FILE_SIZE / 2))
            echo "  [Type 1] Truncating $WAL_FILE to $NEW_SIZE bytes"
            truncate -s "$NEW_SIZE" "$WAL_FILE"
            ;;
        1)
            # Flip random bits
            OFFSET=$((FILE_SIZE / 4))
            echo "  [Type 2] Flipping bits in $WAL_FILE at offset $OFFSET"
            printf '\xFF\xFF\xFF\xFF' | dd of="$WAL_FILE" bs=1 seek="$OFFSET" count=4 conv=notrunc 2>/dev/null
            ;;
        2)
            # Zero out checksum area (assume first 32 bytes)
            echo "  [Type 3] Zeroing checksum in $WAL_FILE"
            dd if=/dev/zero of="$WAL_FILE" bs=32 count=1 conv=notrunc 2>/dev/null
            ;;
        3)
            # Insert garbage in middle
            OFFSET=$((FILE_SIZE / 2))
            echo "  [Type 4] Inserting garbage in $WAL_FILE at offset $OFFSET"
            printf 'CORRUPT' | dd of="$WAL_FILE" bs=1 seek="$OFFSET" count=7 conv=notrunc 2>/dev/null
            ;;
    esac

    CORRUPTION_COUNT=$((CORRUPTION_COUNT + 1))

    if [ "$CORRUPTION_COUNT" -ge 3 ]; then
        break
    fi
done

echo ""
echo "✓ WAL corruption injected successfully"
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""
echo "To restore:"
echo "  rm -rf $WAL_DIR"
echo "  mv $BACKUP_DIR $WAL_DIR"
echo ""
echo "To test recovery:"
echo "  Start Engram and observe error handling"
echo "  Check that corruption is detected via checksums"
echo "  Verify system remains available (degraded mode acceptable)"
echo ""
