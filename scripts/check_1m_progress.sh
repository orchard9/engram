#!/bin/bash
# Monitor 1M Node Production Validation Progress

set -euo pipefail

echo "========================================="
echo "1M Node Production Validation - Status"
echo "========================================="
echo ""

# Check Wikipedia ingestion
echo "Phase 1: Wikipedia Ingestion (50K articles)"
echo "-------------------------------------------"
if ps aux | grep -q "[p]ython ingest.py.*limit 50000"; then
    PID=$(ps aux | grep "[p]ython ingest.py.*limit 50000" | awk '{print $2}')
    CPU=$(ps aux | grep "[p]ython ingest.py.*limit 50000" | awk '{print $3}')
    echo "✓ Status: RUNNING (PID: $PID, CPU: ${CPU}%)"

    # Try to estimate progress (this is approximate)
    if curl -sf http://localhost:7432/health > /dev/null 2>&1; then
        # Note: This endpoint might not exist, adjust as needed
        echo "  Server: Healthy"
    else
        echo "  Server: Unknown status"
    fi
else
    echo "✗ Status: NOT RUNNING"
    echo "  Check if ingestion completed or failed"
fi
echo ""

# Check Engram server
echo "Engram Server Status"
echo "-------------------------------------------"
if curl -sf http://localhost:7432/health > /dev/null 2>&1; then
    echo "✓ Server: RUNNING"
    echo "  Endpoint: http://localhost:7432"
else
    echo "✗ Server: NOT RUNNING"
    echo "  Start with: ./target/release/engram start"
fi
echo ""

# Check tools status
echo "Tools & Environment"
echo "-------------------------------------------"
if [ -d "tools/synthetic_data_generator/venv" ]; then
    echo "✓ Python environment: Ready"
else
    echo "✗ Python environment: Not set up"
fi

if [ -f "tools/synthetic_data_generator/extract_centroids.py" ]; then
    echo "✓ Centroid extraction tool: Ready"
else
    echo "✗ Centroid extraction tool: Missing"
fi

if [ -f "tools/synthetic_data_generator/generate_synthetic.py" ]; then
    echo "✓ Synthetic generation tool: Ready"
else
    echo "✗ Synthetic generation tool: Missing"
fi

if [ -f "tools/synthetic_data_generator/bulk_ingest.py" ]; then
    echo "✓ Bulk ingestion tool: Ready"
else
    echo "✗ Bulk ingestion tool: Missing"
fi
echo ""

# Check disk space
echo "Disk Space"
echo "-------------------------------------------"
AVAILABLE=$(df -h . | tail -1 | awk '{print $4}')
echo "Available: $AVAILABLE"
echo "Required: ~20GB for 1M nodes + 4GB JSONL"
echo ""

# Show next steps
echo "Next Steps"
echo "-------------------------------------------"
if ps aux | grep -q "[p]ython ingest.py.*limit 50000"; then
    echo "1. Wait for Wikipedia ingestion to complete (~12 hours)"
    echo "2. Run: cd tools/synthetic_data_generator && source venv/bin/activate"
    echo "3. Run: python extract_centroids.py --endpoint http://localhost:7432 \\"
    echo "        --num-centroids 100 --output data/centroids.npy"
else
    echo "1. Check Wikipedia ingestion status"
    echo "2. If failed, restart: cd tools/wikipedia_ingest && source venv/bin/activate"
    echo "   python ingest.py --endpoint http://localhost:7432 --limit 50000"
fi
echo ""

echo "========================================="
echo "Monitor progress: watch -n 60 ./scripts/check_1m_progress.sh"
echo "Full plan: roadmap/milestone-17/1M_NODE_PRODUCTION_VALIDATION_PLAN.md"
echo "========================================="
