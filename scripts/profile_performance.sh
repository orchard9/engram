#!/usr/bin/env bash
# Operator-focused performance profiling with cache efficiency analysis
#
# This script profiles Engram's performance including:
# - CPU profiling with cache miss analysis
# - NUMA topology and memory locality
# - Memory profiling with huge page analysis
# - Lock contention detection
# - I/O profiling
# - Query latency percentiles
# - Bottleneck identification
#
# Usage: ./scripts/profile_performance.sh [duration_seconds] [output_dir]

set -euo pipefail

DURATION="${1:-60}"
OUTPUT_DIR="${2:-./profile-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_DIR"

echo "Profiling Engram for ${DURATION}s..."
echo "Output directory: $OUTPUT_DIR"

# 1. Capture baseline metrics
echo "Step 1/8: Capturing baseline metrics..."
if curl -sf http://localhost:9090/api/v1/query --max-time 5 \
  -d 'query=engram_memory_operation_duration_seconds' \
  > "$OUTPUT_DIR/baseline_metrics.json" 2>/dev/null; then
  echo "  Baseline metrics captured"
else
  echo "  WARNING: Prometheus not available, skipping baseline metrics"
fi

# 2. CPU profiling with cache miss analysis
echo "Step 2/8: CPU profiling with cache analysis..."
ENGRAM_PID=$(pgrep engram || echo "")
if [ -z "$ENGRAM_PID" ]; then
  echo "  WARNING: Engram process not found, skipping CPU profiling"
else
  if command -v perf &> /dev/null; then
    # Record CPU cycles and cache misses
    echo "  Recording CPU profile with cache events..."
    sudo perf record -F 999 -p "$ENGRAM_PID" -g \
        -e cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses \
        -- sleep "$DURATION" 2>/dev/null || echo "  WARNING: perf record failed"

    if [ -f perf.data ]; then
      sudo perf script > "$OUTPUT_DIR/cpu_profile.txt" 2>/dev/null || true

      # Generate cache miss statistics
      sudo perf stat -p "$ENGRAM_PID" \
          -e cache-references,cache-misses,LLC-loads,LLC-load-misses \
          sleep 10 2> "$OUTPUT_DIR/cache_stats.txt" || true

      # Generate flamegraph data if stackcollapse is available
      if command -v stackcollapse-perf.pl &> /dev/null; then
        sudo perf script | stackcollapse-perf.pl > "$OUTPUT_DIR/collapsed.txt" 2>/dev/null || true
      fi

      echo "  CPU profile saved to $OUTPUT_DIR/cpu_profile.txt"
      rm -f perf.data
    fi
  else
    echo "  WARNING: perf not available, skipping CPU profile"
  fi
fi

# 3. NUMA topology and memory locality
echo "Step 3/8: NUMA topology analysis..."
if command -v numactl &> /dev/null; then
  numactl --hardware > "$OUTPUT_DIR/numa_topology.txt" 2>/dev/null || true
  numactl --show > "$OUTPUT_DIR/numa_policy.txt" 2>/dev/null || true

  # Per-node memory usage
  if [ -d /sys/devices/system/node ]; then
    for node_dir in /sys/devices/system/node/node*; do
      if [ -d "$node_dir" ]; then
        node=$(basename "$node_dir")
        echo "=== $node ===" >> "$OUTPUT_DIR/numa_memory.txt"
        cat "$node_dir/meminfo" >> "$OUTPUT_DIR/numa_memory.txt" 2>/dev/null || true
      fi
    done
  fi
  echo "  NUMA topology saved"
else
  echo "  WARNING: numactl not available, skipping NUMA analysis"
fi

# 4. Memory profiling with huge page analysis
echo "Step 4/8: Memory profiling..."
if [ -n "$ENGRAM_PID" ] && [ -f "/proc/$ENGRAM_PID/smaps" ]; then
  cat "/proc/$ENGRAM_PID/smaps" > "$OUTPUT_DIR/memory_map.txt" 2>/dev/null || true

  # Extract huge page usage
  if grep -qE "AnonHugePages|ShmemPmdMapped|FilePmdMapped" "/proc/$ENGRAM_PID/smaps" 2>/dev/null; then
    grep -E "AnonHugePages|ShmemPmdMapped|FilePmdMapped" "/proc/$ENGRAM_PID/smaps" | \
        awk '{sum+=$2} END {print "Huge Pages: " sum/1024 " MB"}' > "$OUTPUT_DIR/huge_pages.txt" 2>/dev/null || true
  fi

  ps -p "$ENGRAM_PID" -o pid,vsz,rss,pmem,comm > "$OUTPUT_DIR/memory_usage.txt" 2>/dev/null || true
  echo "  Memory profile saved"
elif [ -n "$ENGRAM_PID" ]; then
  # Fallback for non-Linux systems
  ps -p "$ENGRAM_PID" -o pid,vsz,rss,comm > "$OUTPUT_DIR/memory_usage.txt" 2>/dev/null || true
  echo "  Basic memory usage saved"
else
  echo "  WARNING: Cannot profile memory, Engram not running"
fi

# 5. Lock contention analysis
echo "Step 5/8: Lock contention analysis..."
if [ -n "$ENGRAM_PID" ] && [ -d "/proc/$ENGRAM_PID/task" ]; then
  # Sample thread states to detect contention
  for i in {1..10}; do
    echo "=== Sample $i ===" >> "$OUTPUT_DIR/thread_states.txt"
    for tid_dir in /proc/"$ENGRAM_PID"/task/*; do
      if [ -f "$tid_dir/stat" ]; then
        awk '{print $2, $3}' "$tid_dir/stat" >> "$OUTPUT_DIR/thread_states.txt" 2>/dev/null || true
      fi
    done
    sleep 1
  done
  echo "  Thread states sampled"
else
  echo "  WARNING: Cannot analyze thread states"
fi

# 6. I/O profiling
echo "Step 6/8: I/O profiling..."
if [ -n "$ENGRAM_PID" ]; then
  if command -v iotop &> /dev/null; then
    sudo iotop -b -n 10 -d 1 -p "$ENGRAM_PID" > "$OUTPUT_DIR/io_stats.txt" 2>/dev/null || true
    echo "  I/O stats captured with iotop"
  elif [ -f "/proc/$ENGRAM_PID/io" ]; then
    # Fallback to /proc/io
    for i in {1..10}; do
      echo "=== Sample $i ===" >> "$OUTPUT_DIR/io_stats.txt"
      cat "/proc/$ENGRAM_PID/io" >> "$OUTPUT_DIR/io_stats.txt" 2>/dev/null || true
      sleep 1
    done
    echo "  I/O stats captured from /proc"
  else
    echo "  WARNING: Cannot profile I/O"
  fi
fi

# 7. Query latency analysis with percentiles
echo "Step 7/8: Analyzing query latencies..."
if curl -sf http://localhost:9090/api/v1/query --max-time 5 \
  -d 'query=engram_memory_operation_duration_seconds' >/dev/null 2>&1; then

  for percentile in 50 90 95 99 99.9; do
    # Convert percentile to decimal for histogram_quantile
    quantile=$(echo "scale=3; $percentile / 100" | bc)
    curl -sf http://localhost:9090/api/v1/query --max-time 5 \
      -d "query=histogram_quantile($quantile, engram_memory_operation_duration_seconds_bucket)" \
      2>/dev/null | jq -r ".data.result[] | \"P$percentile \(.metric.operation): \(.value[1])s\"" \
      >> "$OUTPUT_DIR/latency_percentiles.txt" 2>/dev/null || true
  done
  echo "  Latency percentiles captured"
else
  echo "  WARNING: Prometheus not available, skipping latency analysis"
fi

# 8. Comprehensive bottleneck identification
echo "Step 8/8: Identifying bottlenecks..."
cat > "$OUTPUT_DIR/bottleneck_report.txt" <<EOF
Engram Performance Analysis Report
Generated: $(date)
Duration: ${DURATION}s

=== CRITICAL METRICS ===

Cache Efficiency:
$(grep "cache-misses" "$OUTPUT_DIR/cache_stats.txt" 2>/dev/null || echo "N/A - perf not available")

Memory Pressure:
$(cat "$OUTPUT_DIR/memory_usage.txt" 2>/dev/null || echo "N/A")
$(cat "$OUTPUT_DIR/huge_pages.txt" 2>/dev/null || echo "Huge Pages: N/A")

NUMA Locality:
$(grep "node 0" "$OUTPUT_DIR/numa_memory.txt" 2>/dev/null | head -1 || echo "Single NUMA node or N/A")

Latency Analysis (all percentiles):
$(cat "$OUTPUT_DIR/latency_percentiles.txt" 2>/dev/null | grep "recall:" | sort || echo "N/A")

=== BOTTLENECK IDENTIFICATION ===
EOF

# Advanced bottleneck analysis
if [ -f "$OUTPUT_DIR/latency_percentiles.txt" ]; then
  P99_RECALL=$(grep "P99 recall:" "$OUTPUT_DIR/latency_percentiles.txt" 2>/dev/null | cut -d: -f2 | tr -d 's' | tr -d ' ' || echo "")
  if [ -n "$P99_RECALL" ] && command -v bc &> /dev/null; then
    if (( $(echo "$P99_RECALL > 0.010" | bc -l 2>/dev/null) )); then
      echo "WARNING: Recall P99 latency ${P99_RECALL}s exceeds 10ms target" >> "$OUTPUT_DIR/bottleneck_report.txt"
      echo "   Root Cause Analysis:" >> "$OUTPUT_DIR/bottleneck_report.txt"

      # Check cache misses
      if [ -f "$OUTPUT_DIR/cache_stats.txt" ]; then
        CACHE_MISS_RATE=$(grep "cache-misses" "$OUTPUT_DIR/cache_stats.txt" 2>/dev/null | grep -o '[0-9.]*%' | tr -d '%' || echo "0")
        if [ -n "$CACHE_MISS_RATE" ] && (( $(echo "$CACHE_MISS_RATE > 10" | bc -l 2>/dev/null) )); then
          echo "   - High cache miss rate: ${CACHE_MISS_RATE}%" >> "$OUTPUT_DIR/bottleneck_report.txt"
          echo "     ACTION: Increase prefetch distance, review data layout" >> "$OUTPUT_DIR/bottleneck_report.txt"
        fi
      fi

      # Check memory usage
      if [ -f "$OUTPUT_DIR/memory_usage.txt" ]; then
        RSS_KB=$(awk 'NR==2 {print $3}' "$OUTPUT_DIR/memory_usage.txt" 2>/dev/null || echo "0")
        RSS_MB=$((RSS_KB / 1024))
        if [ "$RSS_MB" -gt 4096 ]; then
          echo "   - High memory usage: ${RSS_MB}MB" >> "$OUTPUT_DIR/bottleneck_report.txt"
          echo "     ACTION: Increase hot tier eviction rate" >> "$OUTPUT_DIR/bottleneck_report.txt"
        fi
      fi
    fi
  fi
fi

cat "$OUTPUT_DIR/bottleneck_report.txt"
echo ""
echo "Full report saved to $OUTPUT_DIR/"
if [ -f "$OUTPUT_DIR/collapsed.txt" ]; then
  echo "Generate flamegraph: flamegraph.pl $OUTPUT_DIR/collapsed.txt > $OUTPUT_DIR/flamegraph.svg"
fi
