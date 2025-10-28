#!/bin/bash
# Validate all required metrics are exposed via Prometheus exporter

set -euo pipefail

PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Critical metrics that MUST be present
REQUIRED_METRICS=(
  # Spreading activation
  "engram_spreading_activations_total"
  "engram_spreading_latency_hot_seconds"
  "engram_spreading_latency_warm_seconds"
  "engram_spreading_latency_cold_seconds"
  "engram_spreading_breaker_state"

  # Consolidation
  "engram_consolidation_runs_total"
  "engram_consolidation_failures_total"
  "engram_consolidation_freshness_seconds"
  "engram_consolidation_novelty_gauge"

  # Storage
  "engram_compaction_success_total"
  "engram_wal_recovery_successes_total"

  # Activation pool
  "activation_pool_available_records"
  "activation_pool_hit_rate"
  "activation_pool_utilization"

  # Adaptive batching
  "adaptive_batch_hot_size"
  "adaptive_batch_latency_ewma_ns"
)

MISSING_METRICS=()
FOUND_COUNT=0

echo "Validating metric coverage against Prometheus at ${PROMETHEUS_URL}..."
echo ""

for metric in "${REQUIRED_METRICS[@]}"; do
  if command -v curl &> /dev/null && command -v jq &> /dev/null; then
    result=$(curl -s -G "${PROMETHEUS_URL}/api/v1/query" \
      --data-urlencode "query=${metric}" | jq -r '.data.result | length' 2>/dev/null || echo "0")
  else
    # Fallback: just check if metric endpoint returns the metric name
    result=$(curl -s "${PROMETHEUS_URL}/api/v1/label/__name__/values" | grep -c "$metric" || echo "0")
  fi

  if [[ "$result" != "0" ]]; then
    echo -e "${GREEN}PASS${NC}: Metric '${metric}' found (${result} series)"
    ((FOUND_COUNT++))
  else
    MISSING_METRICS+=("$metric")
    echo -e "${RED}FAIL${NC}: Metric '${metric}' not found in Prometheus"
  fi
done

echo ""
echo "========================================="
if [[ ${#MISSING_METRICS[@]} -gt 0 ]]; then
  echo -e "${RED}ERROR${NC}: ${#MISSING_METRICS[@]} required metrics missing:"
  for metric in "${MISSING_METRICS[@]}"; do
    echo "  - $metric"
  done
  echo ""
  echo "Found: ${FOUND_COUNT}/${#REQUIRED_METRICS[@]}"
  exit 1
else
  echo -e "${GREEN}SUCCESS${NC}: All ${#REQUIRED_METRICS[@]} required metrics present"
  exit 0
fi
