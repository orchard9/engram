#!/usr/bin/env bash
set -euo pipefail

ENDPOINT=${1:-"http://127.0.0.1:7432/api/v1/system/health"}
TIMEOUT=${TIMEOUT:-5}

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

response=$(curl --silent --show-error --fail --max-time "${TIMEOUT}" "${ENDPOINT}")
status=$(python3 -c 'import json, sys; print(json.load(sys.stdin).get("status", "unknown"))' <<<"${response}" 2>/dev/null || echo "unknown")

printf "health=%s endpoint=%s\n" "${status}" "${ENDPOINT}" >&2

echo "${response}"
