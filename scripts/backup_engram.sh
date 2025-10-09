#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:-${ENGRAM_DATA_DIR:-"./data"}}
BACKUP_ROOT=${2:-"./backups"}
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
BACKUP_DIR="${BACKUP_ROOT}/engram-${TIMESTAMP}"
ARCHIVE_PATH="${BACKUP_DIR}.tar.gz"

mkdir -p "${BACKUP_ROOT}"

echo "[engram-backup] source=${DATA_DIR} target=${ARCHIVE_PATH}" >&2

if [ ! -d "${DATA_DIR}" ]; then
  echo "[engram-backup] error: data directory not found" >&2
  exit 1
fi

mkdir -p "${BACKUP_DIR}"
cp -a "${DATA_DIR}/." "${BACKUP_DIR}/"

tar -czf "${ARCHIVE_PATH}" -C "${BACKUP_ROOT}" "$(basename "${BACKUP_DIR}")"
rm -rf "${BACKUP_DIR}"

echo "[engram-backup] backup stored at ${ARCHIVE_PATH}" >&2
