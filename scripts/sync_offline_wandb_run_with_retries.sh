#!/usr/bin/env bash
# Sync one offline W&B run directory with retries.
# Usage:
#   bash scripts/sync_offline_wandb_run_with_retries.sh \
#     /path/to/offline-run-... \
#     "miemimail2020-Institute of High Energy Physics" \
#     CaloArt_paper
#
# The script is designed for Slurm CPU jobs on this cluster:
# - keeps BLAS/OpenMP thread counts at 1
# - uses a per-attempt TMPDIR
# - skips console log upload to reduce flaky streaming failures
# - retries the sync command several times before giving up

set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 OFFLINE_RUN_DIR ENTITY PROJECT [RUN_ID]" >&2
  exit 2
fi

OFFLINE_RUN_DIR="$1"
ENTITY="$2"
PROJECT="$3"
RUN_ID="${4:-}"

RETRY_COUNT="${RETRY_COUNT:-5}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-60}"
REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
WANDB_BIN="/aifs/user/home/huangzhengkun/miniconda3/envs/CaloFlow/bin/wandb"
PYTHON_BIN="/aifs/user/home/huangzhengkun/miniconda3/envs/CaloFlow/bin/python3.10"
BASE_TMPDIR="${BASE_TMPDIR:-${REPO_DIR}/.codex/tmp/wandb-sync}"

if [ ! -d "${OFFLINE_RUN_DIR}" ]; then
  echo "Offline run dir not found: ${OFFLINE_RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${BASE_TMPDIR}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB_DISABLE_GIT=true
export WANDB_DISABLE_CODE=true
export WANDB_SAVE_CODE=false
export WANDB__EXECUTABLE="${PYTHON_BIN}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

for attempt in $(seq 1 "${RETRY_COUNT}"); do
  export TMPDIR="${BASE_TMPDIR}/attempt_${attempt}_$$"
  mkdir -p "${TMPDIR}"

  CMD=(
    "${WANDB_BIN}" sync
    --include-offline
    --mark-synced
    --skip-console
    --project "${PROJECT}"
    --entity "${ENTITY}"
  )
  if [ -n "${RUN_ID}" ]; then
    CMD+=(--id "${RUN_ID}")
  fi
  CMD+=("${OFFLINE_RUN_DIR}")

  echo "[$(date)] sync attempt ${attempt}/${RETRY_COUNT}"
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  echo

  if "${CMD[@]}"; then
    echo "[$(date)] sync succeeded on attempt ${attempt}"
    exit 0
  fi

  echo "[$(date)] sync failed on attempt ${attempt}" >&2
  if [ "${attempt}" -lt "${RETRY_COUNT}" ]; then
    echo "Sleeping ${RETRY_SLEEP_SECONDS}s before retry..." >&2
    sleep "${RETRY_SLEEP_SECONDS}"
  fi
done

echo "All ${RETRY_COUNT} sync attempts failed for ${OFFLINE_RUN_DIR}" >&2
exit 1
