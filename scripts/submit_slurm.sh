#!/bin/bash

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
LOG_DIR="${REPO_DIR}/logs/slurm"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/submit_slurm.sh <slurm-script> [script-args...]"
  exit 1
fi

SCRIPT_PATH="$1"
shift

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "Slurm script not found: ${SCRIPT_PATH}"
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

echo "Submitting ${SCRIPT_PATH}"
sbatch "${SCRIPT_PATH}" "$@"
