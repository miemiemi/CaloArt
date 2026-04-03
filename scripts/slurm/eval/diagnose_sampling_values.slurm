#!/bin/bash
#SBATCH --job-name=sample_diag
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow
cd "${REPO_DIR}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
CONDITION_FILE="${CONDITION_FILE:-}"
NUM_SHOWERS="${NUM_SHOWERS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SAMPLING_STEPS="${SAMPLING_STEPS:-}"
SAMPLING_SOLVER="${SAMPLING_SOLVER:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
DEVICE="${DEVICE:-cuda}"

CMD=(
  python
  -u
  scripts/diagnose_sampling_values.py
  --model-path "${MODEL_PATH}"
  --num-showers "${NUM_SHOWERS}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
)

if [[ -n "${CONDITION_FILE}" ]]; then
  CMD+=(--condition-file "${CONDITION_FILE}")
fi
if [[ -n "${SAMPLING_STEPS}" ]]; then
  CMD+=(--sampling-steps "${SAMPLING_STEPS}")
fi
if [[ -n "${SAMPLING_SOLVER}" ]]; then
  CMD+=(--sampling-solver "${SAMPLING_SOLVER}")
fi
if [[ -n "${OUTPUT_JSON}" ]]; then
  CMD+=(--output-json "${OUTPUT_JSON}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
