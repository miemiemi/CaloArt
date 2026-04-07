#!/bin/bash
#SBATCH --job-name=step_trace
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
TARGET_EVENT="${TARGET_EVENT:?TARGET_EVENT is required}"
TARGET_CELL="${TARGET_CELL:?TARGET_CELL is required}"
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"
CONDITION_FILE="${CONDITION_FILE:-}"
NUM_SHOWERS="${NUM_SHOWERS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAMPLING_STEPS="${SAMPLING_STEPS:-}"
SAMPLING_SOLVER="${SAMPLING_SOLVER:-}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"

CMD=(
  python
  -u
  scripts/trace_sampling_steps.py
  --model-path "${MODEL_PATH}"
  --target-event "${TARGET_EVENT}"
  --target-cell "${TARGET_CELL}"
  --num-showers "${NUM_SHOWERS}"
  --batch-size "${BATCH_SIZE}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --output-json "${OUTPUT_JSON}"
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

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
