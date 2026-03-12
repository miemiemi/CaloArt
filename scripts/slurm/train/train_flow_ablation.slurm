#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-1
#SBATCH --job-name=flow_ablate
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%A_%a.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%A_%a.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

CONFIGS=(
  "experiment/CaloChallenge/flow_logit_normal"
  "experiment/CaloChallenge/flow_jit_like"
)

RUN_NAMES=(
  "flow_ccd2_pred_v_loss_v_logit_normal"
  "flow_ccd2_jit_like"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG_NAME="${CONFIG_NAME:-${CONFIGS[$TASK_ID]}}"
RUN_NAME="${RUN_NAME:-${RUN_NAMES[$TASK_ID]}}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
USE_WANDB="${USE_WANDB:-true}"
MAX_STEPS="${MAX_STEPS:-}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"

echo "Starting job on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "USE_WANDB=${USE_WANDB}"
echo "MAX_STEPS=${MAX_STEPS:-default}"
echo "MIXED_PRECISION=${MIXED_PRECISION}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

CMD=(
  accelerate launch
  --num_processes="${NUM_PROCESSES}"
  scripts/train.py
  --config-name "${CONFIG_NAME}"
  experiment.use_wandb="${USE_WANDB}"
  experiment.output_dir="${OUTPUT_DIR}"
  experiment.run_name="${RUN_NAME}"
  accelerator.mixed_precision="${MIXED_PRECISION}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  CMD+=(train.max_steps="${MAX_STEPS}")
fi

echo "Launching:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
