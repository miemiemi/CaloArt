#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --job-name=caloflow_logit
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting logit-normal training on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-flow_ccd2_pred_v_loss_v_logit_normal}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
USE_WANDB="${USE_WANDB:-true}"
ARCH_CONFIG="${ARCH_CONFIG:-}"
CONFIG_NAME="${CONFIG_NAME:-experiment/CaloChallenge/flow_logit_normal}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

CMD=(
  accelerate
  launch
  --multi_gpu
  --num_processes="${NUM_PROCESSES}"
  --main_process_port="${MAIN_PROCESS_PORT}"
  scripts/train.py
  --config-name
  "${CONFIG_NAME}"
  experiment.use_wandb="${USE_WANDB}"
  experiment.output_dir="${OUTPUT_DIR}"
  experiment.run_name="${RUN_NAME}"
)

if [[ -n "${ARCH_CONFIG}" ]]; then
  CMD+=("model@model.architecture=${ARCH_CONFIG}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=("+train.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
