#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --job-name=edm_calodit_legacy
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting legacy EDM CaloDiT training on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-edm_ccd2_0.5KeV_1en8}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoint/experiments}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
USE_WANDB="${USE_WANDB:-true}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES}" --main_process_port="${MAIN_PROCESS_PORT}" scripts/train.py \
  --config-name experiment/CaloChallenge/edm_calodit_legacy \
  experiment.run_name="${RUN_NAME}" \
  experiment.output_dir="${OUTPUT_DIR}" \
  experiment.use_wandb="${USE_WANDB}"
