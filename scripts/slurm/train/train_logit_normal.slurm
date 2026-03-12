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

accelerate launch --num_processes="${NUM_PROCESSES}" scripts/train.py \
  --config-name experiment/CaloChallenge/flow_logit_normal \
  experiment.use_wandb="${USE_WANDB}" \
  experiment.output_dir="${OUTPUT_DIR}" \
  experiment.run_name="${RUN_NAME}"
