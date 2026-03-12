#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --job-name=caloflow_logit_smoke
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting logit-normal smoke training on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-flow_logit_normal_slurm_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-./tmp_experiments}"
MAX_STEPS="${MAX_STEPS:-1}"

accelerate launch --num_processes=1 scripts/train.py \
  --config-name experiment/CaloChallenge/flow_logit_normal \
  experiment.use_wandb=false \
  experiment.output_dir="${OUTPUT_DIR}" \
  experiment.run_name="${RUN_NAME}" \
  train.max_steps="${MAX_STEPS}" \
  train.logging_steps=1 \
  train.save_strategy=no \
  train.valid_strategy=no \
  train.test_strategy=no \
  train.load_best_model_at_end=false
