#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --job-name=flow_ccd3_sm
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting CCD3 energy-only smoke run on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-flow_ccd3_energy_only_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-./tmp_experiments}"
MAX_STEPS="${MAX_STEPS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"

accelerate launch \
  --num_machines=1 \
  --num_processes=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  scripts/train.py \
  --config-name experiment/CaloChallenge/flow_ccd3_energy_only \
  experiment.use_wandb=false \
  experiment.output_dir="${OUTPUT_DIR}" \
  experiment.run_name="${RUN_NAME}" \
  train.max_steps="${MAX_STEPS}" \
  train.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  train.logging_steps=1 \
  train.valid_strategy=steps \
  train.valid_steps=1 \
  train.save_strategy=no \
  train.test_strategy=no \
  train.load_best_model_at_end=false \
  accelerator.mixed_precision=no
