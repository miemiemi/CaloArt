#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=flow_p385_sm
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting Flow patch=(3,8,5) smoke run on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-flow_ccd2_patch385_coarse_t54_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-./tmp_experiments}"
MAX_STEPS="${MAX_STEPS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"

accelerate launch \
  --num_machines=1 \
  --num_processes=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  scripts/train.py \
  --config-name experiment/CaloChallenge/flow_ccd2_pred_v_all_lightingdit_freatures_patch_3_8_5_coarse_latest \
  experiment.use_wandb=false \
  experiment.output_dir="${OUTPUT_DIR}" \
  experiment.run_name="${RUN_NAME}" \
  train.max_steps="${MAX_STEPS}" \
  train.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  train.logging_steps=1 \
  train.save_strategy=no \
  train.valid_strategy=no \
  train.test_strategy=no \
  train.load_best_model_at_end=false \
  accelerator.mixed_precision=no
