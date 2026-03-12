#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=edm_cld_dbg
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting legacy EDM CaloLightningDiT smoke validation on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-edm_calolightning_legacy_gpudebug_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-./tmp_experiments}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
USE_WANDB="${USE_WANDB:-false}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
MAX_STEPS="${MAX_STEPS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"

accelerate launch --multi_gpu --num_processes="${NUM_PROCESSES}" --main_process_port="${MAIN_PROCESS_PORT}" scripts/train.py \
  --config-name experiment/CaloChallenge/edm_calolightning_legacy \
  experiment.run_name="${RUN_NAME}" \
  experiment.output_dir="${OUTPUT_DIR}" \
  experiment.use_wandb="${USE_WANDB}" \
  train.max_steps="${MAX_STEPS}" \
  train.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  train.logging_steps=1 \
  train.save_strategy=no \
  train.valid_strategy=no \
  train.test_strategy=no \
  train.load_best_model_at_end=false
