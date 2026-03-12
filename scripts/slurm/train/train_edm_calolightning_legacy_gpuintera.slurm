#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpuintera
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=edm_cld_legacy
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting legacy EDM CaloLightningDiT training on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-edm_calolightning_legacy}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoint/experiments}"
USE_WANDB="${USE_WANDB:-true}"

python scripts/train.py \
  --config-name experiment/CaloChallenge/edm_calolightning_legacy \
  experiment.run_name="${RUN_NAME}" \
  experiment.output_dir="${OUTPUT_DIR}" \
  experiment.use_wandb="${USE_WANDB}"
