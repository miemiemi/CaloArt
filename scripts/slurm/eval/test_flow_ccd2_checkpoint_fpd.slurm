#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --job-name=flow_ckpt_fpd
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting checkpoint-only CCD2 FPD test on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/experiments/caloflow_v1/flow_ccd2_pred_v_all_lightingdit_freatures_fpd_e2e/final_model.pt}"
RUN_NAME="${RUN_NAME:-flow_ccd2_pred_v_all_lightingdit_freatures_fpd_e2e_ckpt_test_fpd}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
USE_WANDB="${USE_WANDB:-false}"
FPD_NUM_SHOWERS="${FPD_NUM_SHOWERS:-10000}"

CMD=(
  accelerate
  launch
  --num_processes=1
  scripts/test_checkpoint.py
  experiment.use_wandb="${USE_WANDB}"
  experiment.output_dir="${OUTPUT_DIR}"
  experiment.run_name="${RUN_NAME}"
  model.model_path="${MODEL_PATH}"
  train.enable_plots=false
  train.enable_fpd=true
  train.fpd_config.num_showers="${FPD_NUM_SHOWERS}"
  train.load_best_model_at_end=false
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
