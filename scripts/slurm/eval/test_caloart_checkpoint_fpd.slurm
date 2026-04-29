#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --job-name=caloart_fpd
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/aifs/user/home/huangzhengkun/work/repositories/CaloFlow}"

echo "Starting CaloArt checkpoint-only CCD2 FPD test on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

TARGET_EXPERIMENT="${TARGET_EXPERIMENT:-${REPO_DIR}/experiments/CaloArt_paper/flow_ccd2_pred_v_loss_v_logit_normal_cond_energy_log10_h384_l6_nh6_patch_3_4_3_pixart_fpd100k}"
MODEL_PATH="${MODEL_PATH:-${TARGET_EXPERIMENT}/final_model.pt}"
RUN_NAME="${RUN_NAME:-flow_ccd2_pred_v_loss_v_logit_normal_cond_energy_log10_h384_l6_nh6_patch_3_4_3_pixart_fpd100k_ckpt_fpd}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
USE_WANDB="${USE_WANDB:-false}"
TEST_NUM_SHOWERS="${TEST_NUM_SHOWERS:-100000}"
ENABLE_PLOTS="${ENABLE_PLOTS:-false}"
ENABLE_FPD="${ENABLE_FPD:-true}"
SAVE_GENERATED="${SAVE_GENERATED:-true}"
SAVE_TEST_MODEL_ARTIFACT="${SAVE_TEST_MODEL_ARTIFACT:-true}"

CMD=(
  accelerate
  launch
  --num_processes=1
  scripts/test_checkpoint.py
  experiment.use_wandb="${USE_WANDB}"
  experiment.output_dir="${OUTPUT_DIR}"
  experiment.run_name="${RUN_NAME}"
  model.model_path="${MODEL_PATH}"
  ++train.enable_plots="${ENABLE_PLOTS}"
  ++train.enable_fpd="${ENABLE_FPD}"
  ++train.test_num_showers="${TEST_NUM_SHOWERS}"
  ++train.save_test_model_artifact="${SAVE_TEST_MODEL_ARTIFACT}"
  ++train.save_generated="${SAVE_GENERATED}"
  ++train.fpd_config.save_generated="${SAVE_GENERATED}"
  train.load_best_model_at_end=false
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
