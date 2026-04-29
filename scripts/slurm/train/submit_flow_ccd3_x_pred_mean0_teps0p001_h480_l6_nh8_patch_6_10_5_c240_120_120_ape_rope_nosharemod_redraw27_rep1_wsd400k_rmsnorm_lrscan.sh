#!/bin/bash

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
SLURM_SCRIPT="scripts/slurm/train/train_flow_ccd3_x_pred_mean0_teps0p001_h480_l6_nh8_patch_6_10_5_c240_120_120_ape_rope_nosharemod_redraw27_rep1_wsd400k_rmsnorm_lrscan_5090_gpudvp.slurm"

cd "${REPO_DIR}"

BASE_RUN_NAME="${BASE_RUN_NAME:-flow_ccd3_x_pred_mean0_teps0p001_h480_l6_nh8_patch_6_10_5_c240_120_120_ape_rope_nosharemod_redraw27_rep1_wsd400k_rmsnorm}"
CONFIG_NAME="${CONFIG_NAME:-experiment/CaloArt_paper/flow_ccd3_x_pred_mean0_teps0p001_h480_l6_nh8_patch_6_10_5_c240_120_120_ape_rope_nosharemod_redraw27_rep1_wsd400k_rmsnorm}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
USE_WANDB="${USE_WANDB:-true}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-150000}"
VALID_STEPS="${VALID_STEPS:-2500}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-25000}"
TEST_STEPS="${TEST_STEPS:-25000}"
LR_VALUES="${LR_VALUES:-0.00055 0.00060 0.00065 0.00070 0.00075 0.00080}"

steps_k=$((TRAIN_MAX_STEPS / 1000))

format_lr_tag() {
  local lr="$1"
  local digits="${lr#0.000}"
  local major="${digits:0:1}"
  local minor="${digits:1:1}"
  if [[ "${minor}" == "0" ]]; then
    printf 'lr%se4' "${major}"
  else
    printf 'lr%sp%se4' "${major}" "${minor}"
  fi
}

for lr in ${LR_VALUES}; do
  lr_tag="$(format_lr_tag "${lr}")"
  run_name="${BASE_RUN_NAME}_${lr_tag}_scan${steps_k}k"

  echo "Submitting ${run_name} with lr=${lr}"
  RUN_NAME="${run_name}" \
  CONFIG_NAME="${CONFIG_NAME}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  USE_WANDB="${USE_WANDB}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  TRAIN_LEARNING_RATE="${lr}" \
  TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS}" \
  VALID_STEPS="${VALID_STEPS}" \
  LOGGING_STEPS="${LOGGING_STEPS}" \
  SAVE_STEPS="${SAVE_STEPS}" \
  TEST_STEPS="${TEST_STEPS}" \
  bash scripts/submit_slurm.sh "${SLURM_SCRIPT}"
done
