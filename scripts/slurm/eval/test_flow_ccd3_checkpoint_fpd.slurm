#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --job-name=flow_ccd3_ckpt
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting checkpoint-only CCD3 test/FPD on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/experiments/caloflow_v1/flow_ccd3_h384_l6_nh6_patch_6_10_3_cond192_96_96_testfpd/plots/49999/ema_model_with_config.pt}"
RUN_NAME="${RUN_NAME:-flow_ccd3_h384_l6_nh6_patch_6_10_3_cond192_96_96_testfpd_ckpt_test}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
USE_WANDB="${USE_WANDB:-false}"
TEST_NUM_SHOWERS="${TEST_NUM_SHOWERS:-100000}"
FPD_CUT="${FPD_CUT:-0.015151515151515152}"
SAVE_GENERATED="${SAVE_GENERATED:-false}"
SAVE_TEST_MODEL_ARTIFACT="${SAVE_TEST_MODEL_ARTIFACT:-true}"
ENABLE_PLOTS="${ENABLE_PLOTS:-true}"
ENABLE_FPD="${ENABLE_FPD:-true}"

CMD=(
  accelerate
  launch
  --num_processes=1
  scripts/test_checkpoint.py
  experiment.use_wandb="${USE_WANDB}"
  experiment.output_dir="${OUTPUT_DIR}"
  experiment.run_name="${RUN_NAME}"
  model.model_path="${MODEL_PATH}"
  +train.enable_plots="${ENABLE_PLOTS}"
  +train.enable_fpd="${ENABLE_FPD}"
  +train.test_num_showers="${TEST_NUM_SHOWERS}"
  +train.save_test_model_artifact="${SAVE_TEST_MODEL_ARTIFACT}"
  +train.save_generated="${SAVE_GENERATED}"
  +train.fpd_config.cut="${FPD_CUT}"
  train.load_best_model_at_end=false
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
