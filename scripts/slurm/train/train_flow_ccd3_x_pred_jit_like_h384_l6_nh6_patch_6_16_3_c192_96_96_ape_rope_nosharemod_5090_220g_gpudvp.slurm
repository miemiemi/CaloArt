#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudvp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --job-name=flow_ccd3_x8
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting flow_ccd3 x_pred jit_like patch_6_16_3 on 2x5090 (220G) $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-flow_ccd3_x_pred_jit_like_h384_l6_nh6_patch_6_16_3_c192_96_96_ape_rope_nosharemod_5090_220g}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
CONFIG_NAME="${CONFIG_NAME:-experiment/CaloChallenge/flow_ccd3_x_pred_patch_6_16_3_h384}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
USE_WANDB="${USE_WANDB:-true}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
ENABLE_PLOTS="${ENABLE_PLOTS:-true}"
ENABLE_FPD="${ENABLE_FPD:-true}"
TEST_NUM_SHOWERS="${TEST_NUM_SHOWERS:-100000}"
FPD_CUT="${FPD_CUT:-0.015151515151515152}"
SAVE_GENERATED="${SAVE_GENERATED:-true}"
SAVE_TEST_MODEL_ARTIFACT="${SAVE_TEST_MODEL_ARTIFACT:-true}"

CMD=(
  accelerate
  launch
  --multi_gpu
  --num_processes="${NUM_PROCESSES}"
  --main_process_port="${MAIN_PROCESS_PORT}"
  scripts/train.py
  --config-name
  "${CONFIG_NAME}"
  experiment.use_wandb="${USE_WANDB}"
  experiment.output_dir="${OUTPUT_DIR}"
  experiment.run_name="${RUN_NAME}"
  train.enable_plots="${ENABLE_PLOTS}"
  train.enable_fpd="${ENABLE_FPD}"
  train.test_num_showers="${TEST_NUM_SHOWERS}"
  train.save_test_model_artifact="${SAVE_TEST_MODEL_ARTIFACT}"
  train.save_generated="${SAVE_GENERATED}"
  train.fpd_config.cut="${FPD_CUT}"
)

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=("+train.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
