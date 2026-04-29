#!/bin/bash
# Tail finetune / convergence probe for the CCD3 h480 mean0_teps0p001 run.
# This script exports EMA weights from an existing checkpoint and starts a new
# run with a low-LR WSD schedule so val loss can be used as a cheap probe.
#
# Usage:
#   sbatch scripts/slurm/train/train_flow_ccd3_x_pred_mean0_teps0p001_h480_l6_nh8_patch_6_16_3_c192_96_96_ape_rope_nosharemod_redraw27_tail_valprobe_5090_gpudvp.slurm
#
# Common overrides:
#   RUN_NAME=... LEARNING_RATE=5e-5 MAX_STEPS=80000 sbatch ...
#   NUM_PROCESSES=2 sbatch --gres=gpu:a800:2 --cpus-per-task=4 ...

#SBATCH --partition=gpu
#SBATCH --qos=gpudvp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=ccd3x480tail
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
SOURCE_EXPERIMENT_DIR="${SOURCE_EXPERIMENT_DIR:-/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/CaloArt_paper/flow_ccd3_x_pred_mean0_teps0p001_h480_l6_nh8_patch_6_16_3_c192_96_96_ape_rope_nosharemod_redraw27}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_last}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SOURCE_EXPERIMENT_DIR}/checkpoints/${CHECKPOINT_NAME}}"
EXPORTED_MODEL_PATH="${EXPORTED_MODEL_PATH:-${CHECKPOINT_DIR}/ema_export_tail_valprobe.pt}"

RUN_NAME="${RUN_NAME:-flow_ccd3_x_pred_mean0_teps0p001_h480_l6_nh8_patch_6_16_3_c192_96_96_ape_rope_nosharemod_redraw27_tail_valprobe_lr1e4_wsd60k}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
CONFIG_NAME="${CONFIG_NAME:-experiment/CaloArt_paper/flow_ccd3_x_pred_mean0_teps0p001_h480_l6_nh8_patch_6_16_3_c192_96_96_ape_rope_nosharemod_redraw27}"
USE_WANDB="${USE_WANDB:-true}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_STEPS="${MAX_STEPS:-60000}"
VALID_STEPS="${VALID_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"
WSD_WARMUP_STEPS="${WSD_WARMUP_STEPS:-0}"
WSD_STABLE_STEPS="${WSD_STABLE_STEPS:-20000}"
WSD_DECAY_STEPS="${WSD_DECAY_STEPS:-40000}"
WSD_MIN_LR_RATIO="${WSD_MIN_LR_RATIO:-0.2}"
EMA_POWER="${EMA_POWER:-0.7}"
EMA_MAX_VALUE="${EMA_MAX_VALUE:-0.9999}"
FORCE_EXPORT="${FORCE_EXPORT:-false}"

echo "Starting CCD3 h480 tail val-probe on $(hostname)"
echo "source_experiment_dir=${SOURCE_EXPERIMENT_DIR}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "exported_model_path=${EXPORTED_MODEL_PATH}"
echo "run_name=${RUN_NAME}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint dir not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

if [[ "${FORCE_EXPORT}" == "true" || ! -f "${EXPORTED_MODEL_PATH}" ]]; then
  echo "Exporting EMA weights from ${CHECKPOINT_DIR}"
  python scripts/export_checkpoint.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_path "${EXPORTED_MODEL_PATH}" \
    --use_ema
else
  echo "Reusing existing exported EMA weights at ${EXPORTED_MODEL_PATH}"
fi

CMD=(
  accelerate
  launch
  --num_machines=1
  --num_processes="${NUM_PROCESSES}"
  --mixed_precision=no
  --dynamo_backend=no
)

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  CMD+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT}")
fi

CMD+=(
  scripts/train.py
  --config-name
  "${CONFIG_NAME}"
  experiment.use_wandb="${USE_WANDB}"
  experiment.output_dir="${OUTPUT_DIR}"
  experiment.run_name="${RUN_NAME}"
  model.model_path="${EXPORTED_MODEL_PATH}"
  train.learning_rate="${LEARNING_RATE}"
  train.max_steps="${MAX_STEPS}"
  train.valid_steps="${VALID_STEPS}"
  train.save_steps="${SAVE_STEPS}"
  train.logging_steps="${LOGGING_STEPS}"
  train.enable_plots=false
  train.enable_fpd=false
  train.save_test_model_artifact=false
  train.save_generated=false
  train.test_strategy=no
  train.lr_scheduler_args.lr_scheduler_type=wsd
  ~train.lr_scheduler_args.num_training_steps
  train.lr_scheduler_args.num_warmup_steps="${WSD_WARMUP_STEPS}"
  +train.lr_scheduler_args.num_stable_steps="${WSD_STABLE_STEPS}"
  +train.lr_scheduler_args.num_decay_steps="${WSD_DECAY_STEPS}"
  +train.lr_scheduler_args.min_lr_ratio="${WSD_MIN_LR_RATIO}"
  train.ema_scheduler_args.power="${EMA_POWER}"
  train.ema_scheduler_args.max_value="${EMA_MAX_VALUE}"
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
