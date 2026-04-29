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
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"
TEST_STEP_OVERRIDE="${TEST_STEP_OVERRIDE:-}"
SAMPLING_STEPS="${SAMPLING_STEPS:-}"
SAMPLING_SOLVER="${SAMPLING_SOLVER:-}"
TEST_CONDITIONS_OVERRIDE="${TEST_CONDITIONS_OVERRIDE:-}"
REJECT_REDRAW="${REJECT_REDRAW:-}"
REJECT_REDRAW_MAX_RATIO="${REJECT_REDRAW_MAX_RATIO:-}"
REJECT_REDRAW_REJECT_NONFINITE="${REJECT_REDRAW_REJECT_NONFINITE:-}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
USE_LOCAL_SSD="${USE_LOCAL_SSD:-false}"
LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-/scratch/${USER}/caloflow_job_${SLURM_JOB_ID}}"
REMOVE_LOCAL_STAGE_AFTER_SYNC="${REMOVE_LOCAL_STAGE_AFTER_SYNC:-true}"
export CALOFLOW_H5_GZIP_THREADS="${CALOFLOW_H5_GZIP_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export CALOFLOW_H5_GZIP_LEVEL="${CALOFLOW_H5_GZIP_LEVEL:-9}"

FINAL_OUTPUT_DIR="${OUTPUT_DIR}"
RUN_OUTPUT_DIR="${FINAL_OUTPUT_DIR}"
SYNCED_LOCAL_STAGE=0

sync_local_stage_back() {
  if [[ "${USE_LOCAL_SSD}" != "true" ]]; then
    return 0
  fi

  if [[ "${SYNCED_LOCAL_STAGE}" -eq 1 ]]; then
    return 0
  fi

  if [[ ! -d "${RUN_OUTPUT_DIR}" ]]; then
    echo "Local SSD stage directory ${RUN_OUTPUT_DIR} does not exist yet, skipping sync."
    return 0
  fi

  mkdir -p "${FINAL_OUTPUT_DIR}"
  echo "Syncing staged outputs back to ${FINAL_OUTPUT_DIR}"
  rsync -a "${RUN_OUTPUT_DIR}/" "${FINAL_OUTPUT_DIR}/"
  SYNCED_LOCAL_STAGE=1

  if [[ "${REMOVE_LOCAL_STAGE_AFTER_SYNC}" == "true" ]]; then
    echo "Removing local SSD stage ${RUN_OUTPUT_DIR}"
    rm -rf "${RUN_OUTPUT_DIR}"
  fi
}

if [[ "${USE_LOCAL_SSD}" == "true" ]]; then
  RUN_OUTPUT_DIR="${LOCAL_SSD_ROOT}/experiments"
  mkdir -p "${RUN_OUTPUT_DIR}"
  echo "Using local SSD staging under ${RUN_OUTPUT_DIR}"
  echo "Final outputs will be synced back to ${FINAL_OUTPUT_DIR}"
  trap sync_local_stage_back EXIT
fi

echo "CALOFLOW_H5_GZIP_THREADS=${CALOFLOW_H5_GZIP_THREADS}"
echo "CALOFLOW_H5_GZIP_LEVEL=${CALOFLOW_H5_GZIP_LEVEL}"

CMD=(
  accelerate
  launch
  --num_processes="${NUM_PROCESSES}"
  scripts/test_checkpoint.py
  experiment.use_wandb="${USE_WANDB}"
  experiment.output_dir="${RUN_OUTPUT_DIR}"
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

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  CMD=(
    accelerate
    launch
    --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --main_process_port="${MAIN_PROCESS_PORT}"
    scripts/test_checkpoint.py
    experiment.use_wandb="${USE_WANDB}"
    experiment.output_dir="${RUN_OUTPUT_DIR}"
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
fi

if [[ -n "${GLOBAL_BATCH_SIZE}" ]]; then
  CMD+=("train.global_batch_size=${GLOBAL_BATCH_SIZE}")
fi

if [[ -n "${TEST_STEP_OVERRIDE}" ]]; then
  CMD+=("+train.test_step_override=${TEST_STEP_OVERRIDE}")
fi

if [[ -n "${SAMPLING_STEPS}" ]]; then
  CMD+=("sampling.steps=${SAMPLING_STEPS}")
fi

if [[ -n "${SAMPLING_SOLVER}" ]]; then
  CMD+=("sampling.solver=${SAMPLING_SOLVER}")
fi

if [[ -n "${TEST_CONDITIONS_OVERRIDE}" ]]; then
  CMD+=("+train.test_conditions=${TEST_CONDITIONS_OVERRIDE}")
fi

if [[ -n "${REJECT_REDRAW}" ]]; then
  CMD+=("+sampling.reject_redraw=${REJECT_REDRAW}")
fi

if [[ -n "${REJECT_REDRAW_MAX_RATIO}" ]]; then
  CMD+=("+sampling.reject_redraw_max_ratio=${REJECT_REDRAW_MAX_RATIO}")
fi

if [[ -n "${REJECT_REDRAW_REJECT_NONFINITE}" ]]; then
  CMD+=("+sampling.reject_redraw_reject_nonfinite=${REJECT_REDRAW_REJECT_NONFINITE}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

sync_local_stage_back
