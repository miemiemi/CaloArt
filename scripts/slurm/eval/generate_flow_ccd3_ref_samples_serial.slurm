#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --job-name=flow_ccd3_seq
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
DEFAULT_EXPERIMENT_DIR="${REPO_DIR}/experiments/caloflow_v1/flow_ccd3_x_pred_jit_like_h384_l6_nh6_patch_6_16_3_c192_96_96_ape_rope_nosharemod_5090_80g_redraw27_um04"

echo "Starting serial CCD3 checkpoint generation on $(hostname)"
CONDA_SH=""
if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_SH="$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
  CONDA_SH="$(dirname "$(dirname "${CONDA_BIN}")")/etc/profile.d/conda.sh"
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  for candidate in \
    "/aifs/user/home/huangzhengkun/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "/afs/ihep.ac.cn/users/h/huangzhengkun/miniconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "${candidate}" ]]; then
      CONDA_SH="${candidate}"
      break
    fi
  done
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Could not locate conda.sh. Set CONDA_EXE or edit the script for your environment." >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate CaloFlow

cd "${REPO_DIR}"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-${DEFAULT_EXPERIMENT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/experiments}"
MODEL_FILENAME="${MODEL_FILENAME:-ema_model_with_config.pt}"
STEP_MIN="${STEP_MIN:-49999}"
STEP_MAX="${STEP_MAX:-399992}"
STEP_LIST="${STEP_LIST:-}"
USE_WANDB="${USE_WANDB:-false}"
TEST_NUM_SHOWERS="${TEST_NUM_SHOWERS:-}"
SAVE_GENERATED="${SAVE_GENERATED:-true}"
SAVE_TEST_MODEL_ARTIFACT="${SAVE_TEST_MODEL_ARTIFACT:-false}"
ENABLE_PLOTS="${ENABLE_PLOTS:-false}"
ENABLE_FPD="${ENABLE_FPD:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
SAMPLING_STEPS="${SAMPLING_STEPS:-}"
SAMPLING_SOLVER="${SAMPLING_SOLVER:-}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"
TEST_CONDITIONS_OVERRIDE="${TEST_CONDITIONS_OVERRIDE:-}"
TEST_OUTPUT_SUBDIR="${TEST_OUTPUT_SUBDIR:-}"
REJECT_REDRAW="${REJECT_REDRAW:-}"
REJECT_REDRAW_MAX_RATIO="${REJECT_REDRAW_MAX_RATIO:-}"
REJECT_REDRAW_REJECT_NONFINITE="${REJECT_REDRAW_REJECT_NONFINITE:-}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
PORT_SEED="${SLURM_JOB_ID:-$$}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + PORT_SEED % 20000))}"
export CALOFLOW_H5_GZIP_THREADS="${CALOFLOW_H5_GZIP_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export CALOFLOW_H5_GZIP_LEVEL="${CALOFLOW_H5_GZIP_LEVEL:-9}"

if [[ ! -d "${EXPERIMENT_DIR}" ]]; then
  echo "Experiment directory does not exist: ${EXPERIMENT_DIR}" >&2
  exit 1
fi

declare -a STEPS
if [[ -n "${STEP_LIST}" ]]; then
  STEP_LIST_CLEAN="${STEP_LIST//,/ }"
  read -r -a STEPS <<< "${STEP_LIST_CLEAN}"
else
  mapfile -t STEPS < <(
    find "${EXPERIMENT_DIR}/plots" -mindepth 2 -maxdepth 2 -type f -name "${MODEL_FILENAME}" \
      | awk -F/ '{print $(NF-1)}' \
      | awk -v step_min="${STEP_MIN}" -v step_max="${STEP_MAX}" '
          $0 ~ /^[0-9]+$/ && $0 >= step_min && $0 <= step_max { print $0 }
        ' \
      | sort -n
  )
fi

if [[ "${#STEPS[@]}" -eq 0 ]]; then
  echo "No checkpoints matched in ${EXPERIMENT_DIR}/plots for ${MODEL_FILENAME}" >&2
  exit 1
fi

echo "EXPERIMENT_DIR=${EXPERIMENT_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MODEL_FILENAME=${MODEL_FILENAME}"
echo "STEPS=${STEPS[*]}"
echo "TEST_NUM_SHOWERS=${TEST_NUM_SHOWERS:-<saved-config default>}"
echo "ENABLE_PLOTS=${ENABLE_PLOTS}"
echo "ENABLE_FPD=${ENABLE_FPD}"
echo "SAVE_GENERATED=${SAVE_GENERATED}"
echo "SKIP_EXISTING=${SKIP_EXISTING}"
echo "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
echo "TEST_OUTPUT_SUBDIR=${TEST_OUTPUT_SUBDIR:-<none>}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "CALOFLOW_H5_GZIP_THREADS=${CALOFLOW_H5_GZIP_THREADS}"
echo "CALOFLOW_H5_GZIP_LEVEL=${CALOFLOW_H5_GZIP_LEVEL}"

failures=()

for step in "${STEPS[@]}"; do
  MODEL_PATH="${EXPERIMENT_DIR}/plots/${step}/${MODEL_FILENAME}"
  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Missing model artifact for step ${step}: ${MODEL_PATH}" >&2
    failures+=("${step}:missing_model")
    if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
      continue
    fi
    exit 1
  fi

  if [[ "${SKIP_EXISTING}" == "true" ]]; then
    existing_generated_count="$(find "${EXPERIMENT_DIR}/plots/${step}" -maxdepth 2 -type f -name 'generated.h5' | wc -l)"
    if [[ "${existing_generated_count}" -gt 0 ]]; then
      echo "Skipping step ${step}: found ${existing_generated_count} existing generated.h5 file(s)."
      continue
    fi
  fi

  CMD=(
    accelerate
    launch
    --num_processes="${NUM_PROCESSES}"
    scripts/test_checkpoint.py
    experiment.use_wandb="${USE_WANDB}"
    experiment.output_dir="${OUTPUT_ROOT}"
    model.model_path="${MODEL_PATH}"
    +train.enable_plots="${ENABLE_PLOTS}"
    +train.enable_fpd="${ENABLE_FPD}"
    +train.save_generated="${SAVE_GENERATED}"
    +train.save_test_model_artifact="${SAVE_TEST_MODEL_ARTIFACT}"
    +train.test_step_override="${step}"
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
      experiment.output_dir="${OUTPUT_ROOT}"
      model.model_path="${MODEL_PATH}"
      +train.enable_plots="${ENABLE_PLOTS}"
      +train.enable_fpd="${ENABLE_FPD}"
      +train.save_generated="${SAVE_GENERATED}"
      +train.save_test_model_artifact="${SAVE_TEST_MODEL_ARTIFACT}"
      +train.test_step_override="${step}"
      train.load_best_model_at_end=false
    )
  fi

  if [[ -n "${TEST_NUM_SHOWERS}" ]]; then
    CMD+=("+train.test_num_showers=${TEST_NUM_SHOWERS}")
  fi

  if [[ -n "${SAMPLING_STEPS}" ]]; then
    CMD+=("sampling.steps=${SAMPLING_STEPS}")
  fi

  if [[ -n "${SAMPLING_SOLVER}" ]]; then
    CMD+=("sampling.solver=${SAMPLING_SOLVER}")
  fi

  if [[ -n "${GLOBAL_BATCH_SIZE}" ]]; then
    CMD+=("train.global_batch_size=${GLOBAL_BATCH_SIZE}")
  fi

  if [[ -n "${TEST_CONDITIONS_OVERRIDE}" ]]; then
    CMD+=("+train.test_conditions=${TEST_CONDITIONS_OVERRIDE}")
  fi

  if [[ -n "${TEST_OUTPUT_SUBDIR}" ]]; then
    CMD+=("+train.test_output_subdir=${TEST_OUTPUT_SUBDIR}")
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

  echo
  echo "===== Step ${step} ====="
  printf 'Launching:'
  printf ' %q' "${CMD[@]}"
  echo

  if ! "${CMD[@]}"; then
    echo "Step ${step} failed." >&2
    failures+=("${step}:command_failed")
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      exit 1
    fi
  fi
done

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo
  echo "Finished with failures:"
  printf '  %s\n' "${failures[@]}"
  exit 1
fi

echo
echo "All requested checkpoints completed successfully."
