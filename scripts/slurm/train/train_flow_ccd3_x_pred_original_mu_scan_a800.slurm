#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudvp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --job-name=ccd3_xmu
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%A_%a.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%A_%a.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
CONFIG_NAME="${CONFIG_NAME:-experiment/CaloChallenge/flow_ccd3_energy_only_x_pred_jit_like_patch_6_16_3_redraw27}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
USE_WANDB="${USE_WANDB:-true}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
TEST_NUM_SHOWERS="${TEST_NUM_SHOWERS:-1000}"
ENABLE_PLOTS="${ENABLE_PLOTS:-true}"
ENABLE_FPD="${ENABLE_FPD:-false}"
SAVE_TEST_MODEL_ARTIFACT="${SAVE_TEST_MODEL_ARTIFACT:-true}"
SAVE_GENERATED="${SAVE_GENERATED:-false}"
MAX_STEPS="${MAX_STEPS:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
MU_VALUES_CSV="${MU_VALUES_CSV:-0.0,-0.4,-0.8,-1.2}"

IFS=',' read -r -a MU_VALUES <<< "${MU_VALUES_CSV}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MU_VALUE="${MU_VALUES[$TASK_ID]}"
MU_LABEL="$(python - "${MU_VALUE}" <<'PY'
import sys
value = sys.argv[1].strip()
label = value.replace("-", "m").replace(".", "p")
if not label.startswith("m"):
    label = label
print(f"mu{label}")
PY
)"

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-flow_ccd3_x_pred_original_h384_l6_nh6_patch_6_16_3_redraw27}"
RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${MU_LABEL}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000 + TASK_ID))}"

echo "Starting CCD3 x_pred original mu scan on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "RUN_NAME=${RUN_NAME}"
echo "MU_VALUE=${MU_VALUE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

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
  method.init_args.logit_normal_mean="${MU_VALUE}"
  train.enable_plots="${ENABLE_PLOTS}"
  train.enable_fpd="${ENABLE_FPD}"
  train.test_num_showers="${TEST_NUM_SHOWERS}"
  train.save_test_model_artifact="${SAVE_TEST_MODEL_ARTIFACT}"
  train.save_generated="${SAVE_GENERATED}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  CMD+=(train.max_steps="${MAX_STEPS}")
fi

if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  # Allow optional extra Hydra overrides without editing the script again.
  read -r -a EXTRA_ARGS <<< "${EXTRA_OVERRIDES}"
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
