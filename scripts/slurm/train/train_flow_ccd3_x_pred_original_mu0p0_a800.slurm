#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudvp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --job-name=ccd3_xmu0p0
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
CONFIG_NAME="experiment/CaloChallenge/flow_ccd3_energy_only_x_pred_jit_like_patch_6_16_3_redraw27"
RUN_NAME="flow_ccd3_x_pred_original_h384_l6_nh6_patch_6_16_3_redraw27_mu0p0"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
USE_WANDB="${USE_WANDB:-true}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
TEST_NUM_SHOWERS="${TEST_NUM_SHOWERS:-1000}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

echo "Starting CCD3 x_pred original mu=0.0 on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "RUN_NAME=${RUN_NAME}"

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
  method.init_args.logit_normal_mean="0.0"
  train.enable_plots=true
  train.enable_fpd=false
  train.test_num_showers="${TEST_NUM_SHOWERS}"
  train.save_test_model_artifact=true
  train.save_generated=false
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
