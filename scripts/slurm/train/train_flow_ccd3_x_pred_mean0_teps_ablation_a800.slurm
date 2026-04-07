#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudvp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --job-name=ccd3_xm0_eps
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting CCD3 x_pred_mean0 t_eps ablation on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-flow_ccd3_x_pred_mean0_h384_l6_nh6_patch_6_16_3_c192_96_96_ape_rope_nosharemod_redraw27}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
CONFIG_NAME="${CONFIG_NAME:-experiment/CaloArt_paper/flow_ccd3_x_pred_mean0_h384_l6_nh6_patch_6_16_3_c192_96_96_ape_rope_nosharemod_redraw27}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
USE_WANDB="${USE_WANDB:-true}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

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
)

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=("+train.resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}")
fi

if [[ -n "${EXTRA_OVERRIDES}" ]]; then
  # Allow method/config swaps without creating more wrappers.
  read -r -a EXTRA_ARGS <<< "${EXTRA_OVERRIDES}"
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
