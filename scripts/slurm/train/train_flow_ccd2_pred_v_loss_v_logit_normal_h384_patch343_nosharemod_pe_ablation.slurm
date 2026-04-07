#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudvp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --job-name=flow_pe_ablate
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%A_%a.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%A_%a.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

CONFIGS=(
  "experiment/CaloArt_paper/flow_ccd2_pred_v_loss_v_logit_normal_cond_energy_log10_h384_l6_nh6_patch_3_4_3_nosharemod_fpd100k_ape_only"
  "experiment/CaloArt_paper/flow_ccd2_pred_v_loss_v_logit_normal_cond_energy_log10_h384_l6_nh6_patch_3_4_3_nosharemod_fpd100k_rope_only"
)

RUN_NAMES=(
  "flow_ccd2_pred_v_loss_v_logit_normal_cond_energy_log10_h384_l6_nh6_patch_3_4_3_nosharemod_fpd100k_ape_only"
  "flow_ccd2_pred_v_loss_v_logit_normal_cond_energy_log10_h384_l6_nh6_patch_3_4_3_nosharemod_fpd100k_rope_only"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG_NAME="${CONFIG_NAME:-${CONFIGS[$TASK_ID]}}"
RUN_NAME="${RUN_NAME:-${RUN_NAMES[$TASK_ID]}}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
USE_WANDB="${USE_WANDB:-true}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000 + TASK_ID))}"

echo "Starting positional encoding ablation on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "USE_WANDB=${USE_WANDB}"

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
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
