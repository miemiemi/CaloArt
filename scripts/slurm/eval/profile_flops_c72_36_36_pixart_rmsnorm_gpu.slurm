#!/bin/bash
#SBATCH --job-name=prof_flops
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting GPU FLOPs profiling on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

CONFIG_NAME="${CONFIG_NAME:-experiment/CaloChallenge/flow_ccd2_pred_v_all_lightingdit_freatures_c72_36_36_pixart_sharemod_heun32_latest}"
PROFILE_DEVICE="${PROFILE_DEVICE:-cuda}"
PROFILE_OUTPUT_JSON="${PROFILE_OUTPUT_JSON:-logs/profile_flops_c72_36_36_pixart_rmsnorm_gpu.json}"
PROFILE_DEEPSPEED_REPORT_PATH="${PROFILE_DEEPSPEED_REPORT_PATH:-logs/profile_flops_c72_36_36_pixart_rmsnorm_gpu_deepspeed.txt}"
PROFILE_DEEPSPEED_PROFILE_STEP="${PROFILE_DEEPSPEED_PROFILE_STEP:-1}"
PROFILE_DEEPSPEED_MODULE_DEPTH="${PROFILE_DEEPSPEED_MODULE_DEPTH:--1}"
PROFILE_DEEPSPEED_TOP_MODULES="${PROFILE_DEEPSPEED_TOP_MODULES:-3}"
PROFILE_DEEPSPEED_DETAILED="${PROFILE_DEEPSPEED_DETAILED:-1}"

CMD=(
  /aifs/user/home/huangzhengkun/miniconda3/envs/CaloFlow/bin/python
  scripts/profile_flops.py
  --config-name
  "${CONFIG_NAME}"
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

PROFILE_DEVICE="${PROFILE_DEVICE}" \
PROFILE_OUTPUT_JSON="${PROFILE_OUTPUT_JSON}" \
PROFILE_DEEPSPEED_REPORT_PATH="${PROFILE_DEEPSPEED_REPORT_PATH}" \
PROFILE_DEEPSPEED_PROFILE_STEP="${PROFILE_DEEPSPEED_PROFILE_STEP}" \
PROFILE_DEEPSPEED_MODULE_DEPTH="${PROFILE_DEEPSPEED_MODULE_DEPTH}" \
PROFILE_DEEPSPEED_TOP_MODULES="${PROFILE_DEEPSPEED_TOP_MODULES}" \
PROFILE_DEEPSPEED_DETAILED="${PROFILE_DEEPSPEED_DETAILED}" \
"${CMD[@]}"
