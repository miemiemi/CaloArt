#!/bin/bash
# CCD3 transfer run on 2x5090 (gpuintera): initialize from the CCD2 transfer
# checkpoint but train with the matched CCD3 transfer recipe.

#SBATCH --partition=gpu
#SBATCH --qos=gpuintera
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --job-name=ccd3_xm0_tr
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting CCD3 transfer x_pred_mean0_teps0p001 h384_l6_nh6 patch_6_16_3 condlog10 on 2x5090 (gpuintera) $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-flow_ccd3_x_pred_mean0_teps0p001_h384_l6_nh6_patch_6_16_3_c192_96_96_ape_rope_nosharemod_redraw27_condlog10_transfer_from_ccd2}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
CONFIG_NAME="${CONFIG_NAME:-experiment/CaloArt_paper/flow_ccd3_x_pred_mean0_teps0p001_h384_l6_nh6_patch_6_16_3_c192_96_96_ape_rope_nosharemod_redraw27_condlog10_transfer_from_ccd2}"
MODEL_PATH="${MODEL_PATH:-/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/.codex/tmp/ccd2_to_ccd3_transfer_init.pt}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
USE_WANDB="${USE_WANDB:-true}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

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
  model.model_path="${MODEL_PATH}"
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
