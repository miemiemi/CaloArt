#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudvp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=ccd3_smc5
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting flow_ccd3 snrscan -0.5 teps0p001 h480_l6_nh8 patch_6_10_5 classicdit scratch on 2x5090 (gpudvp) $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-flow_ccd3_x_pred_snrscan_m05_teps0p001_h480_l6_nh8_patch_6_10_5_c240_120_120_ape_rope_nosharemod_redraw27_rep1_wsd400k_rmsnorm_lr8e4_plots1000_classicdit_scratch}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
CONFIG_NAME="${CONFIG_NAME:-experiment/CaloArt_paper/flow_ccd3_x_pred_snrscan_m05_teps0p001_h480_l6_nh8_patch_6_10_5_c240_120_120_ape_rope_nosharemod_redraw27_rep1_wsd400k_rmsnorm_lr8e4_plots1000_classicdit_scratch}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
USE_WANDB="${USE_WANDB:-true}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

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

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
