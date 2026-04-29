#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudvp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=ccd3_m08_h480
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting flow_ccd3 x_pred_mean08_teps0p001 h480_l6_nh8 patch_6_10_5 redraw27 rep1 on 1x5090 (gpudvp) $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-flow_ccd3_x_pred_mean08_teps0p001_h480_l6_nh8_patch_6_10_5_c240_120_120_ape_rope_nosharemod_redraw27_rep1}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
CONFIG_NAME="${CONFIG_NAME:-experiment/CaloArt_paper/flow_ccd3_x_pred_mean08_teps0p001_h480_l6_nh8_patch_6_10_5_c240_120_120_ape_rope_nosharemod_redraw27_rep1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
USE_WANDB="${USE_WANDB:-true}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

CMD=(
  accelerate
  launch
  --num_machines=1
  --num_processes="${NUM_PROCESSES}"
  --mixed_precision=no
  --dynamo_backend=no
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
