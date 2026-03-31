#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudvp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --job-name=edm_d4s_paper
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting edm_ccd2_diffusion4sim_paper on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-edm}"
PROJECT_NAME="${PROJECT_NAME:-ccd2_calodit_cont}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_DIR:-./results}/experiments}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
USE_WANDB="${USE_WANDB:-true}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

CMD=(
  accelerate
  launch
  --num_processes="${NUM_PROCESSES}"
)

if [ "${NUM_PROCESSES}" -gt 1 ]; then
  CMD+=(--multi_gpu)
fi

CMD+=(
  --main_process_port="${MAIN_PROCESS_PORT}"
  scripts/train.py
  --config-name
  experiment/CaloChallenge/edm_ccd2_diffusion4sim_paper
  experiment.project_name="${PROJECT_NAME}"
  experiment.output_dir="${OUTPUT_DIR}"
  experiment.run_name="${RUN_NAME}"
  experiment.use_wandb="${USE_WANDB}"
)

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
