#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpudvp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --job-name=flow_ccd3_h384
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting flow_ccd3_energy_only on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

RUN_NAME="${RUN_NAME:-flow_ccd3_energy_only}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments}"
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
  experiment/CaloChallenge/flow_ccd3_energy_only
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
