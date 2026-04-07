#!/bin/bash
#SBATCH --job-name=recover_wandb_cpu
#SBATCH --partition=cpu
#SBATCH --qos=cpunormal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
ENTITY="${ENTITY:-miemimail2020-Institute of High Energy Physics}"
PROJECT="${PROJECT:-CaloArt_paper}"
CHUNK_SIZE="${CHUNK_SIZE:-100000}"
POLL_SECONDS="${POLL_SECONDS:-20}"
POLL_TIMEOUT_SECONDS="${POLL_TIMEOUT_SECONDS:-1800}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5000}"
EXPERIMENT_DIRS="${EXPERIMENT_DIRS:-/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/CaloArt_paper/flow_ccd3_x_pred_mean08_teps0p001_h384_l6_nh6_patch_6_16_3_c192_96_96_ape_rope_nosharemod_redraw27_rep1|/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/CaloArt_paper/flow_ccd3_x_pred_mean0_h384_l6_nh6_patch_6_16_3_c192_96_96_ape_rope_nosharemod_redraw27|/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/CaloArt_paper/flow_ccd3_h384_l6_nh6_patch_6_10_3_cond192_96_96_ape_rope}"

mkdir -p "${REPO_DIR}/logs/slurm"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow
cd "${REPO_DIR}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB_DISABLE_GIT=true
export WANDB_DISABLE_CODE=true
export WANDB_SAVE_CODE=false
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "ENTITY=${ENTITY}"
echo "PROJECT=${PROJECT}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "POLL_SECONDS=${POLL_SECONDS}"
echo "POLL_TIMEOUT_SECONDS=${POLL_TIMEOUT_SECONDS}"
echo "PROGRESS_EVERY=${PROGRESS_EVERY}"

IFS='|' read -r -a EXPERIMENT_DIRS_ARR <<< "${EXPERIMENT_DIRS}"
printf 'Experiments:\n'
for exp_dir in "${EXPERIMENT_DIRS_ARR[@]}"; do
  echo "  ${exp_dir}"
done

CMD=(
  python
  scripts/recover_wandb_runs_batch.py
  --entity "${ENTITY}"
  --project "${PROJECT}"
  --chunk-size "${CHUNK_SIZE}"
  --poll-seconds "${POLL_SECONDS}"
  --poll-timeout-seconds "${POLL_TIMEOUT_SECONDS}"
  --progress-every "${PROGRESS_EVERY}"
)
CMD+=("${EXPERIMENT_DIRS_ARR[@]}")

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Done at $(date)."
