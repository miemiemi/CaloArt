#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=95G
#SBATCH --time=24:00:00
#SBATCH --job-name=ccd3_h512_a100
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

echo "Starting CCD3 ref generation on A100 at $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

STEP_LIST="${STEP_LIST:-299994}"
TEST_NUM_SHOWERS="${TEST_NUM_SHOWERS:-100000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-100}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${REPO_DIR}/experiments/CaloArt_paper/flow_ccd3_x_pred_mean0_teps0p001_h512_l8_nh8_patch_6_16_5_c192_96_96_ape_rope_nosharemod_redraw27}"

export STEP_LIST
export TEST_NUM_SHOWERS
export GLOBAL_BATCH_SIZE
export EXPERIMENT_DIR

bash "${REPO_DIR}/scripts/slurm/eval/generate_flow_ccd3_ref_samples_serial.slurm"
