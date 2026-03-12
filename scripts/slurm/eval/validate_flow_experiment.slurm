#!/bin/bash
#SBATCH --job-name=validate_flow
#SBATCH --partition=gpu
#SBATCH --qos=gpuintera
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
EXPERIMENT_DIR="${EXPERIMENT_DIR:?EXPERIMENT_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_best}"
EXPORTED_MODEL_PATH="${EXPORTED_MODEL_PATH:-${EXPERIMENT_DIR}/validated_model.pt}"
VALIDATE_CONFIG="${VALIDATE_CONFIG:-experiment/CaloChallenge/validate/ccd2_flow}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
SAMPLING_SOLVER="${SAMPLING_SOLVER:-heun}"
ARCH_TARGET="${ARCH_TARGET:-}"

mkdir -p "${REPO_DIR}/logs/slurm"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow

cd "${REPO_DIR}"

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "EXPERIMENT_DIR=${EXPERIMENT_DIR}"
echo "CHECKPOINT_NAME=${CHECKPOINT_NAME}"
echo "EXPORTED_MODEL_PATH=${EXPORTED_MODEL_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "VALIDATE_CONFIG=${VALIDATE_CONFIG}"
echo "SAMPLING_STEPS=${SAMPLING_STEPS}"
echo "SAMPLING_SOLVER=${SAMPLING_SOLVER}"
echo "ARCH_TARGET=${ARCH_TARGET:-saved-config}"
nvidia-smi

python scripts/export_checkpoint.py \
  --checkpoint_dir "${EXPERIMENT_DIR}/checkpoints/${CHECKPOINT_NAME}" \
  --output_path "${EXPORTED_MODEL_PATH}" \
  --use_ema

CMD=(
  python
  scripts/validate.py
  --config-name "${VALIDATE_CONFIG}"
  model.model_path="${EXPORTED_MODEL_PATH}"
  validate.output_dir="${OUTPUT_DIR}"
  sampling.steps="${SAMPLING_STEPS}"
  sampling.solver="${SAMPLING_SOLVER}"
)

if [[ -n "${ARCH_TARGET}" ]]; then
  CMD+=(+model.architecture.target="${ARCH_TARGET}")
fi

printf 'Running validate command:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Done at $(date)."
