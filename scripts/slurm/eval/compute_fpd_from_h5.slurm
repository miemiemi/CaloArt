#!/bin/bash
#SBATCH --job-name=fpd_h5
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow
cd "${REPO_DIR}"

GENERATED_FILE="${GENERATED_FILE:?GENERATED_FILE is required}"
REFERENCE_FILE="${REFERENCE_FILE:?REFERENCE_FILE is required}"
GEOMETRY="${GEOMETRY:-CCD2}"
NUM_SHOWERS="${NUM_SHOWERS:-}"
NOISE_LEVEL="${NOISE_LEVEL:-5.0e-07}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DATASET_NAME="${DATASET_NAME:-}"
COMPUTE_KPD="${COMPUTE_KPD:-false}"

CMD=(
  python
  scripts/compute_fpd_from_h5.py
  --generated-file "${GENERATED_FILE}"
  --reference-file "${REFERENCE_FILE}"
  --geometry "${GEOMETRY}"
  --noise-level "${NOISE_LEVEL}"
)

if [[ -n "${NUM_SHOWERS}" ]]; then
  CMD+=(--num-showers "${NUM_SHOWERS}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${DATASET_NAME}" ]]; then
  CMD+=(--dataset-name "${DATASET_NAME}")
fi
if [[ "${COMPUTE_KPD}" == "true" ]]; then
  CMD+=(--compute-kpd)
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
