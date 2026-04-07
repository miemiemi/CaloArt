#!/bin/bash
#SBATCH --job-name=plot_gen_h5
#SBATCH --partition=cpu
#SBATCH --qos=cpunormal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow
cd "${REPO_DIR}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR="${REPO_DIR}/logs/tmp/${SLURM_JOB_ID}"
export MPLCONFIGDIR="${REPO_DIR}/logs/matplotlib/${SLURM_JOB_ID}"
export MPLBACKEND=Agg

mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

GENERATED_FILE="${GENERATED_FILE:?GENERATED_FILE is required}"
REFERENCE_FILE="${REFERENCE_FILE:-}"
NUM_SHOWERS="${NUM_SHOWERS:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
KEEP_PREVIOUS="${KEEP_PREVIOUS:-false}"

CMD=(
  python
  -u
  scripts/plot_existing_generated.py
  --generated-file "${GENERATED_FILE}"
  --num-showers "${NUM_SHOWERS}"
)

if [[ -n "${REFERENCE_FILE}" ]]; then
  CMD+=(--reference-file "${REFERENCE_FILE}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ "${KEEP_PREVIOUS}" == "true" ]]; then
  CMD+=(--keep-previous)
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
