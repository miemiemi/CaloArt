#!/bin/bash
#SBATCH --job-name=x_over_einc
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

DATA_FILES="${DATA_FILES:?DATA_FILES is required}"
SAMPLING_FRACTION="${SAMPLING_FRACTION:-0.033}"
CHUNK_SHOWERS="${CHUNK_SHOWERS:-256}"
PER_CHUNK_SAMPLE="${PER_CHUNK_SAMPLE:-20000}"
MAX_POSITIVE_SAMPLE="${MAX_POSITIVE_SAMPLE:-3000000}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
OUTPUT_SCALED_H5="${OUTPUT_SCALED_H5:-}"

read -r -a DATA_FILES_ARR <<< "${DATA_FILES}"

CMD=(
  python
  -u
  scripts/inspect_energy_scaled_voxels.py
  --files
)
CMD+=("${DATA_FILES_ARR[@]}")
CMD+=(--sampling-fraction "${SAMPLING_FRACTION}")
CMD+=(--chunk-showers "${CHUNK_SHOWERS}")
CMD+=(--per-chunk-sample "${PER_CHUNK_SAMPLE}")
CMD+=(--max-positive-sample "${MAX_POSITIVE_SAMPLE}")

if [[ -n "${OUTPUT_JSON}" ]]; then
  CMD+=(--output-json "${OUTPUT_JSON}")
fi

if [[ -n "${OUTPUT_SCALED_H5}" ]]; then
  CMD+=(--save-scaled-h5 "${OUTPUT_SCALED_H5}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
