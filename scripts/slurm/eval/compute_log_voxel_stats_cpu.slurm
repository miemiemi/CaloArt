#!/bin/bash
#SBATCH --job-name=log_voxel_cpu
#SBATCH --partition=cpu
#SBATCH --qos=cpunormal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
DATA_FILES="${DATA_FILES:?DATA_FILES is required}"
SAMPLING_FRACTION="${SAMPLING_FRACTION:-0.033}"
CHUNK_SHOWERS="${CHUNK_SHOWERS:-1024}"
RAW_CUT_NOISE_LEVEL="${RAW_CUT_NOISE_LEVEL:-5e-7}"
TRANSFORM_TYPE="${TRANSFORM_TYPE:-log}"
EPS="${EPS:-1e-6}"
TRANSFORM_SCALE="${TRANSFORM_SCALE:-5e-7}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

mkdir -p "${REPO_DIR}/logs/slurm"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow
cd "${REPO_DIR}"

export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "DATA_FILES=${DATA_FILES}"
echo "SAMPLING_FRACTION=${SAMPLING_FRACTION}"
echo "CHUNK_SHOWERS=${CHUNK_SHOWERS}"
echo "RAW_CUT_NOISE_LEVEL=${RAW_CUT_NOISE_LEVEL}"
echo "TRANSFORM_TYPE=${TRANSFORM_TYPE}"
echo "EPS=${EPS}"
echo "TRANSFORM_SCALE=${TRANSFORM_SCALE}"
echo "OUTPUT_JSON=${OUTPUT_JSON:-none}"

read -r -a DATA_FILES_ARR <<< "${DATA_FILES}"

CMD=(
  python
  scripts/compute_log_voxel_stats.py
  --files
)
CMD+=("${DATA_FILES_ARR[@]}")
CMD+=(--sampling-fraction "${SAMPLING_FRACTION}")
CMD+=(--chunk-showers "${CHUNK_SHOWERS}")
CMD+=(--raw-cut-noise-level "${RAW_CUT_NOISE_LEVEL}")
CMD+=(--transform-type "${TRANSFORM_TYPE}")
CMD+=(--eps "${EPS}")
CMD+=(--transform-scale "${TRANSFORM_SCALE}")

if [[ -n "${OUTPUT_JSON}" ]]; then
  CMD+=(--output-json "${OUTPUT_JSON}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Done at $(date)."
