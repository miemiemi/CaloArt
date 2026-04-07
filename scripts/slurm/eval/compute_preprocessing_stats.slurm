#!/bin/bash
#SBATCH --job-name=prep_stats
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
DATA_FILES="${DATA_FILES:?DATA_FILES is required}"
SAMPLING_FRACTION="${SAMPLING_FRACTION:-0.033}"
CHUNK_SHOWERS="${CHUNK_SHOWERS:-64}"
EPS_LIST="${EPS_LIST:-1e-8 1e-7 5e-7 1e-6}"
NOISE_THRESHOLDS="${NOISE_THRESHOLDS:-1.65e-7 5e-7}"
SUGGESTED_NOISE_LEVEL="${SUGGESTED_NOISE_LEVEL:-5e-7}"
RAW_CUT_NOISE_LEVEL="${RAW_CUT_NOISE_LEVEL:-0}"
SCALE_BY_INCIDENT_ENERGY="${SCALE_BY_INCIDENT_ENERGY:-0}"
SCALE_BY_FACTOR="${SCALE_BY_FACTOR:-1}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

mkdir -p "${REPO_DIR}/logs/slurm"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow
cd "${REPO_DIR}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "DATA_FILES=${DATA_FILES}"
echo "SAMPLING_FRACTION=${SAMPLING_FRACTION}"
echo "CHUNK_SHOWERS=${CHUNK_SHOWERS}"
echo "EPS_LIST=${EPS_LIST}"
echo "NOISE_THRESHOLDS=${NOISE_THRESHOLDS}"
echo "SUGGESTED_NOISE_LEVEL=${SUGGESTED_NOISE_LEVEL}"
echo "RAW_CUT_NOISE_LEVEL=${RAW_CUT_NOISE_LEVEL}"
echo "SCALE_BY_INCIDENT_ENERGY=${SCALE_BY_INCIDENT_ENERGY}"
echo "SCALE_BY_FACTOR=${SCALE_BY_FACTOR}"
echo "OUTPUT_JSON=${OUTPUT_JSON:-none}"
nvidia-smi || true

read -r -a DATA_FILES_ARR <<< "${DATA_FILES}"
read -r -a EPS_LIST_ARR <<< "${EPS_LIST}"
read -r -a NOISE_THRESHOLDS_ARR <<< "${NOISE_THRESHOLDS}"

CMD=(
  python
  scripts/compute_preprocessing_stats.py
  --files
)
CMD+=("${DATA_FILES_ARR[@]}")
CMD+=(--sampling-fraction "${SAMPLING_FRACTION}")
CMD+=(--chunk-showers "${CHUNK_SHOWERS}")
CMD+=(--eps "${EPS_LIST_ARR[@]}")
CMD+=(--noise-thresholds "${NOISE_THRESHOLDS_ARR[@]}")
CMD+=(--suggested-noise-level "${SUGGESTED_NOISE_LEVEL}")
CMD+=(--raw-cut-noise-level "${RAW_CUT_NOISE_LEVEL}")

if [[ "${SCALE_BY_INCIDENT_ENERGY}" == "1" ]]; then
  CMD+=(--scale-by-incident-energy)
fi

if [[ "${SCALE_BY_FACTOR}" != "1" ]]; then
  CMD+=(--scale-by-factor "${SCALE_BY_FACTOR}")
fi

if [[ -n "${OUTPUT_JSON}" ]]; then
  CMD+=(--output-json "${OUTPUT_JSON}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Done at $(date)."
