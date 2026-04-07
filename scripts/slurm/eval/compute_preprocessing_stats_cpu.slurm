#!/bin/bash
#SBATCH --job-name=prep_stats_cpu
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
CHUNK_SHOWERS="${CHUNK_SHOWERS:-64}"
EPS_LIST="${EPS_LIST:-1e-6}"
TRANSFORM_TYPE="${TRANSFORM_TYPE:-log}"
TRANSFORM_SCALE="${TRANSFORM_SCALE:-5e-7}"
NOISE_THRESHOLDS="${NOISE_THRESHOLDS:-5e-7}"
SUGGESTED_NOISE_LEVEL="${SUGGESTED_NOISE_LEVEL:-5e-7}"
RAW_CUT_NOISE_LEVEL="${RAW_CUT_NOISE_LEVEL:-0}"
ADD_NOISE_LEVEL="${ADD_NOISE_LEVEL:-0}"
SCALE_ABOVE_CUT_FACTOR="${SCALE_ABOVE_CUT_FACTOR:-1}"
SCALE_ABOVE_CUT_THRESHOLD="${SCALE_ABOVE_CUT_THRESHOLD:-5e-7}"
RANDOM_SEED="${RANDOM_SEED:-42}"
SCALE_BY_INCIDENT_ENERGY="${SCALE_BY_INCIDENT_ENERGY:-0}"
SCALE_BY_FACTOR="${SCALE_BY_FACTOR:-1}"
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
echo "EPS_LIST=${EPS_LIST}"
echo "TRANSFORM_TYPE=${TRANSFORM_TYPE}"
echo "TRANSFORM_SCALE=${TRANSFORM_SCALE}"
echo "NOISE_THRESHOLDS=${NOISE_THRESHOLDS}"
echo "SUGGESTED_NOISE_LEVEL=${SUGGESTED_NOISE_LEVEL}"
echo "RAW_CUT_NOISE_LEVEL=${RAW_CUT_NOISE_LEVEL}"
echo "ADD_NOISE_LEVEL=${ADD_NOISE_LEVEL}"
echo "SCALE_ABOVE_CUT_FACTOR=${SCALE_ABOVE_CUT_FACTOR}"
echo "SCALE_ABOVE_CUT_THRESHOLD=${SCALE_ABOVE_CUT_THRESHOLD}"
echo "RANDOM_SEED=${RANDOM_SEED}"
echo "SCALE_BY_INCIDENT_ENERGY=${SCALE_BY_INCIDENT_ENERGY}"
echo "SCALE_BY_FACTOR=${SCALE_BY_FACTOR}"
echo "OUTPUT_JSON=${OUTPUT_JSON:-none}"

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
CMD+=(--transform-type "${TRANSFORM_TYPE}")
CMD+=(--eps "${EPS_LIST_ARR[@]}")
CMD+=(--transform-scale "${TRANSFORM_SCALE}")
CMD+=(--noise-thresholds "${NOISE_THRESHOLDS_ARR[@]}")
CMD+=(--suggested-noise-level "${SUGGESTED_NOISE_LEVEL}")
CMD+=(--raw-cut-noise-level "${RAW_CUT_NOISE_LEVEL}")
CMD+=(--add-noise-level "${ADD_NOISE_LEVEL}")
CMD+=(--scale-above-cut-factor "${SCALE_ABOVE_CUT_FACTOR}")
CMD+=(--scale-above-cut-threshold "${SCALE_ABOVE_CUT_THRESHOLD}")
CMD+=(--random-seed "${RANDOM_SEED}")

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
