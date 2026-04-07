#!/bin/bash
#SBATCH --job-name=exact_logit_stats_cpu
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
RAW_CUT_NOISE_LEVEL="${RAW_CUT_NOISE_LEVEL:-5e-7}"
REMOVE_SAMPLING_FRACTION_FACTOR="${REMOVE_SAMPLING_FRACTION_FACTOR:-1}"
SCALE_BY_INCIDENT_ENERGY="${SCALE_BY_INCIDENT_ENERGY:-0}"
SCALE_BY_FACTOR="${SCALE_BY_FACTOR:-1}"
LOGIT_EPS="${LOGIT_EPS:-1e-6}"
STANDARDIZE_MEAN="${STANDARDIZE_MEAN:-}"
STANDARDIZE_STD="${STANDARDIZE_STD:-}"
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
echo "REMOVE_SAMPLING_FRACTION_FACTOR=${REMOVE_SAMPLING_FRACTION_FACTOR}"
echo "SCALE_BY_INCIDENT_ENERGY=${SCALE_BY_INCIDENT_ENERGY}"
echo "SCALE_BY_FACTOR=${SCALE_BY_FACTOR}"
echo "LOGIT_EPS=${LOGIT_EPS}"
echo "STANDARDIZE_MEAN=${STANDARDIZE_MEAN:-none}"
echo "STANDARDIZE_STD=${STANDARDIZE_STD:-none}"
echo "OUTPUT_JSON=${OUTPUT_JSON:-none}"

read -r -a DATA_FILES_ARR <<< "${DATA_FILES}"

CMD=(
  python
  scripts/compute_exact_logit_preprocessing_stats.py
  --files
)
CMD+=("${DATA_FILES_ARR[@]}")
CMD+=(--sampling-fraction "${SAMPLING_FRACTION}")
CMD+=(--chunk-showers "${CHUNK_SHOWERS}")
CMD+=(--raw-cut-noise-level "${RAW_CUT_NOISE_LEVEL}")
CMD+=(--remove-sampling-fraction-factor "${REMOVE_SAMPLING_FRACTION_FACTOR}")
CMD+=(--scale-by-factor "${SCALE_BY_FACTOR}")
CMD+=(--logit-eps "${LOGIT_EPS}")

if [[ "${SCALE_BY_INCIDENT_ENERGY}" == "1" ]]; then
  CMD+=(--scale-by-incident-energy)
fi

if [[ -n "${STANDARDIZE_MEAN}" ]]; then
  CMD+=(--standardize-mean "${STANDARDIZE_MEAN}")
fi

if [[ -n "${STANDARDIZE_STD}" ]]; then
  CMD+=(--standardize-std "${STANDARDIZE_STD}")
fi

if [[ -n "${OUTPUT_JSON}" ]]; then
  CMD+=(--output-json "${OUTPUT_JSON}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Done at $(date)."
