#!/bin/bash
#SBATCH --job-name=rebin_ccd3_one
#SBATCH --partition=cpu
#SBATCH --qos=cpunormal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
INPUT_H5="${INPUT_H5:?INPUT_H5 is required}"
OUTPUT_H5="${OUTPUT_H5:?OUTPUT_H5 is required}"
OUTPUT_XML="${OUTPUT_XML:-${OUTPUT_H5%.h5}.xml}"
CHUNK_SHOWERS="${CHUNK_SHOWERS:-512}"
COMPRESSION="${COMPRESSION:-4}"
VALIDATE_REBIN="${VALIDATE_REBIN:-1}"
REPORT_JSON="${REPORT_JSON:-}"

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
echo "INPUT_H5=${INPUT_H5}"
echo "OUTPUT_H5=${OUTPUT_H5}"
echo "OUTPUT_XML=${OUTPUT_XML}"
echo "CHUNK_SHOWERS=${CHUNK_SHOWERS}"
echo "COMPRESSION=${COMPRESSION}"
echo "VALIDATE_REBIN=${VALIDATE_REBIN}"
echo "REPORT_JSON=${REPORT_JSON:-none}"

python scripts/rebin_ccd3_dataset3.py \
  --input-h5 "${INPUT_H5}" \
  --output-h5 "${OUTPUT_H5}" \
  --output-xml "${OUTPUT_XML}" \
  --chunk-showers "${CHUNK_SHOWERS}" \
  --compression "${COMPRESSION}"

if [[ "${VALIDATE_REBIN}" == "1" ]]; then
  CMD=(
    python
    scripts/validate_rebinned_ccd3.py
    --input-h5 "${INPUT_H5}"
    --output-h5 "${OUTPUT_H5}"
    --output-xml "${OUTPUT_XML}"
    --chunk-showers "${CHUNK_SHOWERS}"
  )
  if [[ -n "${REPORT_JSON}" ]]; then
    CMD+=(--report-json "${REPORT_JSON}")
  fi

  printf 'Validating:'
  printf ' %q' "${CMD[@]}"
  echo
  "${CMD[@]}"
fi

echo "Done at $(date)."
