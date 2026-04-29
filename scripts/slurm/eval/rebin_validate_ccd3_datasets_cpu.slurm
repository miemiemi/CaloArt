#!/bin/bash
#SBATCH --job-name=rebin_ccd3_cpu
#SBATCH --partition=cpu
#SBATCH --qos=cpunormal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"
DATA_DIR="/aifs/user/home/huangzhengkun/datasets/hepdatasets/calochallenge_datasets"
INPUT_FILES="${INPUT_FILES:-${DATA_DIR}/dataset_3_1_2.hdf5 ${DATA_DIR}/dataset_3_3_4.hdf5}"
CHUNK_SHOWERS="${CHUNK_SHOWERS:-512}"
COMPRESSION="${COMPRESSION:-4}"
LOG_DIR="${REPO_DIR}/logs/slurm"
REPORT_DIR="${LOG_DIR}/rebin_ccd3_reports"

mkdir -p "${LOG_DIR}" "${REPORT_DIR}"

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
echo "INPUT_FILES=${INPUT_FILES}"
echo "CHUNK_SHOWERS=${CHUNK_SHOWERS}"
echo "COMPRESSION=${COMPRESSION}"

read -r -a INPUT_FILES_ARR <<< "${INPUT_FILES}"

for INPUT_H5 in "${INPUT_FILES_ARR[@]}"; do
  BASENAME="$(basename "${INPUT_H5}" .hdf5)"
  OUTPUT_H5="${DATA_DIR}/${BASENAME}_rebinned_45x25x9.hdf5"
  OUTPUT_XML="${DATA_DIR}/${BASENAME}_rebinned_45x25x9.xml"
  REPORT_JSON="${REPORT_DIR}/${BASENAME}_rebinned_45x25x9_validation.json"

  echo
  echo "=== Processing ${INPUT_H5} ==="
  echo "OUTPUT_H5=${OUTPUT_H5}"
  echo "OUTPUT_XML=${OUTPUT_XML}"
  echo "REPORT_JSON=${REPORT_JSON}"

  python scripts/rebin_ccd3_dataset3.py \
    --input-h5 "${INPUT_H5}" \
    --output-h5 "${OUTPUT_H5}" \
    --output-xml "${OUTPUT_XML}" \
    --chunk-showers "${CHUNK_SHOWERS}" \
    --compression "${COMPRESSION}"

  python scripts/validate_rebinned_ccd3.py \
    --input-h5 "${INPUT_H5}" \
    --output-h5 "${OUTPUT_H5}" \
    --output-xml "${OUTPUT_XML}" \
    --chunk-showers "${CHUNK_SHOWERS}" \
    --report-json "${REPORT_JSON}"
done

echo
echo "Done at $(date)."
