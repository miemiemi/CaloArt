#!/bin/bash
#SBATCH --job-name=trace_roundtrip_gpu
#SBATCH --partition=gpu
#SBATCH --qos=gpudvp
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow
cd "${REPO_DIR}"

export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR="${REPO_DIR}/logs/tmp/${SLURM_JOB_ID}"

mkdir -p "${REPO_DIR}/logs/slurm" "${TMPDIR}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
MODE="${MODE:-both}"
REFERENCE_FILE="${REFERENCE_FILE:-}"
CONDITION_FILE="${CONDITION_FILE:-}"
NUM_REFERENCE_SHOWERS="${NUM_REFERENCE_SHOWERS:-100000}"
NUM_SAMPLE_SHOWERS="${NUM_SAMPLE_SHOWERS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SAMPLING_STEPS="${SAMPLING_STEPS:-}"
SAMPLING_SOLVER="${SAMPLING_SOLVER:-}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
DUMP_EVENTS="${DUMP_EVENTS:-256}"
VALUE_SAMPLE_LIMIT="${VALUE_SAMPLE_LIMIT:-500000}"

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODE=${MODE}"
echo "REFERENCE_FILE=${REFERENCE_FILE:-default-from-config}"
echo "CONDITION_FILE=${CONDITION_FILE:-reference-or-default-from-config}"
echo "NUM_REFERENCE_SHOWERS=${NUM_REFERENCE_SHOWERS}"
echo "NUM_SAMPLE_SHOWERS=${NUM_SAMPLE_SHOWERS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "SAMPLING_STEPS=${SAMPLING_STEPS:-default-from-config}"
echo "SAMPLING_SOLVER=${SAMPLING_SOLVER:-default-from-config}"
echo "SEED=${SEED}"
echo "DEVICE=${DEVICE}"
echo "DUMP_EVENTS=${DUMP_EVENTS}"
echo "VALUE_SAMPLE_LIMIT=${VALUE_SAMPLE_LIMIT}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

nvidia-smi || true

CMD=(
  python
  -u
  scripts/trace_preprocessing_roundtrip.py
  --mode "${MODE}"
  --model-path "${MODEL_PATH}"
  --num-reference-showers "${NUM_REFERENCE_SHOWERS}"
  --num-sample-showers "${NUM_SAMPLE_SHOWERS}"
  --batch-size "${BATCH_SIZE}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --dump-events "${DUMP_EVENTS}"
  --value-sample-limit "${VALUE_SAMPLE_LIMIT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${REFERENCE_FILE}" ]]; then
  CMD+=(--reference-file "${REFERENCE_FILE}")
fi

if [[ -n "${CONDITION_FILE}" ]]; then
  CMD+=(--condition-file "${CONDITION_FILE}")
fi

if [[ -n "${SAMPLING_STEPS}" ]]; then
  CMD+=(--sampling-steps "${SAMPLING_STEPS}")
fi

if [[ -n "${SAMPLING_SOLVER}" ]]; then
  CMD+=(--sampling-solver "${SAMPLING_SOLVER}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Done at $(date)."
