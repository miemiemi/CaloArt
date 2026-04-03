#!/bin/bash
#SBATCH --job-name=h5_inspect
#SBATCH --partition=cpu
#SBATCH --qos=cpudebug
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CaloFlow
cd "${REPO_DIR}"

INPUT_FILE="${INPUT_FILE:?INPUT_FILE is required}"
CHUNK_SHOWERS="${CHUNK_SHOWERS:-512}"
PER_CHUNK_SAMPLE="${PER_CHUNK_SAMPLE:-10000}"
MAX_ALL_SAMPLE="${MAX_ALL_SAMPLE:-2000000}"
MAX_NONZERO_SAMPLE="${MAX_NONZERO_SAMPLE:-2000000}"
TOP_K="${TOP_K:-10}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

CMD=(
  python
  -u
  scripts/inspect_h5_values.py
  --input-file "${INPUT_FILE}"
  --chunk-showers "${CHUNK_SHOWERS}"
  --per-chunk-sample "${PER_CHUNK_SAMPLE}"
  --max-all-sample "${MAX_ALL_SAMPLE}"
  --max-nonzero-sample "${MAX_NONZERO_SAMPLE}"
  --top-k "${TOP_K}"
)

if [[ -n "${OUTPUT_JSON}" ]]; then
  CMD+=(--output-json "${OUTPUT_JSON}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
