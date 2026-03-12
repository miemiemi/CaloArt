#!/bin/bash
#SBATCH --job-name=eval_flow
#SBATCH --partition=gpu
#SBATCH --qos=gpunormal
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodelist=aigpu[002-005]
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.out
#SBATCH --error=/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/aifs/user/home/huangzhengkun/work/repositories/CaloFlow"

source ~/miniconda3/etc/profile.d/conda.sh
cd "${REPO_DIR}"
conda activate CaloFlow

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

# Export PYTHONPATH to include the project root
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}"

CHECKPOINT_DIR="experiments/caloflow_v1/flow_ccd2_pred_v_loss_v_logit_normal/checkpoints/checkpoint_best"
FINAL_MODEL_PT="experiments/caloflow_v1/flow_ccd2_pred_v_loss_v_logit_normal/final_model.pt"

echo "Exporting checkpoint to .pt..."
python scripts/export_checkpoint.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --output_path ${FINAL_MODEL_PT} \
    --use_ema

echo "Running validation generation..."
python scripts/validate.py --config-name experiment/CaloChallenge/validate/ccd2_flow \
    model.model_path=${FINAL_MODEL_PT}

echo "Done at $(date)."
