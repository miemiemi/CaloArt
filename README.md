# CaloArt CCD2 Flow

Minimal public code branch for the CaloArt CCD2 PixArt flow-matching model.

This branch keeps the source code and the single paper configuration needed to
train and evaluate:

```text
experiments/CaloArt_paper/flow_ccd2_pred_v_loss_v_logit_normal_cond_energy_log10_h384_l6_nh6_patch_3_4_3_pixart_fpd100k
```

Large run artifacts are intentionally not tracked by git. Checkpoints, tensorboard
events, generated HDF5 files, wandb runs, and experiment outputs remain ignored.

## Setup

```bash
conda create -n CaloFlow python=3.10
conda activate CaloFlow
pip install -r requirements.txt
```

Set the CaloChallenge dataset directory if it is not in the default cluster
location:

```bash
export CALOCHALLENGE_DATA_DIR=/path/to/calochallenge_datasets
```

The directory should contain:

```text
dataset_2_1.hdf5
dataset_2_2.hdf5
```

## Train

```bash
accelerate launch --multi_gpu --num_processes=2 scripts/train.py \
  --config-name experiment/CaloArt_paper/flow_ccd2_pred_v_loss_v_logit_normal_cond_energy_log10_h384_l6_nh6_patch_3_4_3_pixart_fpd100k
```

On the local Slurm cluster:

```bash
sbatch scripts/slurm/train/train_caloart_ccd2_pixart.slurm
```

The default output directory is `experiments/CaloArt_paper/...`.

## Evaluate Existing Experiment

The target experiment directory on this machine already contains `final_model.pt`.
Run the checkpoint-only FPD path with:

```bash
export MODEL_PATH=experiments/CaloArt_paper/flow_ccd2_pred_v_loss_v_logit_normal_cond_energy_log10_h384_l6_nh6_patch_3_4_3_pixart_fpd100k/final_model.pt
accelerate launch --num_processes=1 scripts/test_checkpoint.py \
  experiment.output_dir=./experiments \
  experiment.run_name=caloart_ccd2_pixart_checkpoint_fpd \
  model.model_path=${MODEL_PATH} \
  ++train.enable_plots=false \
  ++train.enable_fpd=true \
  ++train.test_num_showers=100000 \
  ++train.fpd_config.save_generated=true \
  ++train.save_generated=true
```

Or submit the Slurm wrapper:

```bash
sbatch scripts/slurm/eval/test_caloart_checkpoint_fpd.slurm
```

To recompute FPD from a saved `generated.h5`:

```bash
GENERATED_FILE=/path/to/generated.h5 \
REFERENCE_FILE=${CALOCHALLENGE_DATA_DIR}/dataset_2_2.hdf5 \
sbatch scripts/slurm/eval/compute_fpd_from_h5.slurm
```

## Kept Configuration

- `configs/data/ccd2.yaml`
- `configs/preprocessing/ccd2_cond_energy_log10.yaml`
- `configs/model/calolightning_h384_ape_rope_c192_96_96_sharemod_rmsnorm_nh6_patch_3_4_3_pixart.yaml`
- `configs/method/flow_matching/logit_normal.yaml`
- `configs/sampling/flow/heun_32.yaml`
- `configs/experiment/CaloArt_paper/flow_ccd2_pred_v_loss_v_logit_normal_cond_energy_log10_h384_l6_nh6_patch_3_4_3_pixart_fpd100k.yaml`
