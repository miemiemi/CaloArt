# Slurm Entry Points

This public branch keeps only the Slurm jobs needed for the CaloArt CCD2
PixArt flow experiment.

## Training

```bash
sbatch scripts/slurm/train/train_caloart_ccd2_pixart.slurm
```

Useful overrides:

- `REPO_DIR`: repository path
- `CALOCHALLENGE_DATA_DIR`: directory containing `dataset_2_1.hdf5` and `dataset_2_2.hdf5`
- `OUTPUT_DIR`: experiment output root
- `NUM_PROCESSES`: number of GPUs/processes
- `USE_WANDB`: enable or disable wandb

## Checkpoint FPD Test

This runs `scripts/test_checkpoint.py` against the target paper experiment by
default:

```bash
sbatch scripts/slurm/eval/test_caloart_checkpoint_fpd.slurm
```

Useful overrides:

- `TARGET_EXPERIMENT`: experiment directory with `final_model.pt`
- `MODEL_PATH`: exported single-file checkpoint
- `TEST_NUM_SHOWERS`: number of validation showers
- `SAVE_GENERATED`: save `generated.h5`

## FPD From HDF5

```bash
GENERATED_FILE=path/to/generated.h5 \
REFERENCE_FILE=${CALOCHALLENGE_DATA_DIR}/dataset_2_2.hdf5 \
sbatch scripts/slurm/eval/compute_fpd_from_h5.slurm
```
