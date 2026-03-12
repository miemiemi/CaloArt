# Slurm Layout

Managed Slurm launchers live here instead of the repository root.

## Structure

- `train/`: long-running training jobs
- `smoke/`: short smoke or debug jobs
- `eval/`: export, validation, and evaluation jobs
- `local/`: private one-off launchers ignored by git

## Usage

Submit through the helper so the shared log directory always exists:

```bash
bash scripts/submit_slurm.sh scripts/slurm/train/train_logit_normal.slurm
```

Slurm stdout/stderr lands in `logs/slurm/`.

## Naming

Keep committed scripts descriptive and stable. Prefer adding environment overrides for run-specific changes instead of creating new root-level launchers.
