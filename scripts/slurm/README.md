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

## FLOPs Profiling

The repository now includes a fixed GPU profiling path for model-complexity checks:

```bash
bash scripts/submit_slurm.sh \
  scripts/slurm/eval/profile_flops_c72_36_36_pixart_rmsnorm_gpu.slurm
```

Or submit it to a different QOS directly:

```bash
sbatch --qos=gpudvp \
  scripts/slurm/eval/profile_flops_c72_36_36_pixart_rmsnorm_gpu.slurm
```

Useful environment overrides:

- `CONFIG_NAME`: Hydra config name to profile
- `PROFILE_DEVICE`: `cuda` or `cpu`
- `PROFILE_OUTPUT_JSON`: machine-readable summary path
- `PROFILE_DEEPSPEED_REPORT_PATH`: DeepSpeed-style detailed text report path
- `PROFILE_DEEPSPEED_TOP_MODULES`: how many top modules to show in aggregated output
- `PROFILE_DEEPSPEED_MODULE_DEPTH`: max aggregation depth (`-1` uses all depths)
- `PROFILE_DEEPSPEED_DETAILED=0`: disable the full detailed module dump

Example:

```bash
CONFIG_NAME=experiment/CaloChallenge/flow_ccd2_pred_v_all_lightingdit_freatures_c72_36_36_pixart_sharemod_heun32_latest \
PROFILE_OUTPUT_JSON=logs/my_profile.json \
PROFILE_DEEPSPEED_REPORT_PATH=logs/my_profile_deepspeed.txt \
sbatch --qos=gpudvp scripts/slurm/eval/profile_flops_c72_36_36_pixart_rmsnorm_gpu.slurm
```

Outputs:

- Slurm launcher logs: `logs/slurm/prof_flops_<jobid>.out|err`
- Structured summary: `PROFILE_OUTPUT_JSON`
- DeepSpeed-style text report: `PROFILE_DEEPSPEED_REPORT_PATH`

Current recommendation:

- Treat DeepSpeed as the primary FLOPs/MACs number.
- Use `ptflops(aten)` as a cross-check on GPU.
- Do not rely on `torchinfo`'s `Total mult-adds` for these transformer-style models.

Notes:

- The profiling script targets the backbone model rather than the training wrapper.
- It temporarily switches `@torch.compile`-decorated modules back to their eager `_forward` path before profiling, which reduces profiler/compile interference.
- The DeepSpeed report is produced in standalone mode, so it does not include training-engine-only fields like backward latency or samples/second unless you later profile inside a real DeepSpeed engine.

## Naming

Keep committed scripts descriptive and stable. Prefer adding environment overrides for run-specific changes instead of creating new root-level launchers.
