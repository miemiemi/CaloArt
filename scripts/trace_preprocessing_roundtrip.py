#!/usr/bin/env python3
"""Compare reference round-trips and model samples across inverse-preprocessing stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, pythonpath=True)

from scripts.diagnose_sampling_values import build_dataset, default_condition_file, load_model
from src.data.preprocessing import CaloShowerPreprocessor
from src.flow.reject_redraw import filter_model_sample_kwargs
from src.utils import set_seed, to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trace preprocessing/inverse-preprocessing stages for a reference dataset and/or model samples."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=("reference", "sample", "both"),
        help="Which source(s) to trace.",
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to an exported model with config.")
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=None,
        help="Reference HDF5 used for round-trip tracing. Defaults to the first test condition file in the saved config.",
    )
    parser.add_argument(
        "--condition-file",
        type=Path,
        default=None,
        help="Condition HDF5 for sampling. Defaults to --reference-file, then to the first test condition file in the saved config.",
    )
    parser.add_argument(
        "--num-reference-showers",
        type=int,
        default=2048,
        help="Number of reference showers to trace when mode includes reference.",
    )
    parser.add_argument(
        "--num-sample-showers",
        type=int,
        default=2048,
        help="Number of sampled showers to trace when mode includes sample.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for both reference and sample passes.")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Optional override for sampler steps.")
    parser.add_argument("--sampling-solver", type=str, default=None, help="Optional override for sampler solver.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--dump-events",
        type=int,
        default=64,
        help="How many events per stage to dump into compressed NPZ files. Use 0 to disable tensor dumps.",
    )
    parser.add_argument(
        "--value-sample-limit",
        type=int,
        default=500000,
        help="Maximum number of per-voxel finite values kept per stage for quantile estimation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON summaries and stage dumps.")
    return parser.parse_args()


def _maybe_squeeze_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 2 and tensor.shape[1] == 1:
        return tensor.squeeze(1)
    return tensor


def _append_sample(store: list[np.ndarray], values: np.ndarray, limit: int):
    if values.size == 0 or limit <= 0:
        return
    current = sum(arr.size for arr in store)
    if current >= limit:
        return
    remaining = limit - current
    store.append(values[:remaining].astype(np.float64, copy=False))


def _quantiles(values: np.ndarray):
    if values.size == 0:
        return None
    return {
        "p01": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def _sanitize_stage_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


class StageAccumulator:
    def __init__(self, name: str, dump_limit: int, value_sample_limit: int, noise_level: float, raw_energy_space: bool):
        self.name = name
        self.dump_limit = dump_limit
        self.value_sample_limit = value_sample_limit
        self.noise_level = noise_level
        self.raw_energy_space = raw_energy_space

        self.event_count = 0
        self.value_count = 0
        self.finite_count = 0
        self.finite_sum = 0.0
        self.finite_sq_sum = 0.0
        self.finite_min = None
        self.finite_max = None

        self.event_sums: list[np.ndarray] = []
        self.event_ratios: list[np.ndarray] = []
        self.finite_value_sample: list[np.ndarray] = []

        self.positive_count = 0
        self.zero_count = 0
        self.negative_count = 0
        self.above_noise_count = 0
        self.nonzero_below_or_equal_noise_count = 0
        self.above_noise_sum = 0.0
        self.nonzero_below_or_equal_noise_sum = 0.0

        self.dumped_showers: list[np.ndarray] = []
        self.dumped_energies: list[np.ndarray] = []
        self.dumped_indices: list[np.ndarray] = []

    def update(self, tensor: torch.Tensor, incident_energy: torch.Tensor, global_offset: int):
        tensor = _maybe_squeeze_channel(tensor)
        flat = tensor.detach().reshape(tensor.shape[0], -1).cpu().numpy().astype(np.float64, copy=False)
        energy = incident_energy.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)

        self.event_count += int(flat.shape[0])
        self.value_count += int(flat.size)

        finite_mask = np.isfinite(flat)
        finite_values = flat[finite_mask]
        self.finite_count += int(finite_values.size)
        if finite_values.size:
            self.finite_sum += float(finite_values.sum())
            self.finite_sq_sum += float(np.square(finite_values).sum())
            batch_min = float(finite_values.min())
            batch_max = float(finite_values.max())
            self.finite_min = batch_min if self.finite_min is None else min(self.finite_min, batch_min)
            self.finite_max = batch_max if self.finite_max is None else max(self.finite_max, batch_max)
            _append_sample(self.finite_value_sample, finite_values, self.value_sample_limit)

        event_sum = np.where(finite_mask, flat, 0.0).sum(axis=1)
        self.event_sums.append(event_sum)
        if self.raw_energy_space:
            self.event_ratios.append(event_sum / np.maximum(energy, 1e-30))

            positive_mask = flat > 0.0
            zero_mask = flat == 0.0
            negative_mask = flat < 0.0
            above_noise_mask = flat > self.noise_level
            nonzero_below_or_equal_noise_mask = positive_mask & ~above_noise_mask

            self.positive_count += int(positive_mask.sum())
            self.zero_count += int(zero_mask.sum())
            self.negative_count += int(negative_mask.sum())
            self.above_noise_count += int(above_noise_mask.sum())
            self.nonzero_below_or_equal_noise_count += int(nonzero_below_or_equal_noise_mask.sum())
            self.above_noise_sum += float(flat[above_noise_mask].sum())
            self.nonzero_below_or_equal_noise_sum += float(flat[nonzero_below_or_equal_noise_mask].sum())

        if self.dump_limit > 0:
            dumped = sum(arr.shape[0] for arr in self.dumped_showers)
            if dumped < self.dump_limit:
                take = min(self.dump_limit - dumped, tensor.shape[0])
                self.dumped_showers.append(tensor[:take].detach().cpu().numpy().astype(np.float32, copy=False))
                self.dumped_energies.append(incident_energy[:take].detach().reshape(-1).cpu().numpy().astype(np.float32, copy=False))
                self.dumped_indices.append(np.arange(global_offset, global_offset + take, dtype=np.int64))

    def write_dump(self, output_path: Path):
        if self.dump_limit <= 0 or not self.dumped_showers:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            showers=np.concatenate(self.dumped_showers, axis=0),
            incident_energy=np.concatenate(self.dumped_energies, axis=0),
            event_index=np.concatenate(self.dumped_indices, axis=0),
        )
        return str(output_path.resolve())

    def summary(self):
        event_sum = np.concatenate(self.event_sums) if self.event_sums else np.array([], dtype=np.float64)
        event_ratio = np.concatenate(self.event_ratios) if self.event_ratios else np.array([], dtype=np.float64)
        finite_sample = (
            np.concatenate(self.finite_value_sample) if self.finite_value_sample else np.array([], dtype=np.float64)
        )

        finite_mean = None
        finite_std = None
        if self.finite_count > 0:
            finite_mean = self.finite_sum / self.finite_count
            variance = max(self.finite_sq_sum / self.finite_count - finite_mean * finite_mean, 0.0)
            finite_std = variance ** 0.5

        result = {
            "raw_energy_space": self.raw_energy_space,
            "event_count": int(self.event_count),
            "value_count": int(self.value_count),
            "finite_fraction": float(self.finite_count / self.value_count) if self.value_count else 0.0,
            "finite_min": self.finite_min,
            "finite_max": self.finite_max,
            "finite_mean": float(finite_mean) if finite_mean is not None else None,
            "finite_std": float(finite_std) if finite_std is not None else None,
            "finite_value_quantiles": _quantiles(finite_sample),
            "event_sum_mean": float(event_sum.mean()) if event_sum.size else None,
            "event_sum_quantiles": _quantiles(event_sum),
        }
        if self.raw_energy_space:
            total_value_count = max(self.value_count, 1)
            total_energy_sum = max(float(event_sum.sum()), 0.0)
            result.update(
                {
                    "noise_level": self.noise_level,
                    "event_ratio_mean": float(event_ratio.mean()) if event_ratio.size else None,
                    "event_ratio_quantiles": _quantiles(event_ratio),
                    "positive_fraction": float(self.positive_count / total_value_count),
                    "zero_fraction": float(self.zero_count / total_value_count),
                    "negative_fraction": float(self.negative_count / total_value_count),
                    "above_noise_fraction": float(self.above_noise_count / total_value_count),
                    "nonzero_below_or_equal_noise_fraction": float(
                        self.nonzero_below_or_equal_noise_count / total_value_count
                    ),
                    "above_noise_energy_fraction": float(self.above_noise_sum / total_energy_sum)
                    if total_energy_sum > 0
                    else 0.0,
                    "nonzero_below_or_equal_noise_energy_fraction": float(
                        self.nonzero_below_or_equal_noise_sum / total_energy_sum
                    )
                    if total_energy_sum > 0
                    else 0.0,
                }
            )
        return result


class DeltaAccumulator:
    def __init__(self, name: str, value_sample_limit: int, raw_energy_space: bool):
        self.name = name
        self.value_sample_limit = value_sample_limit
        self.raw_energy_space = raw_energy_space

        self.event_count = 0
        self.value_count = 0
        self.changed_count = 0
        self.positive_to_zero_count = 0
        self.delta_sum = 0.0
        self.delta_abs_sum = 0.0
        self.delta_min = None
        self.delta_max = None
        self.delta_sample: list[np.ndarray] = []
        self.relative_event_drop: list[np.ndarray] = []

    def update(self, before: torch.Tensor, after: torch.Tensor, incident_energy: torch.Tensor):
        before = _maybe_squeeze_channel(before)
        after = _maybe_squeeze_channel(after)
        before_flat = before.detach().reshape(before.shape[0], -1).cpu().numpy().astype(np.float64, copy=False)
        after_flat = after.detach().reshape(after.shape[0], -1).cpu().numpy().astype(np.float64, copy=False)
        _ = incident_energy

        delta = after_flat - before_flat
        self.event_count += int(delta.shape[0])
        self.value_count += int(delta.size)
        changed_mask = delta != 0.0
        self.changed_count += int(changed_mask.sum())
        self.positive_to_zero_count += int(((before_flat > 0.0) & (after_flat == 0.0)).sum())
        self.delta_sum += float(delta.sum())
        self.delta_abs_sum += float(np.abs(delta).sum())
        if delta.size:
            batch_min = float(delta.min())
            batch_max = float(delta.max())
            self.delta_min = batch_min if self.delta_min is None else min(self.delta_min, batch_min)
            self.delta_max = batch_max if self.delta_max is None else max(self.delta_max, batch_max)
        _append_sample(self.delta_sample, delta.reshape(-1), self.value_sample_limit)

        if self.raw_energy_space:
            before_sum = before_flat.sum(axis=1)
            after_sum = after_flat.sum(axis=1)
            drop = (before_sum - after_sum) / np.maximum(before_sum, 1e-30)
            self.relative_event_drop.append(drop)

    def summary(self):
        delta_sample = np.concatenate(self.delta_sample) if self.delta_sample else np.array([], dtype=np.float64)
        relative_drop = (
            np.concatenate(self.relative_event_drop) if self.relative_event_drop else np.array([], dtype=np.float64)
        )
        result = {
            "raw_energy_space": self.raw_energy_space,
            "event_count": int(self.event_count),
            "value_count": int(self.value_count),
            "changed_fraction": float(self.changed_count / self.value_count) if self.value_count else 0.0,
            "positive_to_zero_fraction": float(self.positive_to_zero_count / self.value_count) if self.value_count else 0.0,
            "delta_sum": float(self.delta_sum),
            "delta_abs_sum": float(self.delta_abs_sum),
            "delta_min": self.delta_min,
            "delta_max": self.delta_max,
            "delta_quantiles": _quantiles(delta_sample),
        }
        if self.raw_energy_space:
            result.update(
                {
                    "relative_event_drop_mean": float(relative_drop.mean()) if relative_drop.size else None,
                    "relative_event_drop_quantiles": _quantiles(relative_drop),
                }
            )
        return result


class SourceTraceCollector:
    def __init__(self, name: str, output_dir: Path, dump_limit: int, value_sample_limit: int, noise_level: float):
        self.name = name
        self.output_dir = output_dir
        self.dump_limit = dump_limit
        self.value_sample_limit = value_sample_limit
        self.noise_level = noise_level

        self.stage_order: list[str] = []
        self.stage_accumulators: dict[str, StageAccumulator] = {}
        self.delta_order: list[str] = []
        self.delta_accumulators: dict[str, DeltaAccumulator] = {}
        self.stage_dump_paths: dict[str, str] = {}

    def record_stage(self, name: str, tensor: torch.Tensor, incident_energy: torch.Tensor, global_offset: int, raw_energy_space: bool):
        if name not in self.stage_accumulators:
            self.stage_order.append(name)
            self.stage_accumulators[name] = StageAccumulator(
                name=name,
                dump_limit=self.dump_limit,
                value_sample_limit=self.value_sample_limit,
                noise_level=self.noise_level,
                raw_energy_space=raw_energy_space,
            )
        self.stage_accumulators[name].update(tensor, incident_energy, global_offset)

    def record_delta(self, name: str, before: torch.Tensor, after: torch.Tensor, incident_energy: torch.Tensor, raw_energy_space: bool):
        if name not in self.delta_accumulators:
            self.delta_order.append(name)
            self.delta_accumulators[name] = DeltaAccumulator(
                name=name,
                value_sample_limit=self.value_sample_limit,
                raw_energy_space=raw_energy_space,
            )
        self.delta_accumulators[name].update(before, after, incident_energy)

    def finalize(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for idx, stage_name in enumerate(self.stage_order):
            dump_path = self.stage_accumulators[stage_name].write_dump(
                self.output_dir / f"{idx:02d}_{_sanitize_stage_name(stage_name)}.npz"
            )
            if dump_path is not None:
                self.stage_dump_paths[stage_name] = dump_path

    def summary(self):
        return {
            "stage_order": self.stage_order,
            "stages": {
                name: {
                    **self.stage_accumulators[name].summary(),
                    "dump_path": self.stage_dump_paths.get(name),
                }
                for name in self.stage_order
            },
            "delta_order": self.delta_order,
            "deltas": {name: self.delta_accumulators[name].summary() for name in self.delta_order},
        }


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stage_defs_from_pipeline(preprocessor: CaloShowerPreprocessor):
    reversed_pipeline = list(reversed(preprocessor.shower_preprocessor.pipeline))
    stage_defs = []
    for idx, step in enumerate(reversed_pipeline):
        remaining_steps = reversed_pipeline[idx + 1 :]
        raw_energy_space = all(type(remaining).__name__ == "CutNoise" for remaining in remaining_steps)
        stage_defs.append(
            {
                "name": f"after_{type(step).__name__}_inverse",
                "step": step,
                "raw_energy_space": raw_energy_space,
            }
        )
    return stage_defs


def _collect_reference_trace(
    collector: SourceTraceCollector,
    preprocessor: CaloShowerPreprocessor,
    dataset,
    batch_size: int,
    device: torch.device,
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    stage_defs = _stage_defs_from_pipeline(preprocessor)

    seen_events = 0
    with torch.inference_mode():
        for showers, conditions in dataloader:
            showers = showers.to(device)
            incident_energy = conditions[0].to(device)
            conditions = to_device(conditions, device)

            raw_showers = _maybe_squeeze_channel(showers)
            transformed_showers, _ = preprocessor.transform(showers=showers, conditions=conditions)
            transformed_showers = _maybe_squeeze_channel(transformed_showers)

            collector.record_stage("raw_input", raw_showers, incident_energy, seen_events, raw_energy_space=True)
            collector.record_stage("pre_inverse", transformed_showers, incident_energy, seen_events, raw_energy_space=False)

            previous_name = "pre_inverse"
            previous_tensor = transformed_showers
            for stage_def in stage_defs:
                current = stage_def["step"].inverse_transform(previous_tensor, incident_energy)
                current_name = stage_def["name"]
                collector.record_stage(
                    current_name,
                    current,
                    incident_energy,
                    seen_events,
                    raw_energy_space=stage_def["raw_energy_space"],
                )
                collector.record_delta(
                    f"{previous_name}__to__{current_name}",
                    previous_tensor,
                    current,
                    incident_energy,
                    raw_energy_space=stage_def["raw_energy_space"]
                    and collector.stage_accumulators[previous_name].raw_energy_space,
                )
                previous_name = current_name
                previous_tensor = current

            collector.record_delta(
                "raw_input__to__final_inverse",
                raw_showers,
                previous_tensor,
                incident_energy,
                raw_energy_space=True,
            )
            seen_events += int(showers.shape[0])


def _collect_sample_trace(
    collector: SourceTraceCollector,
    model,
    preprocessor: CaloShowerPreprocessor,
    dataset,
    batch_size: int,
    device: torch.device,
    sampling_cfg: dict,
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    stage_defs = _stage_defs_from_pipeline(preprocessor)

    seen_events = 0
    with torch.inference_mode():
        for _showers, conditions in dataloader:
            incident_energy = conditions[0].to(device)
            _, transformed_conditions = preprocessor.transform(conditions=conditions)
            transformed_conditions = to_device(transformed_conditions, device)

            sample_output = model.sample(
                conditions=transformed_conditions,
                progress=False,
                **filter_model_sample_kwargs(sampling_cfg),
            ).squeeze(1)
            collector.record_stage("pre_inverse", sample_output, incident_energy, seen_events, raw_energy_space=False)

            previous_name = "pre_inverse"
            previous_tensor = sample_output
            for stage_def in stage_defs:
                current = stage_def["step"].inverse_transform(previous_tensor, incident_energy)
                current_name = stage_def["name"]
                collector.record_stage(
                    current_name,
                    current,
                    incident_energy,
                    seen_events,
                    raw_energy_space=stage_def["raw_energy_space"],
                )
                collector.record_delta(
                    f"{previous_name}__to__{current_name}",
                    previous_tensor,
                    current,
                    incident_energy,
                    raw_energy_space=stage_def["raw_energy_space"]
                    and collector.stage_accumulators[previous_name].raw_energy_space,
                )
                previous_name = current_name
                previous_tensor = current

            seen_events += int(incident_energy.shape[0])


def main():
    args = parse_args()
    set_seed(args.seed, all_gpus=True)

    model, cfg = load_model(args.model_path)
    preprocessor = CaloShowerPreprocessor(**cfg.preprocessing)
    noise_level = float(preprocessor.shower_preprocessor.noise_level)

    reference_file = default_condition_file(cfg, args.reference_file).resolve()
    condition_file = (
        args.condition_file.resolve()
        if args.condition_file is not None
        else reference_file
    )

    device = _resolve_device(args.device)
    model = model.to(device)
    model.eval()

    sampling_cfg = OmegaConf.to_container(cfg.sampling, resolve=True)
    if args.sampling_steps is not None:
        sampling_cfg["steps"] = args.sampling_steps
    if args.sampling_solver is not None:
        sampling_cfg["solver"] = args.sampling_solver

    summary = {
        "model_path": str(args.model_path.resolve()),
        "reference_file": str(reference_file),
        "condition_file": str(condition_file),
        "device": str(device),
        "seed": int(args.seed),
        "dump_events": int(args.dump_events),
        "value_sample_limit": int(args.value_sample_limit),
        "noise_level": noise_level,
        "sampling": {
            "steps": int(sampling_cfg["steps"]),
            "solver": str(sampling_cfg["solver"]),
        },
        "sources": {},
    }

    if args.mode in ("reference", "both"):
        reference_dataset = build_dataset(cfg, reference_file, args.num_reference_showers)
        reference_collector = SourceTraceCollector(
            name="reference",
            output_dir=args.output_dir / "reference",
            dump_limit=args.dump_events,
            value_sample_limit=args.value_sample_limit,
            noise_level=noise_level,
        )
        _collect_reference_trace(
            collector=reference_collector,
            preprocessor=preprocessor,
            dataset=reference_dataset,
            batch_size=args.batch_size,
            device=device,
        )
        reference_collector.finalize()
        summary["sources"]["reference"] = {
            "num_showers": int(args.num_reference_showers),
            "geometry": str(reference_dataset.ccd_geometry or ""),
            **reference_collector.summary(),
        }

    if args.mode in ("sample", "both"):
        sample_dataset = build_dataset(cfg, condition_file, args.num_sample_showers)
        sample_collector = SourceTraceCollector(
            name="sample",
            output_dir=args.output_dir / "sample",
            dump_limit=args.dump_events,
            value_sample_limit=args.value_sample_limit,
            noise_level=noise_level,
        )
        _collect_sample_trace(
            collector=sample_collector,
            model=model,
            preprocessor=preprocessor,
            dataset=sample_dataset,
            batch_size=args.batch_size,
            device=device,
            sampling_cfg=sampling_cfg,
        )
        sample_collector.finalize()
        summary["sources"]["sample"] = {
            "num_showers": int(args.num_sample_showers),
            "geometry": str(sample_dataset.ccd_geometry or ""),
            **sample_collector.summary(),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nsummary_written: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
