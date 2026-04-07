#!/usr/bin/env python3
"""Measure how much energy/occupancy is removed by inverse-preprocessing CutNoise."""

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
from src.evaluation.fpd_kpd import get_evaluator_threshold_from_internal_noise
from src.flow.reject_redraw import filter_model_sample_kwargs
from src.utils import set_seed, to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose whether inverse-preprocessing CutNoise removes many generated voxels."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to an exported model with config.")
    parser.add_argument(
        "--condition-file",
        type=Path,
        default=None,
        help="Optional HDF5 file providing conditioning energies.",
    )
    parser.add_argument("--num-showers", type=int, default=2048, help="Number of showers to sample.")
    parser.add_argument("--batch-size", type=int, default=128, help="Sampling batch size.")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Optional override for sampler steps.")
    parser.add_argument("--sampling-solver", type=str, default=None, help="Optional override for sampler solver.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=500000,
        help="Max per-stage values kept for quantile estimation.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _append_sample(store: list[np.ndarray], values: np.ndarray, limit: int):
    if values.size == 0:
        return
    if sum(arr.size for arr in store) >= limit:
        return
    remaining = limit - sum(arr.size for arr in store)
    store.append(values[:remaining].astype(np.float64, copy=False))


def _summarize_stage(name: str, stage: dict) -> dict:
    event_sum = np.concatenate(stage["event_sum"]) if stage["event_sum"] else np.array([], dtype=np.float64)
    ratio = np.concatenate(stage["ratio"]) if stage["ratio"] else np.array([], dtype=np.float64)
    positive = np.concatenate(stage["positive_sample"]) if stage["positive_sample"] else np.array([], dtype=np.float64)
    near_cut = np.concatenate(stage["near_cut_sample"]) if stage["near_cut_sample"] else np.array([], dtype=np.float64)
    above_cut = np.concatenate(stage["above_cut_sample"]) if stage["above_cut_sample"] else np.array([], dtype=np.float64)

    def quantiles(values: np.ndarray):
        if values.size == 0:
            return None
        return {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)),
        }

    return {
        "name": name,
        "event_count": int(event_sum.size),
        "ratio_mean": float(ratio.mean()) if ratio.size else None,
        "ratio_p50": float(np.percentile(ratio, 50)) if ratio.size else None,
        "event_sum_mean": float(event_sum.mean()) if event_sum.size else None,
        "event_sum_p50": float(np.percentile(event_sum, 50)) if event_sum.size else None,
        "positive_fraction": float(stage["positive_count"] / stage["value_count"]) if stage["value_count"] else 0.0,
        "near_cut_fraction": float(stage["near_cut_count"] / stage["value_count"]) if stage["value_count"] else 0.0,
        "above_cut_fraction": float(stage["above_cut_count"] / stage["value_count"]) if stage["value_count"] else 0.0,
        "near_cut_energy_fraction": float(stage["near_cut_sum"] / stage["total_sum"]) if stage["total_sum"] > 0 else 0.0,
        "positive_stats": quantiles(positive),
        "near_cut_stats": quantiles(near_cut),
        "above_cut_stats": quantiles(above_cut),
    }


def _new_stage():
    return {
        "event_sum": [],
        "ratio": [],
        "positive_count": 0,
        "near_cut_count": 0,
        "above_cut_count": 0,
        "value_count": 0,
        "near_cut_sum": 0.0,
        "total_sum": 0.0,
        "positive_sample": [],
        "near_cut_sample": [],
        "above_cut_sample": [],
    }


def main():
    args = parse_args()
    set_seed(args.seed, all_gpus=True)

    model, cfg = load_model(args.model_path)
    preprocessor = CaloShowerPreprocessor(**cfg.preprocessing)
    condition_file = default_condition_file(cfg, args.condition_file).resolve()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    dataset = build_dataset(cfg, condition_file, args.num_showers)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    sampling_cfg = OmegaConf.to_container(cfg.sampling, resolve=True)
    if args.sampling_steps is not None:
        sampling_cfg["steps"] = args.sampling_steps
    if args.sampling_solver is not None:
        sampling_cfg["solver"] = args.sampling_solver

    noise_level = float(preprocessor.shower_preprocessor.noise_level)
    stages = {
        "after_LogTransform": _new_stage(),
        "after_CutNoise": _new_stage(),
    }

    def add_stage(name: str, tensor: torch.Tensor, energy: torch.Tensor):
        flat = tensor.detach().reshape(tensor.shape[0], -1)
        flat_np = flat.cpu().numpy().astype(np.float64, copy=False)
        energy_np = energy.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
        event_sum = flat_np.sum(axis=1)
        positive_mask = flat_np > 0.0
        near_cut_mask = positive_mask & (flat_np <= noise_level)
        above_cut_mask = flat_np > noise_level

        stage = stages[name]
        stage["event_sum"].append(event_sum)
        stage["ratio"].append(event_sum / np.maximum(energy_np, 1e-30))
        stage["positive_count"] += int(positive_mask.sum())
        stage["near_cut_count"] += int(near_cut_mask.sum())
        stage["above_cut_count"] += int(above_cut_mask.sum())
        stage["value_count"] += int(flat_np.size)
        stage["near_cut_sum"] += float(flat_np[near_cut_mask].sum())
        stage["total_sum"] += float(flat_np.sum())

        _append_sample(stage["positive_sample"], flat_np[positive_mask], args.sample_limit)
        _append_sample(stage["near_cut_sample"], flat_np[near_cut_mask], args.sample_limit)
        _append_sample(stage["above_cut_sample"], flat_np[above_cut_mask], args.sample_limit)

    before_cut_sums = []
    after_cut_sums = []

    with torch.inference_mode():
        for _showers, conditions in dataloader:
            incident_energy = conditions[0].to(device)
            _, transformed_conditions = preprocessor.transform(conditions=conditions)
            transformed_conditions = to_device(transformed_conditions, device)

            current = model.sample(
                conditions=transformed_conditions,
                progress=False,
                **filter_model_sample_kwargs(sampling_cfg),
            ).squeeze(1)

            for step in reversed(preprocessor.shower_preprocessor.pipeline):
                current = step.inverse_transform(current, transformed_conditions[0])
                class_name = type(step).__name__
                if class_name == "LogTransform":
                    add_stage("after_LogTransform", current, incident_energy)
                    before_cut_sums.append(current.detach().reshape(current.shape[0], -1).sum(dim=1).cpu().numpy())
                elif class_name == "CutNoise":
                    add_stage("after_CutNoise", current, incident_energy)
                    after_cut_sums.append(current.detach().reshape(current.shape[0], -1).sum(dim=1).cpu().numpy())

    before_cut_sums = np.concatenate(before_cut_sums) if before_cut_sums else np.array([], dtype=np.float64)
    after_cut_sums = np.concatenate(after_cut_sums) if after_cut_sums else np.array([], dtype=np.float64)
    relative_drop = (before_cut_sums - after_cut_sums) / np.maximum(before_cut_sums, 1e-30)

    output = {
        "model_path": str(args.model_path.resolve()),
        "condition_file": str(condition_file),
        "device": str(device),
        "num_showers": int(args.num_showers),
        "batch_size": int(args.batch_size),
        "sampling": {
            "steps": int(sampling_cfg["steps"]),
            "solver": str(sampling_cfg["solver"]),
        },
        "noise_level_internal_GeV": noise_level,
        "noise_level_evaluator_MeV": float(
            get_evaluator_threshold_from_internal_noise(noise_level, geometry=str(dataset.ccd_geometry or ""))
        ),
        "stages": {name: _summarize_stage(name, stage) for name, stage in stages.items()},
        "cut_effect": {
            "mean_relative_drop": float(relative_drop.mean()) if relative_drop.size else None,
            "median_relative_drop": float(np.percentile(relative_drop, 50)) if relative_drop.size else None,
            "max_relative_drop": float(relative_drop.max()) if relative_drop.size else None,
            "mean_absolute_drop_GeV": float((before_cut_sums - after_cut_sums).mean()) if relative_drop.size else None,
            "median_absolute_drop_GeV": float(np.percentile(before_cut_sums - after_cut_sums, 50)) if relative_drop.size else None,
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2))

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
