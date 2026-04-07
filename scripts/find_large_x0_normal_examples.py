#!/usr/bin/env python3
"""Find voxels with large initial x0 that still end with normal outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rootutils
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, pythonpath=True)

from diagnose_sampling_values import build_dataset, default_condition_file, load_model
from src.data.preprocessing import CaloShowerPreprocessor
from src.utils import set_seed, to_device


def parse_args():
    parser = argparse.ArgumentParser(description="Find large-x0 but normal-final voxel examples.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to an exported model with config.")
    parser.add_argument("--condition-file", type=Path, default=None, help="Optional HDF5 condition file.")
    parser.add_argument("--num-showers", type=int, default=1000, help="Number of showers to inspect.")
    parser.add_argument("--batch-size", type=int, default=128, help="Sampling batch size.")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Optional sampler steps override.")
    parser.add_argument("--sampling-solver", type=str, default=None, help="Optional sampler solver override.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--x0-threshold", type=float, default=4.0, help="Threshold for initial x0.")
    parser.add_argument(
        "--final-abs-threshold",
        type=float,
        default=1.0,
        help="Require abs(final_sample_output) <= this threshold.",
    )
    parser.add_argument(
        "--max-log-inverse",
        type=float,
        default=1.0,
        help="Require abs(after_log_inverse) <= this threshold.",
    )
    parser.add_argument("--max-examples", type=int, default=10, help="Maximum number of examples to keep.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to write JSON output.")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def _coords_for(sample_tensor: torch.Tensor, cell_idx: int):
    sample_shape = sample_tensor.shape[1:]
    coords = []
    remaining = cell_idx
    strides = []
    acc = 1
    for size in reversed(sample_shape[1:]):
        acc *= size
        strides.append(acc)
    strides = list(reversed(strides))
    for dim, size in enumerate(sample_shape):
        stride = strides[dim] if dim < len(strides) else 1
        coord = remaining // stride
        remaining = remaining % stride
        coords.append(int(coord))
    labels = ["r", "phi", "z"] if len(coords) == 3 else [f"dim_{i}" for i in range(len(coords))]
    return {label: value for label, value in zip(labels, coords)}


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

    examples = []
    seen_events = 0
    with torch.inference_mode():
        for _, conditions in dataloader:
            batch_size = int(conditions[0].shape[0])
            _, transformed_conditions = preprocessor.transform(conditions=conditions)
            transformed_conditions = to_device(transformed_conditions, device)

            x_shape = (batch_size, *model.input_size)
            x0 = torch.randn(x_shape, device=device) * model.noise_scale
            sample_output = model.sample(
                conditions=transformed_conditions,
                progress=False,
                **sampling_cfg,
            ).squeeze(1)

            inverse_values = {}

            def trace_fn(class_name: str, tensor: torch.Tensor):
                inverse_values[class_name] = tensor.reshape(tensor.shape[0], -1)

            preprocessor.inverse_transform(
                showers=sample_output,
                conditions=transformed_conditions,
                trace_fn=trace_fn,
            )

            x0_flat = x0.squeeze(1).reshape(batch_size, -1)
            final_flat = sample_output.reshape(batch_size, -1)
            stdinv_flat = inverse_values["Standarize"]
            loginv_flat = inverse_values["LogTransform"]

            mask = (
                (x0_flat > args.x0_threshold)
                & (final_flat.abs() <= args.final_abs_threshold)
                & torch.isfinite(loginv_flat)
                & (loginv_flat.abs() <= args.max_log_inverse)
            )

            coords = torch.nonzero(mask, as_tuple=False)
            for local_event, cell in coords.cpu().tolist():
                item = {
                    "event": seen_events + int(local_event),
                    "cell": int(cell),
                    "coords": _coords_for(sample_output, int(cell)),
                    "initial_x0": float(x0_flat[local_event, cell].item()),
                    "final_sample_output": float(final_flat[local_event, cell].item()),
                    "after_standardize_inverse": float(stdinv_flat[local_event, cell].item()),
                    "after_log_inverse": float(loginv_flat[local_event, cell].item()),
                }
                examples.append(item)
                if len(examples) >= args.max_examples:
                    break
            if len(examples) >= args.max_examples:
                break
            seen_events += batch_size

    summary = {
        "model_path": str(args.model_path.resolve()),
        "condition_file": str(condition_file),
        "num_showers": args.num_showers,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "sampling": sampling_cfg,
        "criteria": {
            "x0_threshold": args.x0_threshold,
            "final_abs_threshold": args.final_abs_threshold,
            "max_log_inverse": args.max_log_inverse,
        },
        "num_examples_found": len(examples),
        "examples": examples,
    }

    print(json.dumps(summary, indent=2))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\njson_written: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
