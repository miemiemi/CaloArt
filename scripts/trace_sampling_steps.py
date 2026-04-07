#!/usr/bin/env python3
"""Trace sampling-time values for one event/cell through the ODE solver."""

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
    parser = argparse.ArgumentParser(description="Trace one event/cell through sampling steps.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to an exported model with config.")
    parser.add_argument(
        "--condition-file",
        type=Path,
        default=None,
        help="Optional HDF5 file providing conditioning energies.",
    )
    parser.add_argument("--num-showers", type=int, default=10000, help="Number of showers to sample.")
    parser.add_argument("--batch-size", type=int, default=128, help="Sampling batch size.")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Optional override for sampler steps.")
    parser.add_argument("--sampling-solver", type=str, default=None, help="Optional override for sampler solver.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--target-event", type=int, required=True, help="Global event index to trace.")
    parser.add_argument("--target-cell", type=int, required=True, help="Flattened cell index within one event.")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to write the trace JSON.")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda:0 or cpu.")
    return parser.parse_args()


def _flatten_event_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 2 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor.reshape(tensor.shape[0], -1)


def _scalar_at(tensor: torch.Tensor, local_idx: int, cell_idx: int) -> float:
    flat = _flatten_event_tensor(tensor)
    return float(flat[local_idx, cell_idx].item())


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


def _velocity_components(method, x, conditions, t):
    if method.predict_mode == "pred_x1":
        model_t = t
    else:
        model_t = t.clamp(method.t_eps, 1.0 - method.t_eps)

    raw_pred = method.model(x, conditions, model_t)
    from_type = method.predict_mode.replace("pred_", "")
    if from_type == "v":
        velocity = raw_pred
    else:
        velocity = method._convert(raw_pred, x, model_t, from_type, "v")
    return raw_pred, velocity, model_t


def trace_euler(method, x, conditions, local_idx: int, cell_idx: int, steps: int):
    dt = 1.0 / steps
    records = []
    for i in range(steps):
        t_cur = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
        x_before = x
        raw_pred1, v1, model_t_cur = _velocity_components(method, x_before, conditions, t_cur)
        x_after = x_before + v1 * dt
        records.append(
            {
                "step": i,
                "t_cur": float(i * dt),
                "t_cur_used": float(model_t_cur[local_idx].item()),
                "x_before": _scalar_at(x_before, local_idx, cell_idx),
                "raw_pred1": _scalar_at(raw_pred1, local_idx, cell_idx),
                "v1": _scalar_at(v1, local_idx, cell_idx),
                "x_after": _scalar_at(x_after, local_idx, cell_idx),
            }
        )
        x = x_after
    return x, records


def trace_heun(method, x, conditions, local_idx: int, cell_idx: int, steps: int):
    dt = 1.0 / steps
    records = []
    for i in range(steps):
        t_cur = i * dt
        t_next = (i + 1) * dt

        t_cur_batch = torch.full((x.shape[0],), t_cur, device=x.device, dtype=x.dtype)
        x_before = x
        raw_pred1, v1, model_t_cur = _velocity_components(method, x_before, conditions, t_cur_batch)
        x_pred = x_before + v1 * dt

        record = {
            "step": i,
            "t_cur": float(t_cur),
            "t_cur_used": float(model_t_cur[local_idx].item()),
            "x_before": _scalar_at(x_before, local_idx, cell_idx),
            "raw_pred1": _scalar_at(raw_pred1, local_idx, cell_idx),
            "v1": _scalar_at(v1, local_idx, cell_idx),
            "x_pred": _scalar_at(x_pred, local_idx, cell_idx),
        }

        if t_next >= 1.0:
            x = x_pred
            record["x_after"] = _scalar_at(x, local_idx, cell_idx)
        else:
            t_next_batch = torch.full((x.shape[0],), t_next, device=x.device, dtype=x.dtype)
            raw_pred2, v2, model_t_next = _velocity_components(method, x_pred, conditions, t_next_batch)
            x = x_before + 0.5 * (v1 + v2) * dt
            record["x_after"] = _scalar_at(x, local_idx, cell_idx)
            record["t_next_used"] = float(model_t_next[local_idx].item())
            record["raw_pred2"] = _scalar_at(raw_pred2, local_idx, cell_idx)
            record["v2"] = _scalar_at(v2, local_idx, cell_idx)

        records.append(record)
    return x, records


def trace_midpoint(method, x, conditions, local_idx: int, cell_idx: int, steps: int):
    dt = 1.0 / steps
    records = []
    for i in range(steps):
        t_cur = i * dt
        t_mid = t_cur + 0.5 * dt
        t_cur_batch = torch.full((x.shape[0],), t_cur, device=x.device, dtype=x.dtype)
        x_before = x
        raw_pred1, v1, model_t_cur = _velocity_components(method, x_before, conditions, t_cur_batch)
        x_mid = x_before + 0.5 * dt * v1
        t_mid_batch = torch.full((x.shape[0],), t_mid, device=x.device, dtype=x.dtype)
        raw_pred2, v_mid, model_t_mid = _velocity_components(method, x_mid, conditions, t_mid_batch)
        x = x_before + v_mid * dt
        records.append(
            {
                "step": i,
                "t_cur": float(t_cur),
                "t_cur_used": float(model_t_cur[local_idx].item()),
                "x_before": _scalar_at(x_before, local_idx, cell_idx),
                "raw_pred1": _scalar_at(raw_pred1, local_idx, cell_idx),
                "v1": _scalar_at(v1, local_idx, cell_idx),
                "x_mid": _scalar_at(x_mid, local_idx, cell_idx),
                "t_mid_used": float(model_t_mid[local_idx].item()),
                "raw_pred2": _scalar_at(raw_pred2, local_idx, cell_idx),
                "v_mid": _scalar_at(v_mid, local_idx, cell_idx),
                "x_after": _scalar_at(x, local_idx, cell_idx),
            }
        )
    return x, records


def trace_rk4(method, x, conditions, local_idx: int, cell_idx: int, steps: int):
    dt = 1.0 / steps
    records = []
    for i in range(steps):
        t_cur = i * dt
        t_mid = t_cur + 0.5 * dt
        t_next = (i + 1) * dt

        x_before = x
        t_cur_batch = torch.full((x.shape[0],), t_cur, device=x.device, dtype=x.dtype)
        raw_pred1, k1, model_t_cur = _velocity_components(method, x_before, conditions, t_cur_batch)

        x_k2 = x_before + 0.5 * dt * k1
        t_mid_batch = torch.full((x.shape[0],), t_mid, device=x.device, dtype=x.dtype)
        raw_pred2, k2, model_t_mid_1 = _velocity_components(method, x_k2, conditions, t_mid_batch)

        x_k3 = x_before + 0.5 * dt * k2
        raw_pred3, k3, model_t_mid_2 = _velocity_components(method, x_k3, conditions, t_mid_batch)

        x_k4 = x_before + dt * k3
        t_next_batch = torch.full((x.shape[0],), t_next, device=x.device, dtype=x.dtype)
        raw_pred4, k4, model_t_next = _velocity_components(method, x_k4, conditions, t_next_batch)

        x = x_before + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        records.append(
            {
                "step": i,
                "t_cur": float(t_cur),
                "t_cur_used": float(model_t_cur[local_idx].item()),
                "x_before": _scalar_at(x_before, local_idx, cell_idx),
                "raw_pred1": _scalar_at(raw_pred1, local_idx, cell_idx),
                "k1": _scalar_at(k1, local_idx, cell_idx),
                "x_k2": _scalar_at(x_k2, local_idx, cell_idx),
                "t_mid_used_1": float(model_t_mid_1[local_idx].item()),
                "raw_pred2": _scalar_at(raw_pred2, local_idx, cell_idx),
                "k2": _scalar_at(k2, local_idx, cell_idx),
                "x_k3": _scalar_at(x_k3, local_idx, cell_idx),
                "t_mid_used_2": float(model_t_mid_2[local_idx].item()),
                "raw_pred3": _scalar_at(raw_pred3, local_idx, cell_idx),
                "k3": _scalar_at(k3, local_idx, cell_idx),
                "x_k4": _scalar_at(x_k4, local_idx, cell_idx),
                "t_next_used": float(model_t_next[local_idx].item()),
                "raw_pred4": _scalar_at(raw_pred4, local_idx, cell_idx),
                "k4": _scalar_at(k4, local_idx, cell_idx),
                "x_after": _scalar_at(x, local_idx, cell_idx),
            }
        )
    return x, records


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

    steps = int(sampling_cfg["steps"])
    solver = str(sampling_cfg["solver"])
    trace_impl = {
        "euler": trace_euler,
        "heun": trace_heun,
        "midpoint": trace_midpoint,
        "rk4": trace_rk4,
    }[solver]

    seen_events = 0
    target = None
    with torch.inference_mode():
        for _, conditions in dataloader:
            batch_size = int(conditions[0].shape[0])

            if args.target_event >= seen_events + batch_size:
                x_shape = (batch_size, *model.input_size)
                torch.randn(x_shape, device=device)
                seen_events += batch_size
                continue

            local_idx = args.target_event - seen_events
            _, transformed_conditions = preprocessor.transform(conditions=conditions)
            transformed_conditions = to_device(transformed_conditions, device)

            x_shape = (batch_size, *model.input_size)
            x0 = torch.randn(x_shape, device=device) * model.noise_scale
            traced_x, step_records = trace_impl(
                model,
                x0.clone(),
                transformed_conditions,
                local_idx,
                args.target_cell,
                steps,
            )
            sample_output = traced_x.squeeze(1)

            inverse_stage_values = {}

            def trace_fn(class_name: str, tensor: torch.Tensor):
                flat = tensor.reshape(tensor.shape[0], -1)
                inverse_stage_values[class_name] = float(flat[local_idx, args.target_cell].item())

            preprocessor.inverse_transform(
                showers=sample_output,
                conditions=transformed_conditions,
                trace_fn=trace_fn,
            )

            target = {
                "event": args.target_event,
                "cell": args.target_cell,
                "local_idx": int(local_idx),
                "coords": _coords_for(sample_output, args.target_cell),
                "initial_x0": _scalar_at(x0, local_idx, args.target_cell),
                "final_sample_output": _scalar_at(sample_output, local_idx, args.target_cell),
                "after_standardize_inverse": inverse_stage_values.get("Standarize"),
                "after_log_inverse": inverse_stage_values.get("LogTransform"),
                "steps": step_records,
            }
            break

    if target is None:
        raise ValueError(
            f"Target event {args.target_event} was not found within num_showers={args.num_showers}."
        )

    summary = {
        "model_path": str(args.model_path.resolve()),
        "condition_file": str(condition_file),
        "num_showers": args.num_showers,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "sampling": sampling_cfg,
        "target": target,
    }

    print(json.dumps(summary, indent=2))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\njson_written: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
