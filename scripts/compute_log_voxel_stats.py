#!/usr/bin/env python3
"""Compute raw and transformed voxel statistics for CCD HDF5 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compute voxel stats for CCD datasets.")
    parser.add_argument("--files", type=Path, nargs="+", required=True, help="Input HDF5 files.")
    parser.add_argument(
        "--sampling-fraction",
        type=float,
        default=0.033,
        help="Internal CCD sampling fraction applied after MeV->GeV conversion.",
    )
    parser.add_argument("--chunk-showers", type=int, default=1024, help="Number of showers per chunk.")
    parser.add_argument(
        "--raw-cut-noise-level",
        type=float,
        default=5.0e-7,
        help="Apply CutNoise in internal GeV before transformed statistics.",
    )
    parser.add_argument(
        "--transform-type",
        type=str,
        default="log",
        choices=["log", "log1p", "asinh"],
        help="Value transform applied after optional raw-space cut.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0e-6,
        help="Epsilon used in log(x + eps) when --transform-type=log.",
    )
    parser.add_argument(
        "--transform-scale",
        type=float,
        default=5.0e-7,
        help="Scale used in log1p(x / scale) or asinh(x / scale).",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def update_running(acc, values: np.ndarray):
    if values.size == 0:
        return
    acc["count"] += int(values.size)
    values64 = values.astype(np.float64, copy=False)
    acc["sum"] += float(values64.sum(dtype=np.float64))
    acc["sumsq"] += float((values64 * values64).sum(dtype=np.float64))
    acc["min"] = float(values64.min()) if acc["min"] is None else min(acc["min"], float(values64.min()))
    acc["max"] = float(values64.max()) if acc["max"] is None else max(acc["max"], float(values64.max()))


def summary_from_acc(acc):
    if acc["count"] == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = acc["sum"] / acc["count"]
    variance = max(acc["sumsq"] / acc["count"] - mean * mean, 0.0)
    return {
        "count": int(acc["count"]),
        "mean": float(mean),
        "std": float(variance ** 0.5),
        "min": float(acc["min"]),
        "max": float(acc["max"]),
    }


def transform_values(flat: np.ndarray, transform_type: str, eps: float, transform_scale: float) -> np.ndarray:
    if transform_type == "log":
        return np.log(flat + eps, dtype=np.float64)
    if transform_type == "log1p":
        return np.log1p(flat / transform_scale, dtype=np.float64)
    if transform_type == "asinh":
        return np.arcsinh(flat / transform_scale)
    raise ValueError(f"Unsupported transform_type: {transform_type}")


def transform_suffix(transform_type: str) -> str:
    if transform_type == "log":
        return "log_x_plus_eps"
    if transform_type == "log1p":
        return "log1p_x_over_scale"
    if transform_type == "asinh":
        return "asinh_x_over_scale"
    raise ValueError(f"Unsupported transform_type: {transform_type}")


def main():
    args = parse_args()
    suffix = transform_suffix(args.transform_type)

    result = {
        "files": [str(path.resolve()) for path in args.files],
        "sampling_fraction": float(args.sampling_fraction),
        "chunk_showers": int(args.chunk_showers),
        "raw_cut_noise_level": float(args.raw_cut_noise_level),
        "transform_type": args.transform_type,
        "eps": float(args.eps),
        "transform_scale": float(args.transform_scale),
        "per_file": {},
    }

    for path in args.files:
        all_acc = {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}
        pos_acc = {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}
        raw_all_acc = {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}
        raw_pos_acc = {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}

        with h5py.File(path, "r") as handle:
            showers = handle["showers"]
            for start in range(0, showers.shape[0], args.chunk_showers):
                end = min(start + args.chunk_showers, showers.shape[0])
                flat = showers[start:end].astype(np.float32)
                flat = flat / 1000.0 * args.sampling_fraction
                if args.raw_cut_noise_level > 0.0:
                    flat = np.where(flat < args.raw_cut_noise_level, 0.0, flat)

                update_running(raw_all_acc, flat.reshape(-1))
                transformed_all = transform_values(
                    flat,
                    transform_type=args.transform_type,
                    eps=args.eps,
                    transform_scale=args.transform_scale,
                ).reshape(-1)
                update_running(all_acc, transformed_all)

                positive = flat[flat > 0.0]
                update_running(raw_pos_acc, positive.reshape(-1))
                if positive.size:
                    transformed_pos = transform_values(
                        positive,
                        transform_type=args.transform_type,
                        eps=args.eps,
                        transform_scale=args.transform_scale,
                    ).reshape(-1)
                    update_running(pos_acc, transformed_pos)

        file_summary = {
            "all_voxels_raw": summary_from_acc(raw_all_acc),
            "positive_voxels_raw": summary_from_acc(raw_pos_acc),
            f"all_voxels_{suffix}": summary_from_acc(all_acc),
            f"positive_voxels_{suffix}": summary_from_acc(pos_acc),
        }
        result["per_file"][str(path.resolve())] = file_summary

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
        print(f"wrote_json={args.output_json.resolve()}")


if __name__ == "__main__":
    main()
