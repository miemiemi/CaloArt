#!/usr/bin/env python3
"""Compute preprocessing statistics for CaloChallenge-style HDF5 showers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute log-space preprocessing statistics from raw HDF5 shower files."
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        required=True,
        help="Input HDF5 files to treat as one combined training set.",
    )
    parser.add_argument(
        "--sampling-fraction",
        type=float,
        default=0.033,
        help="Internal CCD sampling fraction applied after MeV->GeV conversion.",
    )
    parser.add_argument(
        "--chunk-showers",
        type=int,
        default=64,
        help="Number of showers to load per chunk.",
    )
    parser.add_argument(
        "--transform-type",
        type=str,
        default="log",
        choices=["log", "log1p", "asinh"],
        help="Value transform applied after optional cut/noise/scale steps.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[1.0e-8, 1.0e-7, 5.0e-7, 1.0e-6],
        help="Candidate eps values for LogTransform statistics.",
    )
    parser.add_argument(
        "--transform-scale",
        type=float,
        default=5.0e-7,
        help="Scale parameter used for Log1pTransform or AsinhTransform.",
    )
    parser.add_argument(
        "--noise-thresholds",
        type=float,
        nargs="+",
        default=[1.65e-7, 5.0e-7],
        help="Thresholds to count below-threshold cells for.",
    )
    parser.add_argument(
        "--suggested-noise-level",
        type=float,
        default=5.0e-7,
        help="Noise level to use in emitted YAML snippets.",
    )
    parser.add_argument(
        "--raw-cut-noise-level",
        type=float,
        default=0.0,
        help="Optional raw-space CutNoise level applied before any later transforms.",
    )
    parser.add_argument(
        "--add-noise-level",
        type=float,
        default=0.0,
        help="Optional one-sided uniform AddNoise level applied after raw cut and before LogTransform.",
    )
    parser.add_argument(
        "--scale-above-cut-factor",
        type=float,
        default=1.0,
        help="Optional multiplicative factor applied to values above scale-above-cut-threshold before the value transform.",
    )
    parser.add_argument(
        "--scale-above-cut-threshold",
        type=float,
        default=5.0e-7,
        help="Threshold used by ScaleAboveCut before the value transform.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when --add-noise-level is enabled.",
    )
    parser.add_argument(
        "--scale-by-incident-energy",
        action="store_true",
        help="Apply ScaleByIncidentEnergy before LogTransform.",
    )
    parser.add_argument(
        "--scale-by-factor",
        type=float,
        default=1.0,
        help="Apply ScaleByFactor before LogTransform.",
    )
    parser.add_argument(
        "--show-quantiles",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.01, 0.05, 0.5],
        help="Quantiles to report for positive cells from a small stratified sample.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the structured summary as JSON.",
    )
    return parser.parse_args()


def fmt_float(value: float) -> str:
    return f"{value:.8g}"


def transform_label(transform_type: str, value: float) -> str:
    if transform_type == "log":
        return f"eps={value:.8g}"
    return f"scale={value:.8g}"


def apply_value_transform(flat: np.ndarray, transform_type: str, value: float) -> np.ndarray:
    if transform_type == "log":
        return np.log(flat + value, dtype=np.float64)
    if transform_type == "log1p":
        return np.log1p(flat / value, dtype=np.float64)
    if transform_type == "asinh":
        return np.arcsinh(flat / value, dtype=np.float64)
    raise ValueError(f"Unsupported transform_type: {transform_type}")


def stream_stats(
    files: list[Path],
    sampling_fraction: float,
    chunk_showers: int,
    transform_type: str,
    transform_values: list[float],
    noise_thresholds: list[float],
    raw_cut_noise_level: float,
    add_noise_level: float,
    scale_above_cut_factor: float,
    scale_above_cut_threshold: float,
    scale_by_incident_energy: bool,
    scale_by_factor: float,
    random_seed: int,
):
    summary = {value: {"n": 0, "sum": 0.0, "sumsq": 0.0} for value in transform_values}
    lt_counts = {thr: 0 for thr in noise_thresholds}
    pos_lt_counts = {thr: 0 for thr in noise_thresholds}
    total_showers = 0
    total_cells = 0
    zero_cells = 0
    min_positive = None
    rng = np.random.default_rng(random_seed)

    for file_path in files:
        with h5py.File(file_path, "r") as handle:
            showers = handle["showers"]
            energies = handle["incident_energies"]
            total_showers += showers.shape[0]
            for start in range(0, showers.shape[0], chunk_showers):
                end = min(start + chunk_showers, showers.shape[0])
                flat = showers[start:end].astype(np.float32).reshape(-1)
                flat = flat / 1000.0 * sampling_fraction
                if raw_cut_noise_level > 0.0:
                    flat = np.where(flat < raw_cut_noise_level, 0.0, flat)
                if add_noise_level > 0.0:
                    flat = flat + rng.random(flat.shape, dtype=np.float32) * add_noise_level
                if scale_above_cut_factor != 1.0:
                    flat = np.where(
                        flat > scale_above_cut_threshold,
                        flat * scale_above_cut_factor,
                        flat,
                    )
                if scale_by_incident_energy:
                    energy = energies[start:end].astype(np.float32).reshape(-1) / 1000.0
                    flat = (
                        flat.reshape(end - start, -1)
                        / energy[:, None]
                    ).reshape(-1)
                if scale_by_factor != 1.0:
                    flat = flat / scale_by_factor

                total_cells += flat.size
                zero_cells += int((flat == 0).sum())

                positive = flat[flat > 0]
                if positive.size:
                    local_min = float(positive.min())
                    min_positive = local_min if min_positive is None else min(min_positive, local_min)

                for threshold in noise_thresholds:
                    lt_counts[threshold] += int((flat < threshold).sum())
                    pos_lt_counts[threshold] += int(((flat > 0) & (flat < threshold)).sum())

                for value in transform_values:
                    transformed = apply_value_transform(flat, transform_type, value)
                    current = summary[value]
                    current["n"] += transformed.size
                    current["sum"] += float(transformed.sum(dtype=np.float64))
                    current["sumsq"] += float((transformed * transformed).sum(dtype=np.float64))

    return {
        "summary": summary,
        "lt_counts": lt_counts,
        "pos_lt_counts": pos_lt_counts,
        "total_showers": total_showers,
        "total_cells": total_cells,
        "zero_cells": zero_cells,
        "min_positive": min_positive,
    }


def stratified_positive_quantiles(
    files: list[Path],
    sampling_fraction: float,
    quantiles: list[float],
    raw_cut_noise_level: float,
    add_noise_level: float,
    scale_above_cut_factor: float,
    scale_above_cut_threshold: float,
    scale_by_incident_energy: bool,
    scale_by_factor: float,
    random_seed: int,
):
    parts = []
    window = 16
    rng = np.random.default_rng(random_seed)
    for file_path in files:
        with h5py.File(file_path, "r") as handle:
            showers = handle["showers"]
            energies = handle["incident_energies"]
            n_showers = showers.shape[0]
            starts = [0, n_showers // 8, n_showers // 4, 3 * n_showers // 8, n_showers // 2, 5 * n_showers // 8, 3 * n_showers // 4, 7 * n_showers // 8, n_showers - window]
            for start in starts:
                start = max(0, min(start, n_showers - window))
                part = showers[start:start + window].astype(np.float32)
                part = part / 1000.0 * sampling_fraction
                if raw_cut_noise_level > 0.0:
                    part = np.where(part < raw_cut_noise_level, 0.0, part)
                if add_noise_level > 0.0:
                    part = part + rng.random(part.shape, dtype=np.float32) * add_noise_level
                if scale_above_cut_factor != 1.0:
                    part = np.where(
                        part > scale_above_cut_threshold,
                        part * scale_above_cut_factor,
                        part,
                    )
                if scale_by_incident_energy:
                    energy = energies[start:start + window].astype(np.float32).reshape(-1, 1) / 1000.0
                    part = part / energy
                if scale_by_factor != 1.0:
                    part = part / scale_by_factor
                part = part.reshape(-1)
                parts.append(part)

    flat = np.concatenate(parts)
    positive = flat[flat > 0]
    return np.quantile(positive, quantiles), len(parts) * window


def emit_yaml(
    transform_type: str,
    noise_level: float,
    add_noise_level: float,
    scale_above_cut_factor: float,
    scale_above_cut_threshold: float,
    transform_value: float,
    mean: float,
    std: float,
    scale_by_incident_energy: bool,
    scale_by_factor: float,
):
    print("preprocessing:")
    print("  steps:")
    print("    - class_name: src.data.preprocessing.CutNoise")
    print("      init_args:")
    print(f"        noise_level: {noise_level:.8g}")
    print("        both_directions: true")
    if add_noise_level > 0.0:
        print("    - class_name: src.data.preprocessing.AddNoise")
        print("      init_args:")
        print(f"        noise_level: {add_noise_level:.8g}")
    if scale_above_cut_factor != 1.0:
        print("    - class_name: src.data.preprocessing.ScaleAboveCut")
        print("      init_args:")
        print(f"        factor: {scale_above_cut_factor:.8g}")
        print(f"        threshold: {scale_above_cut_threshold:.8g}")
    if scale_by_incident_energy:
        print("    - class_name: src.data.preprocessing.ScaleByIncidentEnergy")
    if scale_by_factor != 1.0:
        print("    - class_name: src.data.preprocessing.ScaleByFactor")
        print("      init_args:")
        print(f"        factor: {scale_by_factor:.8g}")
    if transform_type == "log":
        print("    - class_name: src.data.preprocessing.LogTransform")
        print("      init_args:")
        print(f"        eps: {transform_value:.8g}")
    elif transform_type == "log1p":
        print("    - class_name: src.data.preprocessing.Log1pTransform")
        print("      init_args:")
        print(f"        scale: {transform_value:.8g}")
    elif transform_type == "asinh":
        print("    - class_name: src.data.preprocessing.AsinhTransform")
        print("      init_args:")
        print(f"        scale: {transform_value:.8g}")
    else:
        raise ValueError(f"Unsupported transform_type: {transform_type}")
    print("    - class_name: src.data.preprocessing.Standarize")
    print("      init_args:")
    print(f"        mean: {mean:.7f}")
    print(f"        std: {std:.7f}")
    print("keep_condition_components: [energy]")


def main():
    args = parse_args()
    files = [file_path.resolve() for file_path in args.files]
    transform_values = args.eps if args.transform_type == "log" else [args.transform_scale]

    print("input_files:")
    for file_path in files:
        print(f"  - {file_path}")

    stats = stream_stats(
        files=files,
        sampling_fraction=args.sampling_fraction,
        chunk_showers=args.chunk_showers,
        transform_type=args.transform_type,
        transform_values=transform_values,
        noise_thresholds=args.noise_thresholds,
        raw_cut_noise_level=args.raw_cut_noise_level,
        add_noise_level=args.add_noise_level,
        scale_above_cut_factor=args.scale_above_cut_factor,
        scale_above_cut_threshold=args.scale_above_cut_threshold,
        scale_by_incident_energy=args.scale_by_incident_energy,
        scale_by_factor=args.scale_by_factor,
        random_seed=args.random_seed,
    )

    quantiles, sample_showers = stratified_positive_quantiles(
        files=files,
        sampling_fraction=args.sampling_fraction,
        quantiles=args.show_quantiles,
        raw_cut_noise_level=args.raw_cut_noise_level,
        add_noise_level=args.add_noise_level,
        scale_above_cut_factor=args.scale_above_cut_factor,
        scale_above_cut_threshold=args.scale_above_cut_threshold,
        scale_by_incident_energy=args.scale_by_incident_energy,
        scale_by_factor=args.scale_by_factor,
        random_seed=args.random_seed,
    )

    total_cells = stats["total_cells"]
    zero_cells = stats["zero_cells"]
    candidate_stats = []
    print(f"total_showers={stats['total_showers']}")
    print(f"total_cells={total_cells}")
    print(f"zero_cells={zero_cells} ({zero_cells / total_cells * 100:.6f}%)")
    print(f"min_positive={stats['min_positive']:.12e}")
    print(f"raw_cut_noise_level={args.raw_cut_noise_level:.8g}")
    print(f"add_noise_level={args.add_noise_level:.8g}")
    print(f"scale_above_cut_factor={args.scale_above_cut_factor:.8g}")
    print(f"scale_above_cut_threshold={args.scale_above_cut_threshold:.8g}")
    print(f"scale_by_incident_energy={args.scale_by_incident_energy}")
    print(f"scale_by_factor={args.scale_by_factor:.8g}")
    print(f"random_seed={args.random_seed}")
    print(f"transform_type={args.transform_type}")
    if args.transform_type != "log":
        print(f"transform_scale={args.transform_scale:.8g}")

    for threshold in args.noise_thresholds:
        lt = stats["lt_counts"][threshold]
        pos_lt = stats["pos_lt_counts"][threshold]
        print(f"cells_lt_{threshold:.2e}={lt} ({lt / total_cells * 100:.6f}%)")
        print(f"positive_cells_lt_{threshold:.2e}={pos_lt} ({pos_lt / total_cells * 100:.6f}%)")

    print(f"stratified_sample_showers={sample_showers}")
    print(
        "positive_quantiles="
        + " ".join(
            f"q{quantile:g}={value:.6e}" for quantile, value in zip(args.show_quantiles, quantiles)
        )
    )

    print("candidate_stats:")
    for value in transform_values:
        current = stats["summary"][value]
        mean = current["sum"] / current["n"]
        variance = current["sumsq"] / current["n"] - mean * mean
        std = math.sqrt(max(variance, 0.0))
        candidate_stats.append(
            {
                "transform_type": args.transform_type,
                "value": float(value),
                "mean": float(mean),
                "std": float(std),
            }
        )
        print(f"  {transform_label(args.transform_type, value)} mean={mean:.7f} std={std:.7f}")

    print("\n# Candidate preprocessing snippets")
    for value in transform_values:
        current = stats["summary"][value]
        mean = current["sum"] / current["n"]
        variance = current["sumsq"] / current["n"] - mean * mean
        std = math.sqrt(max(variance, 0.0))
        print(f"\n# {transform_label(args.transform_type, value)}")
        emit_yaml(
            transform_type=args.transform_type,
            noise_level=args.raw_cut_noise_level if args.raw_cut_noise_level > 0.0 else args.suggested_noise_level,
            add_noise_level=args.add_noise_level,
            scale_above_cut_factor=args.scale_above_cut_factor,
            scale_above_cut_threshold=args.scale_above_cut_threshold,
            transform_value=value,
            mean=mean,
            std=std,
            scale_by_incident_energy=args.scale_by_incident_energy,
            scale_by_factor=args.scale_by_factor,
        )

    if args.output_json is not None:
        result = {
            "files": [str(file_path) for file_path in files],
            "sampling_fraction": float(args.sampling_fraction),
            "chunk_showers": int(args.chunk_showers),
            "transform_type": args.transform_type,
            "eps": [float(eps) for eps in args.eps],
            "transform_scale": float(args.transform_scale),
            "noise_thresholds": [float(threshold) for threshold in args.noise_thresholds],
            "raw_cut_noise_level": float(args.raw_cut_noise_level),
            "add_noise_level": float(args.add_noise_level),
            "scale_above_cut_factor": float(args.scale_above_cut_factor),
            "scale_above_cut_threshold": float(args.scale_above_cut_threshold),
            "scale_by_incident_energy": bool(args.scale_by_incident_energy),
            "scale_by_factor": float(args.scale_by_factor),
            "random_seed": int(args.random_seed),
            "total_showers": int(stats["total_showers"]),
            "total_cells": int(total_cells),
            "zero_cells": int(zero_cells),
            "zero_fraction": float(zero_cells / total_cells),
            "min_positive": float(stats["min_positive"]),
            "positive_quantiles": {
                f"q{quantile:g}": float(value)
                for quantile, value in zip(args.show_quantiles, quantiles)
            },
            "candidate_stats": candidate_stats,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote_json={args.output_json.resolve()}")


if __name__ == "__main__":
    main()
