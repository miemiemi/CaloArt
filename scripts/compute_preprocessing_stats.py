#!/usr/bin/env python3
"""Compute preprocessing statistics for CaloChallenge-style HDF5 showers."""

from __future__ import annotations

import argparse
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
        "--eps",
        type=float,
        nargs="+",
        default=[1.0e-8, 1.0e-7, 5.0e-7, 1.0e-6],
        help="Candidate eps values for LogTransform statistics.",
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
        "--show-quantiles",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.01, 0.05, 0.5],
        help="Quantiles to report for positive cells from a small stratified sample.",
    )
    return parser.parse_args()


def fmt_float(value: float) -> str:
    return f"{value:.8g}"


def stream_stats(
    files: list[Path],
    sampling_fraction: float,
    chunk_showers: int,
    eps_values: list[float],
    noise_thresholds: list[float],
):
    summary = {eps: {"n": 0, "sum": 0.0, "sumsq": 0.0} for eps in eps_values}
    lt_counts = {thr: 0 for thr in noise_thresholds}
    pos_lt_counts = {thr: 0 for thr in noise_thresholds}
    total_showers = 0
    total_cells = 0
    zero_cells = 0
    min_positive = None

    for file_path in files:
        with h5py.File(file_path, "r") as handle:
            showers = handle["showers"]
            total_showers += showers.shape[0]
            for start in range(0, showers.shape[0], chunk_showers):
                end = min(start + chunk_showers, showers.shape[0])
                flat = showers[start:end].astype(np.float32).reshape(-1)
                flat = flat / 1000.0 * sampling_fraction

                total_cells += flat.size
                zero_cells += int((flat == 0).sum())

                positive = flat[flat > 0]
                if positive.size:
                    local_min = float(positive.min())
                    min_positive = local_min if min_positive is None else min(min_positive, local_min)

                for threshold in noise_thresholds:
                    lt_counts[threshold] += int((flat < threshold).sum())
                    pos_lt_counts[threshold] += int(((flat > 0) & (flat < threshold)).sum())

                for eps in eps_values:
                    logx = np.log(flat + eps, dtype=np.float64)
                    current = summary[eps]
                    current["n"] += logx.size
                    current["sum"] += float(logx.sum(dtype=np.float64))
                    current["sumsq"] += float((logx * logx).sum(dtype=np.float64))

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
):
    parts = []
    window = 16
    for file_path in files:
        with h5py.File(file_path, "r") as handle:
            showers = handle["showers"]
            n_showers = showers.shape[0]
            starts = [0, n_showers // 8, n_showers // 4, 3 * n_showers // 8, n_showers // 2, 5 * n_showers // 8, 3 * n_showers // 4, 7 * n_showers // 8, n_showers - window]
            for start in starts:
                start = max(0, min(start, n_showers - window))
                part = showers[start:start + window].astype(np.float32).reshape(-1)
                part = part / 1000.0 * sampling_fraction
                parts.append(part)

    flat = np.concatenate(parts)
    positive = flat[flat > 0]
    return np.quantile(positive, quantiles), len(parts) * window


def emit_yaml(noise_level: float, eps: float, mean: float, std: float):
    print("preprocessing:")
    print("  steps:")
    print("    - class_name: src.data.preprocessing.CutNoise")
    print("      init_args:")
    print(f"        noise_level: {noise_level:.8g}")
    print("        both_directions: true")
    print("    - class_name: src.data.preprocessing.LogTransform")
    print("      init_args:")
    print(f"        eps: {eps:.8g}")
    print("    - class_name: src.data.preprocessing.Standarize")
    print("      init_args:")
    print(f"        mean: {mean:.7f}")
    print(f"        std: {std:.7f}")
    print("keep_condition_components: [energy]")


def main():
    args = parse_args()
    files = [file_path.resolve() for file_path in args.files]

    print("input_files:")
    for file_path in files:
        print(f"  - {file_path}")

    stats = stream_stats(
        files=files,
        sampling_fraction=args.sampling_fraction,
        chunk_showers=args.chunk_showers,
        eps_values=args.eps,
        noise_thresholds=args.noise_thresholds,
    )

    quantiles, sample_showers = stratified_positive_quantiles(
        files=files,
        sampling_fraction=args.sampling_fraction,
        quantiles=args.show_quantiles,
    )

    total_cells = stats["total_cells"]
    zero_cells = stats["zero_cells"]
    print(f"total_showers={stats['total_showers']}")
    print(f"total_cells={total_cells}")
    print(f"zero_cells={zero_cells} ({zero_cells / total_cells * 100:.6f}%)")
    print(f"min_positive={stats['min_positive']:.12e}")

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
    for eps in args.eps:
        current = stats["summary"][eps]
        mean = current["sum"] / current["n"]
        variance = current["sumsq"] / current["n"] - mean * mean
        std = math.sqrt(max(variance, 0.0))
        print(f"  eps={eps:.8g} mean={mean:.7f} std={std:.7f}")

    print("\n# Candidate preprocessing snippets")
    for eps in args.eps:
        current = stats["summary"][eps]
        mean = current["sum"] / current["n"]
        variance = current["sumsq"] / current["n"] - mean * mean
        std = math.sqrt(max(variance, 0.0))
        print(f"\n# eps={eps:.8g}")
        emit_yaml(
            noise_level=args.suggested_noise_level,
            eps=eps,
            mean=mean,
            std=std,
        )


if __name__ == "__main__":
    main()
