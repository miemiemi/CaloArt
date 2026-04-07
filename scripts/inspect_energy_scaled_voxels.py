#!/usr/bin/env python3
"""Inspect voxel / incident-energy distributions for CaloChallenge HDF5 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the distribution of x / E_inc on CaloChallenge HDF5 files "
            "using the same internal scaling as the current CaloFlow CCD pipeline."
        )
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        required=True,
        help="Input HDF5 files to treat as one combined dataset.",
    )
    parser.add_argument(
        "--sampling-fraction",
        type=float,
        default=0.033,
        help="Sampling fraction applied to CCD showers inside the current dataset loader.",
    )
    parser.add_argument(
        "--chunk-showers",
        type=int,
        default=256,
        help="Number of showers to load per chunk.",
    )
    parser.add_argument(
        "--per-chunk-sample",
        type=int,
        default=20000,
        help="Number of scaled voxels to sample per chunk for approximate quantiles.",
    )
    parser.add_argument(
        "--max-positive-sample",
        type=int,
        default=3000000,
        help="Maximum retained sample size for positive x/E_inc quantiles.",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 99.99, 100.0],
        help="Percentiles to report. Values are in [0, 100].",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1],
        help="Thresholds for counting positive x/E_inc voxels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampled quantiles.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON.",
    )
    parser.add_argument(
        "--save-scaled-h5",
        type=Path,
        default=None,
        help="Optional path to save the full chunkwise x/E_inc result as an HDF5 file.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        help="Compression codec for the optional saved HDF5 file.",
    )
    parser.add_argument(
        "--compression-opts",
        type=int,
        default=4,
        help="Compression level for the optional saved HDF5 file.",
    )
    return parser.parse_args()


def detect_energy_key(handle: h5py.File) -> str:
    if "incident_energies" in handle:
        return "incident_energies"
    if "incident_energy" in handle:
        return "incident_energy"
    raise KeyError("Could not find `incident_energies` or `incident_energy` in input file.")


def percentile_summary(arr: np.ndarray, quantiles: list[float]) -> dict[str, float] | None:
    if arr.size == 0:
        return None
    values = np.percentile(arr, quantiles)
    result = {f"p{q:g}": float(v) for q, v in zip(quantiles, values)}
    result["mean"] = float(arr.mean())
    return result


def maybe_downsample(rng: np.random.Generator, arr: np.ndarray, max_size: int) -> np.ndarray:
    if arr.size <= max_size:
        return arr
    idx = rng.choice(arr.size, size=max_size, replace=False)
    return arr[idx]


def inspect_dataset(args) -> dict:
    rng = np.random.default_rng(args.seed)
    thresholds = list(args.thresholds)
    quantiles = list(args.quantiles)

    total_events = 0
    total_cells = 0
    positive_cells = 0
    finite_cells = 0
    nan_count = 0
    inf_count = 0

    global_min = np.inf
    global_max = -np.inf
    finite_min = np.inf
    finite_max = -np.inf
    event_positive_fraction_parts = []
    event_scaled_sum_parts = []
    positive_threshold_counts = {threshold: 0 for threshold in thresholds}
    positive_samples = []
    files_summary = []
    writer = None
    scaled_showers_out = None
    energy_out = None
    output_offset = 0

    try:
        for file_path in args.files:
            file_path = file_path.resolve()
            with h5py.File(file_path, "r") as handle:
                showers = handle["showers"]
                energy_key = detect_energy_key(handle)
                energies = handle[energy_key]
                if showers.shape[0] != energies.shape[0]:
                    raise ValueError(
                        f"Shape mismatch for {file_path}: showers={showers.shape}, energies={energies.shape}"
                    )

                files_summary.append(
                    {
                        "path": str(file_path),
                        "showers_shape": [int(dim) for dim in showers.shape],
                        "energy_key": energy_key,
                        "energy_shape": [int(dim) for dim in energies.shape],
                        "showers_dtype": str(showers.dtype),
                        "energies_dtype": str(energies.dtype),
                    }
                )

                total_events += showers.shape[0]
                n_voxels = showers.shape[1]

                if args.save_scaled_h5 is not None and writer is None:
                    args.save_scaled_h5.parent.mkdir(parents=True, exist_ok=True)
                    writer = h5py.File(args.save_scaled_h5, "w")
                    scaled_showers_out = writer.create_dataset(
                        "showers",
                        shape=(0, n_voxels),
                        maxshape=(None, n_voxels),
                        dtype=np.float32,
                        chunks=(min(args.chunk_showers, 1024), n_voxels),
                        compression=args.compression,
                        compression_opts=args.compression_opts,
                    )
                    energy_out = writer.create_dataset(
                        "incident_energies",
                        shape=(0, 1),
                        maxshape=(None, 1),
                        dtype=np.float32,
                        chunks=(min(args.chunk_showers, 1024), 1),
                        compression=args.compression,
                        compression_opts=args.compression_opts,
                    )
                    writer.attrs["transform"] = "showers_internal / incident_energy_internal"
                    writer.attrs["showers_internal_definition"] = "raw_showers_MeV / 1000 * sampling_fraction"
                    writer.attrs["incident_energy_internal_definition"] = "incident_energies_MeV / 1000"
                    writer.attrs["sampling_fraction"] = float(args.sampling_fraction)

                for start in range(0, showers.shape[0], args.chunk_showers):
                    stop = min(start + args.chunk_showers, showers.shape[0])
                    shower_chunk = showers[start:stop].astype(np.float32, copy=False)
                    raw_energy_chunk = energies[start:stop].astype(np.float32, copy=False).reshape(-1, 1)

                    # Match the current CCD internal training representation:
                    # raw MeV -> GeV, then multiply by the sampling fraction.
                    shower_chunk = shower_chunk / 1000.0 * args.sampling_fraction
                    energy_chunk = raw_energy_chunk / 1000.0
                    scaled = shower_chunk / energy_chunk

                    if writer is not None:
                        next_offset = output_offset + scaled.shape[0]
                        scaled_showers_out.resize((next_offset, n_voxels))
                        energy_out.resize((next_offset, 1))
                        scaled_showers_out[output_offset:next_offset] = scaled.astype(np.float32, copy=False)
                        energy_out[output_offset:next_offset] = raw_energy_chunk
                        output_offset = next_offset

                    total_cells += scaled.size
                    finite_mask = np.isfinite(scaled)
                    finite_cells += int(finite_mask.sum())
                    nan_count += int(np.isnan(scaled).sum())
                    inf_count += int(np.isinf(scaled).sum())

                    global_min = min(global_min, float(np.min(scaled)))
                    global_max = max(global_max, float(np.max(scaled)))

                    if finite_mask.any():
                        finite_values = scaled[finite_mask]
                        finite_min = min(finite_min, float(finite_values.min()))
                        finite_max = max(finite_max, float(finite_values.max()))

                    positive_mask = scaled > 0
                    positive_cells += int(positive_mask.sum())
                    event_positive_fraction_parts.append(
                        positive_mask.sum(axis=1, dtype=np.int64).astype(np.float64) / n_voxels
                    )
                    event_scaled_sum_parts.append(scaled.sum(axis=1, dtype=np.float64))

                    positive_values = scaled[positive_mask]
                    if positive_values.size:
                        for threshold in thresholds:
                            positive_threshold_counts[threshold] += int((positive_values > threshold).sum())

                        sample_size = min(positive_values.size, args.per_chunk_sample)
                        sample_idx = rng.choice(positive_values.size, size=sample_size, replace=False)
                        positive_samples.append(positive_values[sample_idx].astype(np.float32, copy=False))
    finally:
        if writer is not None:
            writer.attrs["num_events"] = int(output_offset)
            writer.close()

    positive_sample = (
        np.concatenate(positive_samples) if positive_samples else np.array([], dtype=np.float32)
    )
    positive_sample = maybe_downsample(rng, positive_sample, args.max_positive_sample)
    event_positive_fraction = (
        np.concatenate(event_positive_fraction_parts)
        if event_positive_fraction_parts
        else np.array([], dtype=np.float64)
    )
    event_scaled_sum = (
        np.concatenate(event_scaled_sum_parts)
        if event_scaled_sum_parts
        else np.array([], dtype=np.float64)
    )

    result = {
        "files": files_summary,
        "sampling_fraction": float(args.sampling_fraction),
        "total_events": int(total_events),
        "total_cells": int(total_cells),
        "positive_cells": int(positive_cells),
        "positive_fraction": float(positive_cells / total_cells) if total_cells else None,
        "finite_cells": int(finite_cells),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "scaled_min": float(global_min),
        "scaled_max": float(global_max),
        "scaled_finite_min": None if finite_min == np.inf else float(finite_min),
        "scaled_finite_max": None if finite_max == -np.inf else float(finite_max),
        "positive_scaled_sample": percentile_summary(positive_sample, quantiles),
        "event_positive_fraction": percentile_summary(event_positive_fraction, quantiles),
        "event_scaled_sum": percentile_summary(event_scaled_sum, quantiles),
        "positive_threshold_counts": {
            str(threshold): {
                "count": int(count),
                "fraction_of_positive": float(count / positive_cells) if positive_cells else None,
                "fraction_of_all": float(count / total_cells) if total_cells else None,
            }
            for threshold, count in positive_threshold_counts.items()
        },
        "interpretation_hint": (
            "If most positive x/E_inc voxels remain spread across many decades "
            "(for example p1 near 1e-9 and p99 near 1e-2 or above), log compression is still valuable."
        ),
        "saved_scaled_h5": str(args.save_scaled_h5.resolve()) if args.save_scaled_h5 is not None else None,
    }
    return result


def emit_text(summary: dict):
    print("files:")
    for file_info in summary["files"]:
        print(
            f"  - {file_info['path']} "
            f"showers_shape={tuple(file_info['showers_shape'])} "
            f"energy_key={file_info['energy_key']}"
        )
    print(f"sampling_fraction={summary['sampling_fraction']}")
    print(f"total_events={summary['total_events']}")
    print(f"total_cells={summary['total_cells']}")
    print(f"positive_cells={summary['positive_cells']} ({summary['positive_fraction']:.6%})")
    print(f"nan_count={summary['nan_count']} inf_count={summary['inf_count']}")
    print(
        "scaled_range="
        f"[{summary['scaled_finite_min']:.6e}, {summary['scaled_finite_max']:.6e}]"
    )

    positive_sample = summary["positive_scaled_sample"]
    if positive_sample is not None:
        print("positive_x_over_einc_sample:")
        for key, value in positive_sample.items():
            print(f"  {key}={value:.6e}")

    event_positive_fraction = summary["event_positive_fraction"]
    if event_positive_fraction is not None:
        print("event_positive_fraction:")
        for key, value in event_positive_fraction.items():
            print(f"  {key}={value:.6e}")

    event_scaled_sum = summary["event_scaled_sum"]
    if event_scaled_sum is not None:
        print("event_sum_of_x_over_einc:")
        for key, value in event_scaled_sum.items():
            print(f"  {key}={value:.6e}")

    print("positive_threshold_counts:")
    for threshold, stats in summary["positive_threshold_counts"].items():
        print(
            f"  >{threshold}: count={stats['count']} "
            f"fraction_of_positive={stats['fraction_of_positive']:.6%} "
            f"fraction_of_all={stats['fraction_of_all']:.6%}"
        )

    print(f"hint: {summary['interpretation_hint']}")
    if summary["saved_scaled_h5"] is not None:
        print(f"saved_scaled_h5={summary['saved_scaled_h5']}")


def main():
    args = parse_args()
    summary = inspect_dataset(args)
    emit_text(summary)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"wrote_json={args.output_json.resolve()}")


if __name__ == "__main__":
    main()
