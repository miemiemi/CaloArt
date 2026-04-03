#!/usr/bin/env python3
"""Inspect numerical ranges and sampled distributions of a CaloChallenge HDF5 file."""

from __future__ import annotations

import argparse
import json
from heapq import heappush, heappushpop
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 shower values without loading the full file into memory."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Input HDF5 file to inspect.")
    parser.add_argument(
        "--chunk-showers",
        type=int,
        default=512,
        help="Number of showers to load per chunk.",
    )
    parser.add_argument(
        "--per-chunk-sample",
        type=int,
        default=10000,
        help="Number of cells to sample per chunk for approximate cell quantiles.",
    )
    parser.add_argument(
        "--max-all-sample",
        type=int,
        default=2_000_000,
        help="Maximum total sampled cells retained for all-cell quantiles.",
    )
    parser.add_argument(
        "--max-nonzero-sample",
        type=int,
        default=2_000_000,
        help="Maximum total sampled positive cells retained for nonzero-cell quantiles.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top events/cells to report.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0],
        help="Value thresholds for shower cell counts.",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 100.0],
        help="Percentiles to report. Values are in [0, 100].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampled cell quantiles.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the structured summary as JSON.",
    )
    return parser.parse_args()


def percentile_summary(arr: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    values = np.percentile(arr, quantiles)
    result = {f"p{q:g}": float(v) for q, v in zip(quantiles, values)}
    result["mean"] = float(arr.mean())
    return result


def push_top(heap: list[tuple[float, int, int]], value: float, idx0: int, idx1: int, keep_top: int):
    item = (value, idx0, idx1)
    if len(heap) < keep_top:
        heappush(heap, item)
    elif value > heap[0][0]:
        heappushpop(heap, item)


def maybe_downsample(rng: np.random.Generator, arr: np.ndarray, max_size: int) -> np.ndarray:
    if arr.size <= max_size:
        return arr
    idx = rng.choice(arr.size, size=max_size, replace=False)
    return arr[idx]


def detect_energy_key(handle: h5py.File) -> str:
    if "incident_energies" in handle:
        return "incident_energies"
    if "incident_energy" in handle:
        return "incident_energy"
    raise KeyError("Could not find `incident_energies` or `incident_energy` in the input file.")


def inspect_file(args):
    rng = np.random.default_rng(args.seed)
    thresholds = list(args.thresholds)
    quantiles = list(args.quantiles)

    all_sample_parts = []
    nonzero_sample_parts = []
    threshold_counts = {threshold: 0 for threshold in thresholds}
    top_cells = []
    top_events = []

    total_nan = 0
    total_inf = 0
    nonzero_total = 0
    cell_total = 0
    global_min = np.inf
    global_max = -np.inf
    finite_min = np.inf
    finite_max = -np.inf

    with h5py.File(args.input_file, "r") as handle:
        showers = handle["showers"]
        energy_key = detect_energy_key(handle)
        energies = handle[energy_key][:].reshape(-1)
        showers_dtype = str(showers.dtype)
        energies_dtype = str(energies.dtype)
        n_events, n_voxels = showers.shape

        event_sums = np.empty(n_events, dtype=np.float64)
        event_max = np.empty(n_events, dtype=np.float32)
        event_nonzero = np.empty(n_events, dtype=np.int32)

        for start in range(0, n_events, args.chunk_showers):
            stop = min(start + args.chunk_showers, n_events)
            arr = showers[start:stop]
            cell_total += arr.size

            global_min = min(global_min, float(arr.min()))
            global_max = max(global_max, float(arr.max()))
            total_nan += int(np.isnan(arr).sum())
            total_inf += int(np.isinf(arr).sum())

            finite_mask = np.isfinite(arr)
            if finite_mask.any():
                finite_vals = arr[finite_mask]
                finite_min = min(finite_min, float(finite_vals.min()))
                finite_max = max(finite_max, float(finite_vals.max()))

            nonzero_mask = arr > 0
            nonzero_total += int(nonzero_mask.sum())
            event_sums[start:stop] = arr.sum(axis=1, dtype=np.float64)
            event_max[start:stop] = arr.max(axis=1)
            event_nonzero[start:stop] = nonzero_mask.sum(axis=1)

            flat = arr.reshape(-1)
            positive_flat = flat[flat > 0]
            for threshold in thresholds:
                threshold_counts[threshold] += int((flat > threshold).sum())

            sample_size = min(flat.size, args.per_chunk_sample)
            if sample_size > 0:
                idx = rng.choice(flat.size, size=sample_size, replace=False)
                all_sample_parts.append(flat[idx].astype(np.float32, copy=False))

            if positive_flat.size:
                nz_sample_size = min(positive_flat.size, args.per_chunk_sample)
                idx_nz = rng.choice(positive_flat.size, size=nz_sample_size, replace=False)
                nonzero_sample_parts.append(positive_flat[idx_nz].astype(np.float32, copy=False))

            top_count = min(args.top_k, flat.size)
            if top_count > 0:
                candidate_idx = np.argpartition(flat, -top_count)[-top_count:]
                for flat_idx in candidate_idx:
                    value = float(flat[flat_idx])
                    event_idx, cell_idx = divmod(int(flat_idx), n_voxels)
                    push_top(top_cells, value, start + event_idx, cell_idx, args.top_k)

        for event_idx, value in enumerate(event_sums):
            push_top(top_events, float(value), event_idx, -1, args.top_k)

    all_sample = np.concatenate(all_sample_parts) if all_sample_parts else np.array([], dtype=np.float32)
    nonzero_sample = (
        np.concatenate(nonzero_sample_parts) if nonzero_sample_parts else np.array([], dtype=np.float32)
    )
    all_sample = maybe_downsample(rng, all_sample, args.max_all_sample)
    nonzero_sample = maybe_downsample(rng, nonzero_sample, args.max_nonzero_sample)

    result = {
        "path": str(args.input_file.resolve()),
        "shape": [int(n_events), int(n_voxels)],
        "showers_dtype": showers_dtype,
        "energy_key": energy_key,
        "energies_shape": [int(energies.shape[0])],
        "energies_dtype": energies_dtype,
        "showers_nan": int(total_nan),
        "showers_inf": int(total_inf),
        "showers_min": float(global_min),
        "showers_max": float(global_max),
        "showers_finite_min": None if finite_min == np.inf else float(finite_min),
        "showers_finite_max": None if finite_max == -np.inf else float(finite_max),
        "nonzero_fraction": float(nonzero_total / cell_total),
        "nonzero_cells_total": int(nonzero_total),
        "cell_total": int(cell_total),
        "event_sum": percentile_summary(event_sums, quantiles),
        "event_max_cell": percentile_summary(event_max, quantiles),
        "event_nonzero_cells": percentile_summary(event_nonzero, quantiles),
        "incident_energy": percentile_summary(energies, quantiles),
        "showers_all_sample": percentile_summary(all_sample, quantiles),
        "showers_nonzero_sample": percentile_summary(nonzero_sample, quantiles),
        "threshold_counts": {
            str(threshold): {
                "count": int(count),
                "fraction": float(count / cell_total),
            }
            for threshold, count in threshold_counts.items()
        },
        "top_events_by_sum": [
            {
                "event": int(event_idx),
                "event_sum": float(event_sum),
                "event_max_cell": float(event_max[event_idx]),
                "event_nonzero_cells": int(event_nonzero[event_idx]),
                "incident_energy": float(energies[event_idx]),
            }
            for event_sum, event_idx, _ in sorted(top_events, reverse=True)
        ],
        "top_cells": [
            {
                "event": int(event_idx),
                "cell": int(cell_idx),
                "value": float(value),
                "event_sum": float(event_sums[event_idx]),
                "incident_energy": float(energies[event_idx]),
            }
            for value, event_idx, cell_idx in sorted(top_cells, reverse=True)
        ],
    }
    return result


def emit_text(summary: dict):
    print(f"path: {summary['path']}")
    print(
        "shape: "
        f"events={summary['shape'][0]} voxels={summary['shape'][1]} "
        f"showers_dtype={summary['showers_dtype']}"
    )
    print(
        "energies: "
        f"key={summary['energy_key']} shape={tuple(summary['energies_shape'])} "
        f"dtype={summary['energies_dtype']}"
    )
    print(
        "showers: "
        f"nan={summary['showers_nan']} inf={summary['showers_inf']} "
        f"min={summary['showers_min']:.8g} max={summary['showers_max']:.8g}"
    )
    print(
        "finite_showers: "
        f"min={summary['showers_finite_min']:.8g} max={summary['showers_finite_max']:.8g}"
    )
    print(
        "density: "
        f"nonzero_fraction={summary['nonzero_fraction']:.8%} "
        f"nonzero_cells_total={summary['nonzero_cells_total']}"
    )
    print()

    for key in [
        "event_sum",
        "event_max_cell",
        "event_nonzero_cells",
        "incident_energy",
        "showers_all_sample",
        "showers_nonzero_sample",
    ]:
        print(key)
        for quantile_key, value in summary[key].items():
            print(f"  {quantile_key}: {value:.8g}")
        print()

    print("threshold_counts")
    for threshold, payload in summary["threshold_counts"].items():
        print(
            f"  > {threshold}: {payload['count']} "
            f"({payload['fraction']:.8%})"
        )
    print()

    print("top_events_by_sum")
    for item in summary["top_events_by_sum"]:
        print(
            "  "
            f"event={item['event']} sum={item['event_sum']:.8g} "
            f"max_cell={item['event_max_cell']:.8g} "
            f"nonzero_cells={item['event_nonzero_cells']} "
            f"energy={item['incident_energy']:.8g}"
        )
    print()

    print("top_cells")
    for item in summary["top_cells"]:
        print(
            "  "
            f"event={item['event']} cell={item['cell']} value={item['value']:.8g} "
            f"event_sum={item['event_sum']:.8g} energy={item['incident_energy']:.8g}"
        )


def main():
    args = parse_args()
    summary = inspect_file(args)
    emit_text(summary)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
        print()
        print(f"json_written: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
