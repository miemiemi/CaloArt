#!/usr/bin/env python3
"""Compute exact logit-preprocessing statistics for CCD HDF5 showers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute exact pre/post-standardization logit stats for CCD datasets."
    )
    parser.add_argument("--files", type=Path, nargs="+", required=True, help="Input HDF5 files.")
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
        "--raw-cut-noise-level",
        type=float,
        default=5.0e-7,
        help="Cut threshold in internal GeV used only for reporting/inverse compatibility checks.",
    )
    parser.add_argument(
        "--remove-sampling-fraction-factor",
        type=float,
        default=1.0,
        help="Optional multiplicative factor applied before energy scaling, e.g. 1/0.033.",
    )
    parser.add_argument(
        "--scale-by-incident-energy",
        action="store_true",
        help="Apply ScaleByIncidentEnergy before the logit transform.",
    )
    parser.add_argument(
        "--scale-by-factor",
        type=float,
        default=1.0,
        help="Apply ScaleByFactor before the logit transform.",
    )
    parser.add_argument(
        "--logit-eps",
        type=float,
        default=1.0e-6,
        help="Epsilon used in LogitTransform.",
    )
    parser.add_argument(
        "--standardize-mean",
        type=float,
        default=None,
        help="Optional mean used to summarize post-standardization stats.",
    )
    parser.add_argument(
        "--standardize-std",
        type=float,
        default=None,
        help="Optional std used to summarize post-standardization stats.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _new_acc():
    return {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None}


def _update_acc(acc: dict, values: np.ndarray):
    flat = values.reshape(-1).astype(np.float64, copy=False)
    acc["count"] += int(flat.size)
    acc["sum"] += float(flat.sum(dtype=np.float64))
    acc["sumsq"] += float((flat * flat).sum(dtype=np.float64))
    local_min = float(flat.min())
    local_max = float(flat.max())
    acc["min"] = local_min if acc["min"] is None else min(acc["min"], local_min)
    acc["max"] = local_max if acc["max"] is None else max(acc["max"], local_max)


def _finalize_acc(acc: dict):
    if acc["count"] == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = acc["sum"] / acc["count"]
    variance = max(acc["sumsq"] / acc["count"] - mean * mean, 0.0)
    return {
        "count": int(acc["count"]),
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "min": float(acc["min"]),
        "max": float(acc["max"]),
    }


def main():
    args = parse_args()

    logit_acc = _new_acc()
    standardized_acc = _new_acc() if args.standardize_mean is not None and args.standardize_std is not None else None

    total_showers = 0
    total_cells = 0
    zero_cells = 0
    positive_cells = 0
    positive_cells_below_or_eq_cut = 0
    total_energy = 0.0
    energy_below_or_eq_cut = 0.0
    event_ratio_sum = 0.0
    event_ratio_sumsq = 0.0
    event_ratio_count = 0

    for path in args.files:
        with h5py.File(path, "r") as handle:
            showers = handle["showers"]
            energies = handle["incident_energies"]
            total_showers += int(showers.shape[0])

            for start in range(0, showers.shape[0], args.chunk_showers):
                end = min(start + args.chunk_showers, showers.shape[0])
                x = showers[start:end].astype(np.float64)
                e = energies[start:end].astype(np.float64).reshape(-1, 1)

                # Match CaloFlow internal CCD convention first.
                x = x / 1000.0 * args.sampling_fraction
                e = e / 1000.0

                total_cells += int(x.size)
                zero_cells += int((x == 0.0).sum())
                positive_mask = x > 0.0
                positive_cells += int(positive_mask.sum())
                below_cut_mask = positive_mask & (x <= args.raw_cut_noise_level)
                positive_cells_below_or_eq_cut += int(below_cut_mask.sum())
                total_energy += float(x.sum(dtype=np.float64))
                energy_below_or_eq_cut += float(x[below_cut_mask].sum(dtype=np.float64))

                work = x
                if args.remove_sampling_fraction_factor != 1.0:
                    work = work * args.remove_sampling_fraction_factor
                if args.scale_by_incident_energy:
                    work = work / e.reshape(-1, 1)
                if args.scale_by_factor != 1.0:
                    work = work / args.scale_by_factor

                z = args.logit_eps + (1.0 - 2.0 * args.logit_eps) * work
                logit = np.log(z / (1.0 - z), dtype=np.float64)
                _update_acc(logit_acc, logit)

                if standardized_acc is not None:
                    standardized = (logit - args.standardize_mean) / args.standardize_std
                    _update_acc(standardized_acc, standardized)

                event_ratio = x.reshape(x.shape[0], -1).sum(axis=1) / e.reshape(-1)
                event_ratio_sum += float(event_ratio.sum(dtype=np.float64))
                event_ratio_sumsq += float((event_ratio * event_ratio).sum(dtype=np.float64))
                event_ratio_count += int(event_ratio.size)

    result = {
        "files": [str(path.resolve()) for path in args.files],
        "sampling_fraction": float(args.sampling_fraction),
        "chunk_showers": int(args.chunk_showers),
        "raw_cut_noise_level": float(args.raw_cut_noise_level),
        "remove_sampling_fraction_factor": float(args.remove_sampling_fraction_factor),
        "scale_by_incident_energy": bool(args.scale_by_incident_energy),
        "scale_by_factor": float(args.scale_by_factor),
        "logit_eps": float(args.logit_eps),
        "standardize_mean": None if args.standardize_mean is None else float(args.standardize_mean),
        "standardize_std": None if args.standardize_std is None else float(args.standardize_std),
        "total_showers": int(total_showers),
        "total_cells": int(total_cells),
        "zero_cells": int(zero_cells),
        "zero_fraction": float(zero_cells / total_cells) if total_cells else 0.0,
        "positive_voxel_fraction_below_or_eq_cut": (
            float(positive_cells_below_or_eq_cut / positive_cells) if positive_cells else 0.0
        ),
        "energy_fraction_below_or_eq_cut": float(energy_below_or_eq_cut / total_energy) if total_energy else 0.0,
        "pre_standardization_logit": _finalize_acc(logit_acc),
    }

    if event_ratio_count:
        ratio_mean = event_ratio_sum / event_ratio_count
        ratio_var = max(event_ratio_sumsq / event_ratio_count - ratio_mean * ratio_mean, 0.0)
        result["event_energy_ratio_internal"] = {
            "count": int(event_ratio_count),
            "mean": float(ratio_mean),
            "std": float(math.sqrt(ratio_var)),
        }

    if standardized_acc is not None:
        result["post_standardization"] = _finalize_acc(standardized_acc)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
        print(f"wrote_json={args.output_json.resolve()}")


if __name__ == "__main__":
    main()
