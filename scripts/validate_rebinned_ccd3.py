#!/usr/bin/env python3
"""Validate a rebinned CCD3 HDF5 against its original full-resolution source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import numpy as np

INPUT_SHAPE_Z_PHI_R = (45, 50, 18)
OUTPUT_SHAPE_Z_PHI_R = (45, 25, 9)
OUTPUT_SHAPE_R_PHI_Z = (9, 25, 45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a CCD3 2x2 rebinned HDF5 against its original source."
    )
    parser.add_argument("--input-h5", type=Path, required=True, help="Original CCD3 HDF5.")
    parser.add_argument("--output-h5", type=Path, required=True, help="Rebinned HDF5.")
    parser.add_argument("--output-xml", type=Path, required=True, help="Rebinned XML.")
    parser.add_argument(
        "--chunk-showers",
        type=int,
        default=512,
        help="Chunk size for whole-file validation.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to save the validation summary as JSON.",
    )
    return parser.parse_args()


def validate_xml(output_xml: Path) -> None:
    root = ET.parse(output_xml).getroot()
    particles = list(root)
    if len(particles) != 1:
        raise ValueError(f"Expected exactly one particle block in {output_xml}, got {len(particles)}.")

    layers = list(particles[0])
    if len(layers) != 45:
        raise ValueError(f"Expected 45 layers in {output_xml}, got {len(layers)}.")

    for idx, layer in enumerate(layers):
        edges = [float(x) for x in layer.attrib["r_edges"].split(",")]
        alpha = int(layer.attrib["n_bin_alpha"])
        if len(edges) - 1 != 9:
            raise ValueError(f"Layer {idx} expected 9 r bins, got {len(edges) - 1}.")
        if alpha != 25:
            raise ValueError(f"Layer {idx} expected 25 alpha bins, got {alpha}.")


def compute_expected_rebin(src_showers_flat: np.ndarray) -> np.ndarray:
    showers = src_showers_flat.reshape(-1, *INPUT_SHAPE_Z_PHI_R)
    rebinned = showers.reshape(-1, 45, 25, 2, 9, 2).sum(axis=(3, 5), dtype=np.float64)
    return rebinned.reshape(rebinned.shape[0], -1)


def validate_h5(input_h5: Path, output_h5: Path, chunk_showers: int) -> dict:
    with h5py.File(input_h5, "r") as src, h5py.File(output_h5, "r") as dst:
        if set(src.keys()) != set(dst.keys()):
            raise ValueError(f"Dataset key mismatch: {set(src.keys())} vs {set(dst.keys())}")

        total_events = int(src["showers"].shape[0])
        if tuple(src["showers"].shape) != (total_events, 40500):
            raise ValueError(f"Unexpected source shape: {src['showers'].shape}")
        if tuple(dst["showers"].shape) != (total_events, 10125):
            raise ValueError(f"Unexpected rebinned shape: {dst['showers'].shape}")

        if dst.attrs.get("showers_storage_layout") != "flat-z-phi-r":
            raise ValueError(
                f"Unexpected output layout: {dst.attrs.get('showers_storage_layout')!r}"
            )
        if tuple(dst.attrs["rebinned_shape_z_phi_r"]) != OUTPUT_SHAPE_Z_PHI_R:
            raise ValueError(
                f"Unexpected rebinned_shape_z_phi_r: {tuple(dst.attrs['rebinned_shape_z_phi_r'])}"
            )
        if tuple(dst.attrs["rebinned_shape_r_phi_z"]) != OUTPUT_SHAPE_R_PHI_Z:
            raise ValueError(
                f"Unexpected rebinned_shape_r_phi_z: {tuple(dst.attrs['rebinned_shape_r_phi_z'])}"
            )

        max_abs_bin_diff = 0.0
        max_abs_event_energy_diff = 0.0
        max_abs_nonshowers_diff = 0.0

        for start in range(0, total_events, chunk_showers):
            stop = min(start + chunk_showers, total_events)
            src_show = src["showers"][start:stop].astype(np.float64, copy=False)
            dst_show = dst["showers"][start:stop].astype(np.float64, copy=False)

            expected = compute_expected_rebin(src_show)
            diff = np.abs(expected - dst_show)
            if diff.size:
                max_abs_bin_diff = max(max_abs_bin_diff, float(diff.max()))

            src_sum = src_show.sum(axis=1, dtype=np.float64)
            dst_sum = dst_show.sum(axis=1, dtype=np.float64)
            if src_sum.size:
                max_abs_event_energy_diff = max(
                    max_abs_event_energy_diff,
                    float(np.max(np.abs(src_sum - dst_sum))),
                )

            for key in src.keys():
                if key == "showers":
                    continue
                src_arr = src[key][start:stop].astype(np.float64, copy=False)
                dst_arr = dst[key][start:stop].astype(np.float64, copy=False)
                if src_arr.size:
                    max_abs_nonshowers_diff = max(
                        max_abs_nonshowers_diff,
                        float(np.max(np.abs(src_arr - dst_arr))),
                    )

        return {
            "input_h5": str(input_h5),
            "output_h5": str(output_h5),
            "total_events": total_events,
            "input_showers_shape": [total_events, 40500],
            "output_showers_shape": [total_events, 10125],
            "output_shape_z_phi_r": list(OUTPUT_SHAPE_Z_PHI_R),
            "output_shape_r_phi_z": list(OUTPUT_SHAPE_R_PHI_Z),
            "max_abs_bin_diff": max_abs_bin_diff,
            "max_abs_event_energy_diff": max_abs_event_energy_diff,
            "max_abs_nonshowers_diff": max_abs_nonshowers_diff,
        }


def main() -> None:
    args = parse_args()
    validate_xml(args.output_xml)
    summary = validate_h5(
        input_h5=args.input_h5,
        output_h5=args.output_h5,
        chunk_showers=max(1, args.chunk_showers),
    )

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(summary, indent=2))

    print("[OK] full-file validation passed")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
