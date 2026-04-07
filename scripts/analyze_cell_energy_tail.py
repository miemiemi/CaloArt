#!/usr/bin/env python3
"""Quantify and visualize the high-energy cell tail for a generated HDF5 sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from src.data.geometry import get_geometry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the high-energy cell tail for CaloFlow/CaloChallenge HDF5 files."
    )
    parser.add_argument("--generated-file", type=Path, required=True, help="Generated HDF5 file.")
    parser.add_argument("--reference-file", type=Path, required=True, help="Reference Geant4 HDF5 file.")
    parser.add_argument("--geometry", default="CCD2", help="CCD geometry name, e.g. CCD2 or CCD3.")
    parser.add_argument(
        "--num-showers",
        type=int,
        default=1000,
        help="Number of showers to compare from each file.",
    )
    parser.add_argument(
        "--tail-thresholds",
        type=float,
        nargs="+",
        default=[80.0, 90.0, 100.0, 110.0],
        help="Tail thresholds in MeV.",
    )
    parser.add_argument(
        "--zoom-min-mev",
        type=float,
        default=70.0,
        help="Left edge of the zoomed linear-energy panel in MeV.",
    )
    parser.add_argument(
        "--zoom-log-min",
        type=float,
        default=1.8,
        help="Left edge of the zoomed log10(E/MeV) panel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the zoom plot and JSON summary. Defaults next to the generated file.",
    )
    parser.add_argument(
        "--output-prefix",
        default="CellEnergyTail",
        help="Prefix for generated artifact filenames.",
    )
    parser.add_argument(
        "--preprocess-mean",
        type=float,
        default=-15.7554139,
        help="Standardization mean used before sampling.",
    )
    parser.add_argument(
        "--preprocess-std",
        type=float,
        default=4.7630354,
        help="Standardization std used before sampling.",
    )
    return parser.parse_args()


def load_ccd_showers(path: Path, geometry_name: str, num_showers: int) -> np.ndarray:
    geometry = get_geometry(geometry_name)
    with h5py.File(path, "r") as handle:
        showers = handle["showers"][:num_showers].astype(np.float32)
    showers = (
        showers.reshape(
            showers.shape[0],
            geometry.N_CELLS_Z,
            geometry.N_CELLS_PHI,
            geometry.N_CELLS_R,
        )
        .transpose(0, 3, 2, 1)
        * 0.033
    )
    return showers


def summarize_tail(
    reference_nonzero: np.ndarray,
    generated_nonzero: np.ndarray,
    generated_raw_nonzero: np.ndarray,
    thresholds: list[float],
) -> dict:
    reference_max = float(reference_nonzero.max())
    generated_raw_max = float(generated_raw_nonzero.max())
    total_nonzero_reference = int(reference_nonzero.size)
    total_nonzero_generated = int(generated_nonzero.size)
    total_raw_generated = int(generated_raw_nonzero.size)

    threshold_summary = {}
    for threshold in thresholds:
        reference_count = int((reference_nonzero >= threshold).sum())
        generated_count = int((generated_nonzero >= threshold).sum())
        threshold_summary[str(threshold)] = {
            "reference_count": reference_count,
            "reference_fraction_nonzero": float(reference_count / total_nonzero_reference),
            "generated_count": generated_count,
            "generated_fraction_nonzero": float(generated_count / total_nonzero_generated),
            "generated_to_reference_ratio": float(generated_count / reference_count)
            if reference_count
            else None,
        }

    visible_tail_threshold = 100.0
    reference_visible_tail = int((reference_nonzero >= visible_tail_threshold).sum())
    generated_visible_tail = int((generated_nonzero >= visible_tail_threshold).sum())
    generated_raw_exceed = int((generated_raw_nonzero > reference_max).sum())

    return {
        "reference_nonzero_cells": total_nonzero_reference,
        "generated_nonzero_cells": total_nonzero_generated,
        "reference_max_mev": reference_max,
        "generated_raw_max_mev": generated_raw_max,
        "thresholds_mev": threshold_summary,
        "visible_tail_ge_100_mev": {
            "reference_count": reference_visible_tail,
            "reference_fraction_nonzero": float(reference_visible_tail / total_nonzero_reference),
            "generated_count": generated_visible_tail,
            "generated_fraction_nonzero": float(generated_visible_tail / total_nonzero_generated),
        },
        "raw_generated_above_reference_max": {
            "count": generated_raw_exceed,
            "fraction_nonzero_generated": float(generated_raw_exceed / total_raw_generated),
        },
    }


def collect_top_anomalies(
    generated_raw: np.ndarray,
    incident_energy_mev: np.ndarray,
    reference_max_mev: float,
    preprocess_mean: float,
    preprocess_std: float,
) -> list[dict]:
    coords = np.argwhere(generated_raw > reference_max_mev)
    anomalies = []
    for event_idx, r_idx, phi_idx, z_idx in coords:
        cell_energy_mev = float(generated_raw[event_idx, r_idx, phi_idx, z_idx])
        cell_energy_gev = cell_energy_mev / 1000.0
        log_space_value = float(np.log(cell_energy_gev + 1.0e-8))
        standardized_value = float((log_space_value - preprocess_mean) / preprocess_std)
        event_energy_sum_mev = float(generated_raw[event_idx].sum())
        anomalies.append(
            {
                "event_index": int(event_idx),
                "r": int(r_idx),
                "phi": int(phi_idx),
                "z": int(z_idx),
                "cell_energy_mev": cell_energy_mev,
                "cell_energy_gev": cell_energy_gev,
                "incident_energy_mev": float(incident_energy_mev[event_idx]),
                "incident_energy_gev": float(incident_energy_mev[event_idx] / 1000.0),
                "event_energy_sum_mev": event_energy_sum_mev,
                "cell_fraction_of_event_energy": float(cell_energy_mev / event_energy_sum_mev),
                "log_space_value": log_space_value,
                "standardized_value": standardized_value,
            }
        )
    anomalies.sort(key=lambda item: item["cell_energy_mev"], reverse=True)
    return anomalies


def make_zoom_plot(
    reference_nonzero: np.ndarray,
    generated_nonzero: np.ndarray,
    generated_raw_nonzero: np.ndarray,
    output_path: Path,
    zoom_min_mev: float,
    zoom_log_min: float,
):
    log_bins = np.linspace(-3, 4, 1000)
    energy_bins = np.linspace(1e-2, 4000, 1000)

    reference_log = np.log10(reference_nonzero)
    generated_log = np.log10(generated_nonzero)

    full_max = float(reference_nonzero.max())
    raw_exceed_count = int((generated_raw_nonzero > full_max).sum())
    raw_max = float(generated_raw_nonzero.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    axes[0].hist(
        reference_log,
        bins=log_bins,
        histtype="stepfilled",
        color="lightgray",
        edgecolor="dimgray",
        linewidth=1,
        label="Geant4",
    )
    axes[0].hist(
        generated_log,
        bins=log_bins,
        histtype="step",
        color="#1f77b4",
        linewidth=1.4,
        label="Generated (plot-matched clipping)",
    )
    axes[0].set_xlim(zoom_log_min, np.log10(full_max) + 0.02)
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=1)
    axes[0].set_xlabel("log10(E/MeV)")
    axes[0].set_ylabel("Number of cells")
    axes[0].set_title("Tail Zoom: log10(E/MeV)")
    axes[0].legend(loc="upper left")

    axes[1].hist(
        reference_nonzero,
        bins=energy_bins,
        histtype="stepfilled",
        color="lightgray",
        edgecolor="dimgray",
        linewidth=1,
        label="Geant4",
    )
    axes[1].hist(
        generated_nonzero,
        bins=energy_bins,
        histtype="step",
        color="#1f77b4",
        linewidth=1.4,
        label="Generated (plot-matched clipping)",
    )
    axes[1].set_xlim(zoom_min_mev, full_max + 2.0)
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=1)
    axes[1].set_xlabel("Energy [MeV]")
    axes[1].set_ylabel("Number of cells")
    axes[1].set_title("Tail Zoom: linear energy")
    axes[1].legend(loc="upper left")
    axes[1].text(
        0.98,
        0.97,
        (
            f"raw generated > Geant4 max: {raw_exceed_count}\n"
            f"Geant4 max: {full_max:.3f} MeV\n"
            f"raw generated max: {raw_max:.3f} MeV"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.geometry.startswith("CCD"):
        raise ValueError("This helper currently supports CCD geometries only.")

    output_dir = args.output_dir or args.generated_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_ccd_showers(args.reference_file, args.geometry, args.num_showers)
    generated_raw = load_ccd_showers(args.generated_file, args.geometry, args.num_showers)
    with h5py.File(args.generated_file, "r") as handle:
        incident_energy_mev = handle["incident_energies"][: args.num_showers].reshape(-1)

    reference_flat = reference.reshape(-1)
    generated_raw_flat = generated_raw.reshape(-1)

    reference_nonzero = reference_flat[reference_flat > 0]
    generated_raw_nonzero = generated_raw_flat[generated_raw_flat > 0]
    generated_nonzero = np.clip(generated_raw_nonzero, 0.0, reference_nonzero.max())

    summary = {
        "generated_file": str(args.generated_file.resolve()),
        "reference_file": str(args.reference_file.resolve()),
        "geometry": args.geometry,
        "num_showers": int(args.num_showers),
        "all_cells_total": int(reference_flat.size),
        "tail_summary": summarize_tail(
            reference_nonzero=reference_nonzero,
            generated_nonzero=generated_nonzero,
            generated_raw_nonzero=generated_raw_nonzero,
            thresholds=list(args.tail_thresholds),
        ),
        "raw_anomalies_above_reference_max": collect_top_anomalies(
            generated_raw=generated_raw,
            incident_energy_mev=incident_energy_mev,
            reference_max_mev=float(reference_nonzero.max()),
            preprocess_mean=args.preprocess_mean,
            preprocess_std=args.preprocess_std,
        ),
    }

    plot_path = output_dir / f"{args.output_prefix}_zoom.png"
    json_path = output_dir / f"{args.output_prefix}_summary.json"

    make_zoom_plot(
        reference_nonzero=reference_nonzero,
        generated_nonzero=generated_nonzero,
        generated_raw_nonzero=generated_raw_nonzero,
        output_path=plot_path,
        zoom_min_mev=args.zoom_min_mev,
        zoom_log_min=args.zoom_log_min,
    )

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Saved zoom plot to {plot_path}")
    print(f"Saved summary JSON to {json_path}")
    print(json.dumps(summary["tail_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
