#!/usr/bin/env python3
"""Compute FPD/KPD for a generated HDF5 file using the local project pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from src.evaluation.fpd_kpd import compute_fpd_kpd, get_evaluator_threshold_from_internal_noise


def load_h5(path: Path, geometry: str, num_showers: int | None):
    is_ccd = geometry.startswith("CCD")
    energy_key = "incident_energies" if is_ccd else "incident_energy"

    with h5py.File(path, "r") as handle:
        showers = handle["showers"][:]
        incident_energy = handle[energy_key][:]

    if num_showers is not None:
        showers = showers[:num_showers]
        incident_energy = incident_energy[:num_showers]

    return showers, incident_energy.reshape(-1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute project-local FPD/KPD from raw HDF5 files.")
    parser.add_argument("--generated-file", type=Path, required=True, help="Generated HDF5 file.")
    parser.add_argument("--reference-file", type=Path, required=True, help="Reference HDF5 file.")
    parser.add_argument("--geometry", default="CCD2", help="Geometry name, e.g. CCD2.")
    parser.add_argument(
        "--num-showers",
        type=int,
        default=None,
        help="Optional cap on the number of showers loaded from each file.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.5e-6,
        help="Internal GeV noise threshold used by the project preprocessing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store the FPD/KPD text file. Defaults next to the generated file.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset tag for the output filename.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    generated_showers, generated_energy = load_h5(
        args.generated_file,
        args.geometry,
        args.num_showers,
    )
    reference_showers, reference_energy = load_h5(
        args.reference_file,
        args.geometry,
        args.num_showers,
    )

    threshold = get_evaluator_threshold_from_internal_noise(
        args.noise_level,
        geometry=args.geometry,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.generated_file.parent / "fpd_local_eval"

    dataset_name = args.dataset_name or args.generated_file.stem

    results = compute_fpd_kpd(
        generated_showers=generated_showers,
        reference_showers=reference_showers,
        generated_energy=generated_energy,
        reference_energy=reference_energy,
        geometry=args.geometry,
        cut=threshold,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )

    print(f"generated_file={args.generated_file}")
    print(f"reference_file={args.reference_file}")
    print(f"geometry={args.geometry}")
    print(f"num_showers={len(generated_showers)}")
    print(f"threshold={threshold:.8e}")
    print(f"output_dir={output_dir}")
    print(
        "FPD (x10^3): "
        f"{results['fpd_val'] * 1e3:.4f} ± {results['fpd_err'] * 1e3:.4f}"
    )
    print(
        "KPD (x10^3): "
        f"{results['kpd_val'] * 1e3:.4f} ± {results['kpd_err'] * 1e3:.4f}"
    )


if __name__ == "__main__":
    sys.exit(main())
