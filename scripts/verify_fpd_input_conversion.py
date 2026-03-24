#!/usr/bin/env python3
"""Verify that the FPD input conversion matches the CCD raw evaluator convention.

This script compares the same shower sample through two paths:

1. The current `trainer.test()` FPD preparation path:
   dataset load -> internal CCD representation -> internal noise cut ->
   `prepare_fpd_inputs()`
2. The reference evaluator path:
   raw HDF5 shower/incident energy -> evaluator-space threshold cut

If the conversion is correct, both paths should agree up to floating point
rounding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import CaloShowerDataset
from src.data.preprocessing import cut_below_noise_level
from src.evaluation.fpd_kpd import (
    get_evaluator_threshold_from_internal_noise,
    prepare_fpd_inputs,
)


def load_reference_hdf5(path: Path, geometry: str, num_showers: int):
    is_ccd = geometry.startswith("CCD")
    with h5py.File(path, "r") as handle:
        showers = handle["showers"][:num_showers].astype(np.float32)
        energy_key = "incident_energies" if is_ccd else "incident_energy"
        incident_energy = handle[energy_key][:num_showers].astype(np.float32).reshape(-1, 1)
    return showers, incident_energy


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float):
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch {actual.shape} != {expected.shape}")
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = np.abs(actual - expected)
        raise AssertionError(
            f"{name}: max_abs_diff={diff.max():.8e}, "
            f"mean_abs_diff={diff.mean():.8e}, "
            f"nonzero_mismatches={(diff > (atol + rtol * np.abs(expected))).sum()}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Verify that the FPD evaluator conversion matches the raw HDF5 convention."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Path to the reference HDF5 file.")
    parser.add_argument("--geometry", default="CCD2", help="Geometry name, e.g. CCD2.")
    parser.add_argument("--num-showers", type=int, default=128, help="Number of showers to compare.")
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.5e-6,
        help="Internal shower noise threshold in GeV, matching the preprocessing config.",
    )
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for the comparison.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for the comparison.")
    parser.add_argument("--verbose", action="store_true", help="Print extra diagnostics.")
    args = parser.parse_args()

    is_ccd = args.geometry.startswith("CCD")
    dataset = CaloShowerDataset(
        files=[str(args.input_file)],
        need_geo_condn=False,
        train_on=None,
        is_ccd=is_ccd,
        max_num_showers=args.num_showers,
    )

    internal_showers = cut_below_noise_level(
        torch.from_numpy(dataset.showers.copy()), noise_level=args.noise_level
    ).cpu().numpy()
    internal_energy = dataset.energy.reshape(-1, 1).copy()

    test_showers, test_energy = prepare_fpd_inputs(
        internal_showers,
        internal_energy,
        geometry=args.geometry,
    )

    raw_showers, raw_energy = load_reference_hdf5(args.input_file, args.geometry, args.num_showers)
    evaluator_threshold = get_evaluator_threshold_from_internal_noise(
        args.noise_level,
        geometry=args.geometry,
    )
    expected_showers = np.where(raw_showers < evaluator_threshold, 0.0, raw_showers)

    assert_close("showers", test_showers, expected_showers, atol=args.atol, rtol=args.rtol)
    assert_close("incident_energy", test_energy, raw_energy, atol=args.atol, rtol=args.rtol)

    print("[OK] FPD conversion matches the raw evaluator convention.")
    print(f"geometry={args.geometry}")
    print(f"num_showers={len(test_showers)}")
    print(f"internal_noise_level={args.noise_level:.8e} GeV")
    print(f"evaluator_threshold={evaluator_threshold:.8e}")
    print(f"showers_shape={test_showers.shape}")
    print(f"energies_shape={test_energy.shape}")

    if args.verbose:
        print(f"showers_max={test_showers.max():.8e}")
        print(f"showers_min={test_showers.min():.8e}")
        print(f"energy_max={test_energy.max():.8e}")
        print(f"energy_min={test_energy.min():.8e}")


if __name__ == "__main__":
    main()
