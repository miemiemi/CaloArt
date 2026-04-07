#!/usr/bin/env python3
"""Rebuild observable plots for an existing CCD generated.h5 sample."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, pythonpath=True)

from src.data.geometry import get_geometry
from src.evaluation.utils import compare_observables
from src.utils import get_logger


logger = get_logger()

CONDITIONS_RE = re.compile(
    r"Geo_(?P<geometry>[^_]+)_E_(?P<energy>[^_]+)_Phi_(?P<phi>[^_]+)_Theta_(?P<theta>[^/]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CaloFlow observables from an existing generated.h5 sample.")
    parser.add_argument("--generated-file", type=Path, required=True, help="Path to generated.h5.")
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=None,
        help="Reference HDF5 file. If omitted, infer it from the experiment checkpoint config.",
    )
    parser.add_argument("--geometry", type=str, default=None, help="CCD geometry name, e.g. CCD3.")
    parser.add_argument("--energy", type=float, default=None, help="Condition energy label used in plot titles.")
    parser.add_argument("--phi", type=float, default=None, help="Condition phi used in plot titles.")
    parser.add_argument("--theta", type=float, default=None, help="Condition theta used in plot titles.")
    parser.add_argument(
        "--num-showers",
        type=int,
        default=1000,
        help="Number of showers to load from each file for plotting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults to the generated.h5 parent directory.",
    )
    parser.add_argument(
        "--keep-previous",
        action="store_true",
        help="Keep existing PNGs in the output directory instead of removing them first.",
    )
    return parser.parse_args()


def _maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def infer_conditions_from_path(generated_file: Path):
    match = CONDITIONS_RE.search(str(generated_file.parent))
    if match is None:
        raise ValueError(f"Could not infer conditions from path: {generated_file}")

    geometry = match.group("geometry")
    energy = _maybe_float(match.group("energy"))
    phi = float(match.group("phi"))
    theta = float(match.group("theta"))
    return geometry, energy, phi, theta


def infer_experiment_dir(generated_file: Path) -> Path:
    for parent in generated_file.parents:
        if parent.name == "plots":
            return parent.parent
    raise ValueError(f"Could not infer experiment dir from generated file path: {generated_file}")


def values_match(lhs, rhs) -> bool:
    if lhs == rhs:
        return True
    try:
        return abs(float(lhs) - float(rhs)) < 1e-6
    except (TypeError, ValueError):
        return False


def infer_reference_file(generated_file: Path, geometry, energy, phi, theta) -> Path:
    experiment_dir = infer_experiment_dir(generated_file)
    config_candidates = [
        experiment_dir / "checkpoints" / "checkpoint_last" / "config.yaml",
        experiment_dir / "checkpoints" / "checkpoint_best" / "config.yaml",
    ]

    for config_path in config_candidates:
        if not config_path.is_file():
            continue

        cfg = OmegaConf.load(config_path)
        test_conditions = cfg.get("train", {}).get("test_conditions")
        if not test_conditions:
            continue

        for condition in test_conditions:
            if len(condition) < 5:
                continue
            cond_geometry, cond_energy, cond_phi, cond_theta, reference_file = condition[:5]
            if (
                cond_geometry == geometry
                and values_match(cond_energy, energy)
                and values_match(cond_phi, phi)
                and values_match(cond_theta, theta)
            ):
                return Path(reference_file)

        if len(test_conditions) == 1 and len(test_conditions[0]) >= 5:
            return Path(test_conditions[0][4])

    raise FileNotFoundError(
        f"Could not infer reference file from checkpoint config under {experiment_dir / 'checkpoints'}"
    )


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
        / 1000.0
    )
    return showers


def main():
    args = parse_args()
    generated_file = args.generated_file.resolve()

    if not generated_file.is_file():
        raise FileNotFoundError(f"Generated file not found: {generated_file}")

    inferred_geometry, inferred_energy, inferred_phi, inferred_theta = infer_conditions_from_path(generated_file)
    geometry = args.geometry or inferred_geometry
    energy = inferred_energy if args.energy is None else args.energy
    phi = inferred_phi if args.phi is None else args.phi
    theta = inferred_theta if args.theta is None else args.theta

    if not geometry.startswith("CCD"):
        raise ValueError(f"This helper currently supports CCD geometries only, got: {geometry}")

    reference_file = args.reference_file
    if reference_file is None:
        reference_file = infer_reference_file(generated_file, geometry, energy, phi, theta)
    reference_file = reference_file.resolve()
    if not reference_file.is_file():
        raise FileNotFoundError(f"Reference file not found: {reference_file}")

    output_dir = (args.output_dir or generated_file.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading generated sample from %s", generated_file)
    generated_showers = load_ccd_showers(generated_file, geometry, args.num_showers)
    logger.info("Loading reference sample from %s", reference_file)
    reference_showers = load_ccd_showers(reference_file, geometry, args.num_showers)

    num_showers = min(reference_showers.shape[0], generated_showers.shape[0])
    if num_showers == 0:
        raise ValueError("At least one shower is required to plot observables.")

    if num_showers < args.num_showers:
        logger.warning(
            "Requested %s showers but only %s are available in both files; plotting the shared prefix.",
            args.num_showers,
            num_showers,
        )

    logger.info(
        "Plotting %s showers for geometry=%s energy=%s phi=%s theta=%s into %s",
        num_showers,
        geometry,
        energy,
        phi,
        theta,
        output_dir,
    )
    compare_observables(
        reference_showers[:num_showers],
        generated_showers[:num_showers],
        output_dir,
        geometry,
        energy,
        phi,
        theta,
        keep_previous=args.keep_previous,
    )
    logger.info("Finished writing plots to %s", output_dir)


if __name__ == "__main__":
    sys.exit(main())
