"""FPD/KPD utilities for calorimeter shower evaluation."""

import logging
from pathlib import Path

import numpy as np

from src.evaluation.hlf.HighLevelFeatures import HighLevelFeatures

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
CCD_SAMPLING_FRACTION = 0.033

GEOMETRY_TO_PARTICLE = {
    "CCD2": "electron",
    "CCD3": "electron",
    "1-photons": "photon",
    "1-pions": "pion",
}

GEOMETRY_TO_XML = {
    "CCD2": REPO_ROOT / "cc_metrics" / "binning_dataset_2.xml",
    "CCD3": REPO_ROOT / "cc_metrics" / "binning_dataset_3.xml",
    "1-photons": REPO_ROOT / "cc_metrics" / "binning_dataset_1_photons.xml",
    "1-pions": REPO_ROOT / "cc_metrics" / "binning_dataset_1_pions.xml",
}


def _get_sampling_fraction(geometry):
    if geometry is None:
        return 1.0
    if geometry.startswith("CCD"):
        return CCD_SAMPLING_FRACTION
    return 1.0


def convert_to_fpd_evaluation_space(showers, incident_energy, geometry=None):
    """Convert CaloFlow internal showers to the evaluator convention.

    CaloFlow keeps dataset showers in GeV and, for CCD geometries, applies the
    detector sampling fraction on load. The reference FPD/KPD evaluator expects
    showers in MeV and in the raw CCD voxel convention, so CCD inputs must also
    be divided by the sampling fraction.
    """
    showers = np.asarray(showers, dtype=np.float32)
    incident_energy = np.asarray(incident_energy, dtype=np.float32)

    shower_scale = 1000.0 / _get_sampling_fraction(geometry)
    energy_scale = 1000.0
    return showers * shower_scale, incident_energy * energy_scale


def prepare_fpd_inputs(showers, incident_energy, geometry=None):
    """Prepare showers and energies exactly as `trainer.test()` sends them to FPD."""
    showers, incident_energy = convert_to_fpd_evaluation_space(
        showers,
        incident_energy,
        geometry=geometry,
    )
    showers = _to_evaluator_layout(showers, geometry=geometry)
    return showers, incident_energy


def get_evaluator_threshold_from_internal_noise(noise_level, geometry=None):
    """Map the internal GeV noise threshold to the evaluator shower convention."""
    return float(noise_level) * 1000.0 / _get_sampling_fraction(geometry)


def prepare_high_data_for_classifier(voxel_orig, E_inc_orig, hlf_class, label, cut=0.0):
    """Takes voxel data, extracts high-level features, appends label, returns array.

    Ported from ugr_evaluation/evaluate.py.
    """
    E_inc = E_inc_orig.copy()
    E_layer = []
    for layer_id in hlf_class.GetElayers():
        E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
    EC_etas = []
    EC_phis = []
    Width_etas = []
    Width_phis = []
    for layer_id in hlf_class.layersBinnedInAlpha:
        EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
        Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
        Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
    E_layer = np.concatenate(E_layer, axis=1)
    EC_etas = np.concatenate(EC_etas, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    Width_etas = np.concatenate(Width_etas, axis=1)
    Width_phis = np.concatenate(Width_phis, axis=1)
    ret = np.concatenate(
        [
            np.log10(E_inc),
            np.log10(E_layer + 1e-8),
            EC_etas / 1e2,
            EC_phis / 1e2,
            Width_etas / 1e2,
            Width_phis / 1e2,
            label * np.ones_like(E_inc),
        ],
        axis=1,
    )
    return ret


def _flatten_showers(showers):
    """Convert shower tensors to the flat voxel layout expected by HighLevelFeatures."""
    showers = np.asarray(showers)
    if showers.ndim < 2:
        raise ValueError(f"Expected shower array with at least 2 dims, got shape {showers.shape}.")
    if showers.ndim == 2:
        return showers
    return showers.reshape(showers.shape[0], -1)


def _to_evaluator_layout(showers, geometry=None):
    """Restore the raw voxel ordering expected by the evaluator."""
    showers = np.asarray(showers)
    if geometry is not None and geometry.startswith("CCD") and showers.ndim == 4:
        return showers.transpose(0, 3, 2, 1).reshape(showers.shape[0], -1)
    return _flatten_showers(showers)


def resolve_fpd_config(geometry=None, particle=None, xml_filename=None):
    """Resolve particle and XML defaults from geometry when possible."""
    if particle is None:
        if geometry is None or geometry not in GEOMETRY_TO_PARTICLE:
            raise ValueError(
                "FPD/KPD requires `particle` or a supported `geometry` "
                f"(got geometry={geometry!r})."
            )
        particle = GEOMETRY_TO_PARTICLE[geometry]

    if xml_filename is None:
        if geometry is None or geometry not in GEOMETRY_TO_XML:
            raise ValueError(
                "FPD/KPD requires `xml_filename` or a supported `geometry` "
                f"(got geometry={geometry!r})."
            )
        xml_filename = GEOMETRY_TO_XML[geometry]

    xml_path = Path(xml_filename).expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"FPD/KPD XML file not found: {xml_path}")

    return {
        "particle": particle,
        "xml_filename": xml_path,
    }


def compute_fpd_kpd(
    generated_showers,
    reference_showers,
    generated_energy,
    reference_energy,
    particle=None,
    xml_filename=None,
    geometry=None,
    cut=0.0,
    output_dir=None,
    dataset_name=None,
    min_samples=10000,
    batch_size=10000,
):
    """Compute FPD and KPD between generated and reference calorimeter showers.

    Args:
        generated_showers: numpy array of generated shower voxels, shape (N, num_voxels)
        reference_showers: numpy array of reference shower voxels, shape (M, num_voxels)
        generated_energy: numpy array of incident energies for generated, shape (N, 1)
        reference_energy: numpy array of incident energies for reference, shape (M, 1)
        particle: particle type string (e.g. "electron", "photon", "pion")
        xml_filename: path to the CaloChallenge binning XML file
        geometry: optional geometry name used to infer particle/XML defaults
        cut: energy cut threshold (default 0.0)
        output_dir: optional directory to save FPD/KPD results as a text file
        dataset_name: optional dataset name for the output filename

    Returns:
        dict with keys: fpd_val, fpd_err, kpd_val, kpd_err
    """
    import jetnet

    resolved = resolve_fpd_config(
        geometry=geometry,
        particle=particle,
        xml_filename=xml_filename,
    )
    particle = resolved["particle"]
    xml_filename = resolved["xml_filename"]

    # Ensure energy arrays are 2D (N, 1)
    if generated_energy.ndim == 1:
        generated_energy = generated_energy.reshape(-1, 1)
    if reference_energy.ndim == 1:
        reference_energy = reference_energy.reshape(-1, 1)

    generated_showers = _to_evaluator_layout(generated_showers, geometry=geometry)
    reference_showers = _to_evaluator_layout(reference_showers, geometry=geometry)

    # Clean generated showers
    np.nan_to_num(generated_showers, copy=False, nan=0.0, neginf=0.0, posinf=0.0)
    generated_showers[generated_showers < cut] = 0.0
    reference_showers = reference_showers.copy()
    reference_showers[reference_showers < cut] = 0.0

    # Calculate high-level features
    logger.info("Computing high-level features for FPD/KPD...")
    hlf_gen = HighLevelFeatures(particle, filename=str(xml_filename))
    hlf_gen.CalculateFeatures(generated_showers)
    hlf_gen.Einc = generated_energy

    hlf_ref = HighLevelFeatures(particle, filename=str(xml_filename))
    hlf_ref.CalculateFeatures(reference_showers)
    hlf_ref.Einc = reference_energy

    # Extract high-level feature arrays (remove class label column)
    source_array = prepare_high_data_for_classifier(
        generated_showers, generated_energy, hlf_gen, 0.0, cut=cut
    )[:, :-1]
    reference_array = prepare_high_data_for_classifier(
        reference_showers, reference_energy, hlf_ref, 1.0, cut=cut
    )[:, :-1]

    # Compute FPD and KPD
    logger.info("Computing FPD...")
    effective_min_samples = min(min_samples, len(reference_array), len(source_array))
    if effective_min_samples < min_samples:
        logger.warning(
            "Reducing FPD min_samples from %s to %s to match available events.",
            min_samples,
            effective_min_samples,
        )
    fpd_val, fpd_err = jetnet.evaluation.fpd(
        reference_array, source_array, min_samples=effective_min_samples
    )
    logger.info("Computing KPD...")
    kpd_val, kpd_err = jetnet.evaluation.kpd(
        reference_array, source_array, batch_size=min(batch_size, len(source_array))
    )

    result_str = (
        f"FPD (x10^3): {fpd_val * 1e3:.4f} ± {fpd_err * 1e3:.4f}\n"
        f"KPD (x10^3): {kpd_val * 1e3:.4f} ± {kpd_err * 1e3:.4f}"
    )
    logger.info(f"FPD/KPD results:\n{result_str}")

    # Optionally save to file
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"fpd_kpd_{dataset_name}.txt" if dataset_name else "fpd_kpd.txt"
        with open(output_dir / fname, "w") as f:
            f.write(result_str)

    return {
        "fpd_val": fpd_val,
        "fpd_err": fpd_err,
        "kpd_val": kpd_val,
        "kpd_err": kpd_err,
    }
