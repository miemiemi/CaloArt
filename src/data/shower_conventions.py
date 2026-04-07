import numpy as np


CCD_SAMPLING_FRACTION = 0.033


def get_sampling_fraction(geometry=None):
    if geometry is None:
        return 1.0
    if geometry.startswith("CCD"):
        return CCD_SAMPLING_FRACTION
    return 1.0


def convert_to_evaluator_energy_space(showers, incident_energy, geometry=None):
    """Convert internal showers/energies to the saved/evaluator energy convention."""
    showers = np.asarray(showers, dtype=np.float32)
    incident_energy = np.asarray(incident_energy, dtype=np.float32)

    shower_scale = 1000.0 / get_sampling_fraction(geometry)
    energy_scale = 1000.0
    return showers * shower_scale, incident_energy * energy_scale


def compute_event_energy_ratio(showers, incident_energy, geometry=None):
    """Compute Edep/Einc in the same convention used by saved H5s/evaluator."""
    showers = np.asarray(showers, dtype=np.float32)
    incident_energy = np.asarray(incident_energy, dtype=np.float32).reshape(-1)
    flat_showers = showers.reshape(showers.shape[0], -1)
    event_sum_internal = flat_showers.sum(axis=1, dtype=np.float64)
    return event_sum_internal / incident_energy / get_sampling_fraction(geometry)
