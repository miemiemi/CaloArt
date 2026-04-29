"""Geometry helpers for calorimeter datasets."""

import math
from types import SimpleNamespace


CCD_GEOMETRIES = {
    "CCD2": {
        "NAME": "CCD2",
        "N_CELLS_Z": 45,
        "N_CELLS_PHI": 16,
        "N_CELLS_R": 9,
        "SIZE_Z": 3.4,  # mm = 2 x (0.3mm of Si + 1.4mm of W)
        "SIZE_R": 4.65,  # mm
        "RAW_FLAT_SIZE": 45 * 16 * 9,
    },
    "CCD3": {
        "NAME": "CCD3",
        "N_CELLS_Z": 45,
        "N_CELLS_PHI": 50,
        "N_CELLS_R": 18,
        "SIZE_Z": 3.4,
        "SIZE_R": 2.325,
        "RAW_FLAT_SIZE": 45 * 50 * 18,
    },
    "CCD3_REBINNED_45X25X9": {
        "NAME": "CCD3_REBINNED_45X25X9",
        "N_CELLS_Z": 45,
        "N_CELLS_PHI": 25,
        "N_CELLS_R": 9,
        "SIZE_Z": 3.4,
        "SIZE_R": 4.65,
        "RAW_FLAT_SIZE": 45 * 25 * 9,
    },
}


def infer_geometry_name(flat_size: int) -> str:
    for geometry_name, spec in CCD_GEOMETRIES.items():
        if spec["RAW_FLAT_SIZE"] == int(flat_size):
            return geometry_name
    raise ValueError(f"Unsupported CCD flat size {flat_size}. Known sizes: {[spec['RAW_FLAT_SIZE'] for spec in CCD_GEOMETRIES.values()]}")


def get_geometry(geometry_name: str) -> SimpleNamespace:
    if geometry_name not in CCD_GEOMETRIES:
        raise ValueError(f"Unsupported CCD geometry '{geometry_name}'. Expected one of {sorted(CCD_GEOMETRIES)}.")
    spec = dict(CCD_GEOMETRIES[geometry_name])
    spec["SIZE_PHI"] = 2 * math.pi / spec["N_CELLS_PHI"]
    return SimpleNamespace(**spec)


GEOMETRY = SimpleNamespace()


def set_geometry(geometry_name: str) -> SimpleNamespace:
    resolved = get_geometry(geometry_name)
    for key, value in vars(resolved).items():
        setattr(GEOMETRY, key, value)
    return GEOMETRY


set_geometry("CCD2")
