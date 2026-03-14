from pathlib import Path
from typing import Union

import h5py
import numpy as np


def load_showers(file_path: Union[str, Path], showers: np.ndarray, energy: np.ndarray, phi: np.ndarray, theta: np.ndarray,
            start_idx: int, convert_MeV_to_GeV: bool = True, is_ccd: bool = False):
    scale = 1000.0 if convert_MeV_to_GeV else 1.0
    if not is_ccd:
        with h5py.File(file_path, "r") as f:
            end_idx = start_idx + f["showers"].shape[0]
            f["showers"].read_direct(showers[start_idx:end_idx])
            f["incident_energy"].read_direct(energy[start_idx:end_idx])
            f["incident_phi"].read_direct(phi[start_idx:end_idx])
            f["incident_theta"].read_direct(theta[start_idx:end_idx])
    else:
        with h5py.File(file_path, "r") as f:
            end_idx = start_idx + f["showers"].shape[0]
            f["showers"].read_direct(showers[start_idx:end_idx])

            # shape mismatch for read_direct
            arr = np.empty((end_idx - start_idx, 1), dtype=np.float32)
            f["incident_energies"].read_direct(arr)            
            # only energy here
            energy[start_idx:end_idx] = arr.flatten()
            # leave theta and phi as zeros

    showers[start_idx:end_idx] /= scale
    energy[start_idx:end_idx] /= scale
    return end_idx


def save_showers(showers: np.ndarray, energy: Union[int, float, np.ndarray], phi: Union[float, np.ndarray], theta: Union[float, np.ndarray],
            output_path: Union[str, Path], convert_GeV_to_MeV: bool = True, is_ccd: bool = False):
    scale = 1000.0 if convert_GeV_to_MeV else 1.0
    showers = showers * scale
    energy = energy * scale
    
    # Check if the input is a single value and broadcast it to the length of showers
    if isinstance(energy, (int, float)):
        energy = np.full(len(showers), energy)
    if isinstance(theta, float):
        theta = np.full(len(showers), theta)
    if isinstance(phi, float):
        phi = np.full(len(showers), phi)

    if not is_ccd:
        with h5py.File(output_path, "w") as f:
            f.create_dataset("showers", data=showers, compression="gzip", compression_opts=9)
            f.create_dataset("incident_energy", data=energy, compression="gzip", compression_opts=9)
            f.create_dataset("incident_phi", data=phi, compression="gzip", compression_opts=9)
            f.create_dataset("incident_theta", data=theta, compression="gzip", compression_opts=9)
    else:
        # transpose showers, sampling fraction of 0.033
        showers = showers.transpose(0, 3, 2, 1).reshape(showers.shape[0], -1) / 0.033
        with h5py.File(output_path, "w") as f:
            f.create_dataset("showers", data=showers, compression="gzip", compression_opts=9)
            f.create_dataset("incident_energies", data=energy, compression="gzip", compression_opts=9)
