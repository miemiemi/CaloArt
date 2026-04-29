import logging
import os
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default

    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, falling back to %d", name, value, default)
        return default

    return max(1, parsed)


def _resolve_gzip_threads() -> int:
    configured = os.getenv("CALOFLOW_H5_GZIP_THREADS")
    if configured not in (None, ""):
        return _get_env_int("CALOFLOW_H5_GZIP_THREADS", 1)

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus not in (None, ""):
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            logger.warning(
                "Invalid integer for SLURM_CPUS_PER_TASK=%r, falling back to single-threaded HDF5 gzip",
                slurm_cpus,
            )

    return 1


def _chunk_rows_for_array(array: np.ndarray) -> int:
    configured_rows = os.getenv("CALOFLOW_H5_CHUNK_ROWS")
    if configured_rows not in (None, ""):
        return min(array.shape[0], _get_env_int("CALOFLOW_H5_CHUNK_ROWS", 1))

    target_chunk_bytes = _get_env_int("CALOFLOW_H5_TARGET_CHUNK_MB", 8) * 1024 * 1024
    row_bytes = int(np.prod(array.shape[1:], dtype=np.int64) * array.dtype.itemsize) if array.ndim > 1 else array.dtype.itemsize
    return max(1, min(array.shape[0], target_chunk_bytes // max(1, row_bytes)))


def _chunk_shape_for_array(array: np.ndarray, chunk_rows: int) -> tuple[int, ...]:
    if array.ndim == 1:
        return (chunk_rows,)
    return (chunk_rows, *array.shape[1:])


def _compress_chunk(chunk: np.ndarray, compression_level: int) -> bytes:
    contiguous_chunk = np.ascontiguousarray(chunk)
    return zlib.compress(memoryview(contiguous_chunk), level=compression_level)


def _write_dataset_parallel_gzip(
    handle: h5py.File,
    name: str,
    data: np.ndarray,
    compression_level: int,
    num_threads: int,
) -> None:
    if data.shape[0] == 0:
        handle.create_dataset(name, data=data, compression="gzip", compression_opts=compression_level)
        return

    chunk_rows = _chunk_rows_for_array(data)
    chunk_shape = _chunk_shape_for_array(data, chunk_rows)
    dataset = handle.create_dataset(
        name,
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunk_shape,
        compression="gzip",
        compression_opts=compression_level,
    )

    if num_threads <= 1 or not hasattr(dataset.id, "write_direct_chunk"):
        dataset[...] = np.ascontiguousarray(data)
        return

    tail_offsets = (0,) * (data.ndim - 1)
    max_pending = max(1, num_threads * 2)

    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            pending: deque[tuple[int, object]] = deque()
            next_start = 0

            while next_start < data.shape[0] or pending:
                while next_start < data.shape[0] and len(pending) < max_pending:
                    next_stop = min(next_start + chunk_rows, data.shape[0])
                    future = executor.submit(
                        _compress_chunk,
                        data[next_start:next_stop],
                        compression_level,
                    )
                    pending.append((next_start, future))
                    next_start = next_stop

                chunk_start, future = pending.popleft()
                dataset.id.write_direct_chunk(
                    (chunk_start, *tail_offsets),
                    future.result(),
                )
    except Exception:
        logger.exception(
            "Parallel HDF5 gzip write failed for dataset %s, falling back to standard h5py gzip writer",
            name,
        )
        dataset[...] = np.ascontiguousarray(data)


def _create_dataset(
    handle: h5py.File,
    name: str,
    data: np.ndarray,
    compression_level: int,
    num_threads: int,
) -> None:
    if num_threads > 1:
        _write_dataset_parallel_gzip(
            handle,
            name,
            data,
            compression_level=compression_level,
            num_threads=num_threads,
        )
        return

    handle.create_dataset(name, data=np.ascontiguousarray(data), compression="gzip", compression_opts=compression_level)


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
    showers = np.asarray(showers) * scale
    energy = energy * scale
    
    # Check if the input is a single value and broadcast it to the length of showers
    if np.isscalar(energy):
        energy = np.full(len(showers), energy)
    if np.isscalar(theta):
        theta = np.full(len(showers), theta)
    if np.isscalar(phi):
        phi = np.full(len(showers), phi)

    energy = np.asarray(energy)
    phi = np.asarray(phi)
    theta = np.asarray(theta)

    num_threads = _resolve_gzip_threads()
    compression_level = _get_env_int("CALOFLOW_H5_GZIP_LEVEL", 9)
    logger.info(
        "Saving %s with HDF5 gzip level=%d using %d CPU thread(s)",
        output_path,
        compression_level,
        num_threads,
    )

    if not is_ccd:
        with h5py.File(output_path, "w") as f:
            _create_dataset(f, "showers", showers, compression_level, num_threads)
            _create_dataset(f, "incident_energy", np.asarray(energy), compression_level, num_threads)
            _create_dataset(f, "incident_phi", np.asarray(phi), compression_level, num_threads)
            _create_dataset(f, "incident_theta", np.asarray(theta), compression_level, num_threads)
    else:
        # transpose showers, sampling fraction of 0.033
        showers = showers.transpose(0, 3, 2, 1).reshape(showers.shape[0], -1) / 0.033
        with h5py.File(output_path, "w") as f:
            _create_dataset(f, "showers", showers, compression_level, num_threads)
            _create_dataset(f, "incident_energies", np.asarray(energy), compression_level, num_threads)
