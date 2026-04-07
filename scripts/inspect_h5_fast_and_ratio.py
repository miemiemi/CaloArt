import argparse
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def to_json_number(value):
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def find_energy_key(h5file: h5py.File) -> str:
    if "incident_energies" in h5file:
        return "incident_energies"
    if "incident_energy" in h5file:
        return "incident_energy"
    raise KeyError("Could not find incident energy key")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--fast-summary-json", required=True)
    parser.add_argument("--ratio-summary-json", required=True)
    parser.add_argument("--chunk-events", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    quantiles = [0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 99.99, 100.0]
    ratio_thresholds = [1.0, 1.2, 1.5, 2.0, 2.6, 5.0, 10.0]
    finite_max_thresholds = [1e4, 1e6, 1e9, 1e20]

    path = Path(args.input_file)
    fast_path = Path(args.fast_summary_json)
    ratio_path = Path(args.ratio_summary_json)
    fast_path.parent.mkdir(parents=True, exist_ok=True)
    ratio_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as f:
        showers = f["showers"]
        voxels = int(showers.shape[1])
        n_events = int(showers.shape[0])
        energy_key = find_energy_key(f)
        energies = f[energy_key][:].reshape(-1).astype(np.float64)

        event_sum = np.empty(n_events, dtype=np.float64)
        event_max_finite = np.empty(n_events, dtype=np.float64)
        event_nan_cells = np.zeros(n_events, dtype=np.int64)
        event_inf_cells = np.zeros(n_events, dtype=np.int64)

        showers_nan = 0
        showers_inf = 0
        showers_finite_max = -np.inf

        first_nonfinite_events = []

        for start in range(0, n_events, args.chunk_events):
            stop = min(start + args.chunk_events, n_events)
            arr = showers[start:stop]
            nan_mask = np.isnan(arr)
            inf_mask = np.isinf(arr)
            finite_mask = np.isfinite(arr)

            showers_nan += int(nan_mask.sum())
            showers_inf += int(inf_mask.sum())

            chunk_finite = arr[finite_mask]
            if chunk_finite.size:
                showers_finite_max = max(showers_finite_max, float(chunk_finite.max()))

            event_sum[start:stop] = arr.sum(axis=1, dtype=np.float64)
            event_nan_cells[start:stop] = nan_mask.sum(axis=1, dtype=np.int64)
            event_inf_cells[start:stop] = inf_mask.sum(axis=1, dtype=np.int64)
            event_max_finite[start:stop] = np.max(np.where(finite_mask, arr, -np.inf), axis=1)

            if len(first_nonfinite_events) < args.top_k:
                bad_rows = np.where((event_nan_cells[start:stop] + event_inf_cells[start:stop]) > 0)[0]
                for local_idx in bad_rows:
                    if len(first_nonfinite_events) >= args.top_k:
                        break
                    global_idx = start + int(local_idx)
                    bad_cells = np.where(~finite_mask[local_idx])[0][:10]
                    first_nonfinite_events.append(
                        {
                            "event": global_idx,
                            "incident_energy": float(energies[global_idx]),
                            "nan_cells": int(event_nan_cells[global_idx]),
                            "inf_cells": int(event_inf_cells[global_idx]),
                            "first_bad_cells": [int(x) for x in bad_cells],
                        }
                    )

    event_has_nan = event_nan_cells > 0
    event_has_inf = event_inf_cells > 0
    event_has_nonfinite = event_has_nan | event_has_inf
    events_with_sum_inf = int(np.isinf(event_sum).sum())

    top_finite_idx = np.argsort(np.nan_to_num(event_max_finite, nan=-np.inf, posinf=np.inf))[-args.top_k:][::-1]
    top_events_by_finite_max = [
        {"event": int(i), "finite_max": to_json_number(event_max_finite[i])}
        for i in top_finite_idx
    ]

    fast_summary = {
        "path": str(path),
        "events": n_events,
        "voxels": voxels,
        "showers_nan": int(showers_nan),
        "showers_inf": int(showers_inf),
        "showers_finite_max": to_json_number(showers_finite_max),
        "events_with_nan": int(event_has_nan.sum()),
        "events_with_inf": int(event_has_inf.sum()),
        "events_with_nonfinite": int(event_has_nonfinite.sum()),
        "events_with_sum_inf": events_with_sum_inf,
        "events_with_max_finite_gt": {
            str(threshold): int((event_max_finite > threshold).sum())
            for threshold in finite_max_thresholds
        },
        "first_nonfinite_events": first_nonfinite_events,
        "top_events_by_finite_max": top_events_by_finite_max,
    }

    ratio = event_sum / energies
    finite_ratio = ratio[np.isfinite(ratio)]
    ratio_quantiles = np.percentile(finite_ratio, quantiles) if finite_ratio.size else np.full(len(quantiles), np.nan)
    top_ratio_idx = np.argsort(np.nan_to_num(ratio, nan=-np.inf, posinf=np.inf))[-args.top_k:][::-1]
    top_events_by_ratio = []
    for idx in top_ratio_idx:
        top_events_by_ratio.append(
            {
                "event": int(idx),
                "ratio": to_json_number(ratio[idx]),
                "incident_energy": float(energies[idx]),
            }
        )

    ratio_summary = {
        "path": str(path),
        "events": n_events,
        "event_sum_inf": int(np.isinf(event_sum).sum()),
        "event_sum_nan": int(np.isnan(event_sum).sum()),
        "finite_ratio_quantiles": {
            f"p{q:g}": to_json_number(v) for q, v in zip(quantiles, ratio_quantiles)
        },
        "finite_ratio_mean": to_json_number(finite_ratio.mean()) if finite_ratio.size else "nan",
        "threshold_counts": {
            str(threshold): int((finite_ratio > threshold).sum()) for threshold in ratio_thresholds
        },
        "threshold_fractions": {
            str(threshold): float((finite_ratio > threshold).mean()) if finite_ratio.size else 0.0
            for threshold in ratio_thresholds
        },
        "top_events_by_ratio": top_events_by_ratio,
    }

    fast_path.write_text(json.dumps(fast_summary, indent=2, ensure_ascii=False))
    ratio_path.write_text(json.dumps(ratio_summary, indent=2, ensure_ascii=False))
    print(f"fast_summary_written: {fast_path}")
    print(f"ratio_summary_written: {ratio_path}")


if __name__ == "__main__":
    main()
