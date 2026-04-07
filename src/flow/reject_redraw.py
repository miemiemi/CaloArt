import numpy as np

from src.data.shower_conventions import compute_event_energy_ratio


REDRAW_SAMPLING_KEYS = {
    "reject_redraw",
    "reject_redraw_max_rounds",
    "reject_redraw_max_ratio",
    "reject_redraw_reject_nonfinite",
}


def filter_model_sample_kwargs(sampling_args):
    return {
        key: value
        for key, value in sampling_args.items()
        if key not in REDRAW_SAMPLING_KEYS
    }


def compute_redraw_mask(generated_events, incident_energy, *, geometry=None, max_ratio=None, reject_nonfinite=True):
    flat = generated_events.reshape(generated_events.shape[0], -1)
    nonfinite_mask = ~np.isfinite(flat).all(axis=1)
    event_ratio = compute_event_energy_ratio(
        generated_events,
        incident_energy,
        geometry=geometry,
    )

    ratio_mask = np.zeros(len(generated_events), dtype=bool)
    if max_ratio is not None:
        ratio_mask = event_ratio > float(max_ratio)

    mask = ratio_mask.copy()
    if reject_nonfinite:
        mask |= nonfinite_mask

    summary = {
        "nonfinite_count": int(nonfinite_mask.sum()),
        "ratio_count": int(ratio_mask.sum()),
        "combined_bad_count": int(mask.sum()),
        "event_ratio_max": float(np.nanmax(event_ratio)) if len(event_ratio) else 0.0,
    }
    return mask, summary


def apply_reject_and_redraw(
    generated_events,
    incident_energy,
    *,
    geometry,
    sampling_args,
    sample_fn,
    original_events=None,
):
    if not sampling_args.get("reject_redraw", False):
        return generated_events, original_events, incident_energy, None

    max_ratio = sampling_args.get("reject_redraw_max_ratio")
    reject_nonfinite = bool(sampling_args.get("reject_redraw_reject_nonfinite", True))
    max_rounds = max(0, int(sampling_args.get("reject_redraw_max_rounds", 10)))

    redraw_batch_size = 256
    drop_unresolved = True

    summary = {
        "enabled": True,
        "thresholds": {
            "max_ratio": max_ratio,
            "reject_nonfinite": reject_nonfinite,
        },
        "max_rounds": max_rounds,
        "rounds": [],
    }

    mask, initial_summary = compute_redraw_mask(
        generated_events,
        incident_energy,
        geometry=geometry,
        max_ratio=max_ratio,
        reject_nonfinite=reject_nonfinite,
    )
    summary["initial"] = initial_summary

    for redraw_round in range(1, max_rounds + 1):
        bad_indices = np.flatnonzero(mask)
        if len(bad_indices) == 0:
            break

        for start in range(0, len(bad_indices), redraw_batch_size):
            stop = min(start + redraw_batch_size, len(bad_indices))
            batch_indices = bad_indices[start:stop]
            generated_events[batch_indices] = sample_fn(incident_energy[batch_indices])

        mask, round_summary = compute_redraw_mask(
            generated_events,
            incident_energy,
            geometry=geometry,
            max_ratio=max_ratio,
            reject_nonfinite=reject_nonfinite,
        )
        round_summary["round"] = redraw_round
        round_summary["remaining_bad_count"] = int(mask.sum())
        summary["rounds"].append(round_summary)

    unresolved = np.flatnonzero(mask)
    summary["final_bad_count"] = int(len(unresolved))
    summary["dropped_count"] = 0
    summary["kept_count"] = int(len(generated_events))

    if len(unresolved) > 0 and drop_unresolved:
        keep_mask = ~mask
        generated_events = generated_events[keep_mask]
        incident_energy = incident_energy[keep_mask]
        if original_events is not None:
            original_events = original_events[keep_mask]
        summary["dropped_count"] = int(len(unresolved))
        summary["kept_count"] = int(keep_mask.sum())

    return generated_events, original_events, incident_energy, summary
