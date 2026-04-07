#!/usr/bin/env python3
"""Supplement dropped events after reject-redraw and rebuild a complete HDF5 sample."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf

rootutils.setup_root(__file__, pythonpath=True)

from src.data.preprocessing import CaloShowerPreprocessor
from src.data.shower_conventions import get_sampling_fraction
from src.data.utils import save_showers
from src.flow.reject_redraw import compute_redraw_mask, filter_model_sample_kwargs
from src.models.calodit_3drope import FinalLayer as LegacyGatedFinalLayer
from src.models.factory import create_model_from_config
from src.utils import get_logger, import_class_by_name, set_seed, to_device

LOGGER = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find missing incident energies by aligning a dropped generated.h5 against its "
            "reference HDF5, resample replacements, and write a supplemented output HDF5."
        )
    )
    parser.add_argument("--generated-h5", type=Path, required=True, help="Incomplete generated HDF5 after reject-redraw.")
    parser.add_argument("--reference-h5", type=Path, required=True, help="Reference/fullsim HDF5 that defined the original conditioning order.")
    parser.add_argument("--model-path", type=Path, required=True, help="Exported model with saved config.")
    parser.add_argument("--output-h5", type=Path, default=None, help="Output path for the supplemented HDF5.")
    parser.add_argument("--supplement-h5", type=Path, default=None, help="Optional path to save only the replacement events.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument(
        "--reject-summary",
        type=Path,
        default=None,
        help="Optional reject_redraw_summary.yaml to recover the original thresholds.",
    )
    parser.add_argument("--geometry", type=str, default=None, help="Geometry name, e.g. CCD3. Inferred from the path when omitted.")
    parser.add_argument("--phi", type=float, default=None, help="Incident phi used for sampling. Inferred from the path when omitted.")
    parser.add_argument("--theta", type=float, default=None, help="Incident theta used for sampling. Inferred from the path when omitted.")
    parser.add_argument("--batch-size", type=int, default=256, help="Sampling batch size for each supplement round.")
    parser.add_argument("--max-rounds", type=int, default=32, help="Maximum supplement rounds for unresolved samples.")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Optional override for sampler steps.")
    parser.add_argument("--sampling-solver", type=str, default=None, help="Optional override for sampler solver.")
    parser.add_argument("--reject-max-ratio", type=float, default=None, help="Optional override for reject_redraw_max_ratio.")
    parser.add_argument(
        "--reject-nonfinite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override reject_redraw_reject_nonfinite. Use --no-reject-nonfinite to disable.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Force a device such as cuda:0 or cpu.")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for energy alignment.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for energy alignment.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write outputs even if some supplement events are still unresolved after max rounds.",
    )
    return parser.parse_args()


def _load_saved_cfg(model_path: Path):
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    return OmegaConf.create(state["config"])


def _build_model_with_legacy_final_layer(cfg):
    architecture_cfg = OmegaConf.create(cfg.model.architecture)
    method_cfg = OmegaConf.create(cfg.method)

    architecture_cls = import_class_by_name(architecture_cfg["target"])
    method_cls = import_class_by_name(method_cfg["target"])

    architecture = architecture_cls(**architecture_cfg.get("init_args", {}))
    architecture.final_layer = LegacyGatedFinalLayer(
        channels=architecture.model_channels,
        patch_size=architecture.patch_size,
        out_channels=architecture.out_channels,
        use_checkpoint=architecture.use_checkpoint,
        use_rmsnorm=architecture.use_rmsnorm,
    )
    architecture.final_layer_uses_pos_emb = False

    model = method_cls(model=architecture, **method_cfg.get("init_args", {}))
    model.load_state(cfg.model.model_path)
    model.eval()
    return model


def load_model(model_path: Path):
    cfg = _load_saved_cfg(model_path)
    model_cfg = OmegaConf.create({"model_path": str(model_path)})
    method_cfg = OmegaConf.create({})
    try:
        model = create_model_from_config(model_cfg, method_cfg)
    except RuntimeError as exc:
        error_text = str(exc)
        legacy_final_layer_mismatch = (
            "final_layer.scale_shift_table" in error_text
            and "final_layer.adaLN_modulation.1.weight" in error_text
        )
        if not legacy_final_layer_mismatch:
            raise
        cfg.model.model_path = str(model_path)
        model = _build_model_with_legacy_final_layer(cfg)
    if model.config is None:
        model.save_config(cfg)
    return model, OmegaConf.create(model.config)


def infer_metadata_from_path(path: Path):
    pattern = re.compile(r"Geo_(?P<geometry>[^_]+)_E_[^_]+_Phi_(?P<phi>[^_]+)_Theta_(?P<theta>[^/]+)")
    for candidate in [path.name, path.parent.name, path.parent.parent.name]:
        match = pattern.search(candidate)
        if match is not None:
            return (
                match.group("geometry"),
                float(match.group("phi")),
                float(match.group("theta")),
            )
    return None, None, None


def find_energy_key(h5file: h5py.File) -> str:
    if "incident_energies" in h5file:
        return "incident_energies"
    if "incident_energy" in h5file:
        return "incident_energy"
    raise KeyError("Could not find an incident energy dataset.")


def _energy_to_internal(saved_energy):
    return np.asarray(saved_energy, dtype=np.float32) / 1000.0


def _energy_to_saved(internal_energy):
    return np.asarray(internal_energy, dtype=np.float32) * 1000.0


def find_missing_indices(reference_energy, generated_energy, *, rtol, atol):
    ref = np.asarray(reference_energy, dtype=np.float32).reshape(-1)
    gen = np.asarray(generated_energy, dtype=np.float32).reshape(-1)

    missing = []
    i = 0
    j = 0
    while i < len(ref) and j < len(gen):
        if np.isclose(ref[i], gen[j], rtol=rtol, atol=atol):
            i += 1
            j += 1
        else:
            missing.append(i)
            i += 1

    if i < len(ref):
        missing.extend(range(i, len(ref)))

    keep_mask = np.ones(len(ref), dtype=bool)
    keep_mask[missing] = False
    if keep_mask.sum() != len(gen):
        raise RuntimeError(
            f"Alignment failed: reference keep count {int(keep_mask.sum())} does not match generated count {len(gen)}."
        )
    if not np.allclose(ref[keep_mask], gen, rtol=rtol, atol=atol):
        mismatch = int(np.flatnonzero(~np.isclose(ref[keep_mask], gen, rtol=rtol, atol=atol))[0])
        raise RuntimeError(
            "Reference/generated energies are not consistent with pure deletions. "
            f"First mismatch after deletion alignment is at kept position {mismatch}."
        )

    return np.asarray(missing, dtype=np.int64)


def _sample_batch(model, preprocessor, device, incident_energy_gev, phi, theta, sampling_cfg):
    energy = np.asarray(incident_energy_gev, dtype=np.float32).reshape(-1, 1)
    cond_e = torch.as_tensor(energy, dtype=torch.float32)
    cond_phi = torch.full_like(cond_e, float(phi))
    cond_theta = torch.full_like(cond_e, float(theta))
    _, conditions = preprocessor.transform(conditions=(cond_e, cond_phi, cond_theta))
    conditions = to_device(conditions, device)

    with torch.inference_mode():
        generated_events = model.sample(
            conditions=conditions,
            progress=False,
            **filter_model_sample_kwargs(sampling_cfg),
        ).squeeze(1)
        generated_events, _ = preprocessor.inverse_transform(generated_events, conditions)
    return generated_events.detach().cpu().numpy()


def sample_supplements(
    *,
    model,
    preprocessor,
    device,
    energies_gev,
    geometry,
    phi,
    theta,
    sampling_cfg,
    max_ratio,
    reject_nonfinite,
    batch_size,
    max_rounds,
):
    pending = np.arange(len(energies_gev), dtype=np.int64)
    accepted = [None] * len(energies_gev)
    rounds = []

    for round_idx in range(1, max_rounds + 1):
        if len(pending) == 0:
            break

        pending_before = int(len(pending))
        next_pending = []
        total_bad = 0
        total_nonfinite = 0
        total_ratio = 0
        round_ratio_max = 0.0

        for start in range(0, len(pending), batch_size):
            chunk_indices = pending[start:start + batch_size]
            chunk_energies = energies_gev[chunk_indices]
            chunk_events = _sample_batch(
                model,
                preprocessor,
                device,
                chunk_energies,
                phi,
                theta,
                sampling_cfg,
            )
            bad_mask, chunk_summary = compute_redraw_mask(
                chunk_events,
                chunk_energies.reshape(-1, 1),
                geometry=geometry,
                max_ratio=max_ratio,
                reject_nonfinite=reject_nonfinite,
            )

            total_bad += int(chunk_summary["combined_bad_count"])
            total_nonfinite += int(chunk_summary["nonfinite_count"])
            total_ratio += int(chunk_summary["ratio_count"])
            round_ratio_max = max(round_ratio_max, float(chunk_summary["event_ratio_max"]))

            for local_idx, global_idx in enumerate(chunk_indices):
                if bad_mask[local_idx]:
                    next_pending.append(int(global_idx))
                else:
                    accepted[int(global_idx)] = chunk_events[local_idx]

        pending = np.asarray(next_pending, dtype=np.int64)
        rounds.append(
            {
                "round": round_idx,
                "pending_before": pending_before,
                "accepted_this_round": pending_before - len(pending),
                "remaining_after": int(len(pending)),
                "nonfinite_count": total_nonfinite,
                "ratio_count": total_ratio,
                "combined_bad_count": total_bad,
                "event_ratio_max": round_ratio_max,
            }
        )

    unresolved = [int(idx) for idx, event in enumerate(accepted) if event is None]
    resolved_indices = [idx for idx, event in enumerate(accepted) if event is not None]
    resolved_events = (
        np.stack([accepted[idx] for idx in resolved_indices], axis=0)
        if resolved_indices
        else np.empty((0,), dtype=np.float32)
    )
    return resolved_events, np.asarray(resolved_indices, dtype=np.int64), unresolved, rounds


def prepare_saved_showers(showers_internal, *, is_ccd):
    showers_internal = np.asarray(showers_internal, dtype=np.float32)
    if showers_internal.shape[0] == 0:
        return showers_internal
    if is_ccd:
        return (
            showers_internal.transpose(0, 3, 2, 1).reshape(showers_internal.shape[0], -1)
            * (1000.0 / get_sampling_fraction("CCD"))
        ).astype(np.float32)
    return (showers_internal * 1000.0).astype(np.float32)


def _copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _create_like_dataset(dst_file, name, src_dataset, total_len):
    chunks = src_dataset.chunks
    if chunks is not None:
        chunks = (min(chunks[0], total_len),) + chunks[1:]
    dataset = dst_file.create_dataset(
        name,
        shape=(total_len,) + src_dataset.shape[1:],
        dtype=src_dataset.dtype,
        chunks=chunks,
        compression=src_dataset.compression,
        compression_opts=src_dataset.compression_opts,
        shuffle=src_dataset.shuffle,
        fletcher32=src_dataset.fletcher32,
        scaleoffset=src_dataset.scaleoffset,
    )
    _copy_attrs(src_dataset, dataset)
    return dataset


def write_merged_h5(
    *,
    input_h5,
    output_h5,
    missing_indices,
    supplement_saved_showers,
    supplement_saved_energy,
    supplement_phi,
    supplement_theta,
):
    missing_indices = np.asarray(missing_indices, dtype=np.int64)

    with h5py.File(input_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        _copy_attrs(src, dst)
        total_len = int(len(missing_indices) + src["showers"].shape[0])
        energy_key = find_energy_key(src)

        def _reshape_like(values, dataset):
            values = np.asarray(values, dtype=dataset.dtype)
            target_shape = (len(values),) + dataset.shape[1:]
            return values.reshape(target_shape)

        supplement_by_key = {
            "showers": _reshape_like(supplement_saved_showers, src["showers"]),
            energy_key: _reshape_like(supplement_saved_energy, src[energy_key]),
        }
        if "incident_phi" in src:
            supplement_by_key["incident_phi"] = _reshape_like(supplement_phi, src["incident_phi"])
        if "incident_theta" in src:
            supplement_by_key["incident_theta"] = _reshape_like(supplement_theta, src["incident_theta"])

        for key in src.keys():
            if key not in supplement_by_key:
                raise KeyError(f"Unsupported dataset '{key}' in {input_h5}.")

            src_dataset = src[key]
            dst_dataset = _create_like_dataset(dst, key, src_dataset, total_len)
            supplement_values = supplement_by_key[key]
            src_pos = 0
            out_pos = 0

            for supplement_idx, miss_idx in enumerate(missing_indices.tolist()):
                segment_len = miss_idx - out_pos
                if segment_len > 0:
                    dst_dataset[out_pos:miss_idx] = src_dataset[src_pos:src_pos + segment_len]
                    src_pos += segment_len
                    out_pos = miss_idx
                dst_dataset[out_pos:out_pos + 1] = supplement_values[supplement_idx:supplement_idx + 1]
                out_pos += 1

            remaining = src_dataset.shape[0] - src_pos
            if remaining > 0:
                dst_dataset[out_pos:out_pos + remaining] = src_dataset[src_pos:src_pos + remaining]


def main():
    args = parse_args()
    set_seed(args.seed, all_gpus=True)

    generated_h5 = args.generated_h5.resolve()
    reference_h5 = args.reference_h5.resolve()
    output_h5 = args.output_h5.resolve() if args.output_h5 is not None else generated_h5.with_name(
        f"{generated_h5.stem}_supplemented{generated_h5.suffix}"
    )
    summary_json = args.summary_json.resolve() if args.summary_json is not None else output_h5.with_suffix(".summary.json")
    supplement_h5 = args.supplement_h5.resolve() if args.supplement_h5 is not None else None
    reject_summary_path = (
        args.reject_summary.resolve()
        if args.reject_summary is not None
        else generated_h5.with_name("reject_redraw_summary.yaml")
    )

    inferred_geometry, inferred_phi, inferred_theta = infer_metadata_from_path(generated_h5)
    geometry = args.geometry or inferred_geometry
    phi = args.phi if args.phi is not None else inferred_phi
    theta = args.theta if args.theta is not None else inferred_theta
    if geometry is None or phi is None or theta is None:
        raise ValueError("Could not infer geometry/phi/theta from the path. Please pass them explicitly.")

    model, cfg = load_model(args.model_path.resolve())
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    sampling_cfg = OmegaConf.create(cfg.get("sampling", {}))
    OmegaConf.set_struct(sampling_cfg, False)
    if args.sampling_steps is not None:
        sampling_cfg.steps = args.sampling_steps
    if args.sampling_solver is not None:
        sampling_cfg.solver = args.sampling_solver

    reject_summary_cfg = None
    if reject_summary_path.exists():
        reject_summary_cfg = OmegaConf.load(reject_summary_path)

    max_ratio = args.reject_max_ratio
    if max_ratio is None:
        max_ratio = sampling_cfg.get("reject_redraw_max_ratio")
    if max_ratio is None and reject_summary_cfg is not None:
        max_ratio = reject_summary_cfg.get("thresholds", {}).get("max_ratio")
    reject_nonfinite = args.reject_nonfinite
    if reject_nonfinite is None:
        reject_nonfinite = bool(sampling_cfg.get("reject_redraw_reject_nonfinite", True))
    if args.reject_nonfinite is None and reject_summary_cfg is not None:
        reject_nonfinite = bool(
            reject_summary_cfg.get("thresholds", {}).get("reject_nonfinite", reject_nonfinite)
        )

    preprocessor = CaloShowerPreprocessor(**cfg.preprocessing)

    with h5py.File(reference_h5, "r") as ref_file, h5py.File(generated_h5, "r") as gen_file:
        reference_energy_saved = ref_file[find_energy_key(ref_file)][:].reshape(-1)
        generated_energy_saved = gen_file[find_energy_key(gen_file)][:].reshape(-1)
        is_ccd = find_energy_key(gen_file) == "incident_energies"

    missing_indices = find_missing_indices(
        reference_energy_saved,
        generated_energy_saved,
        rtol=args.rtol,
        atol=args.atol,
    )
    missing_energy_saved = reference_energy_saved[missing_indices]
    missing_energy_internal = _energy_to_internal(missing_energy_saved)

    LOGGER.info(
        "Found %d missing event(s) in %s relative to %s.",
        len(missing_indices),
        generated_h5,
        reference_h5,
    )

    summary = {
        "generated_h5": str(generated_h5),
        "reference_h5": str(reference_h5),
        "model_path": str(args.model_path.resolve()),
        "output_h5": str(output_h5),
        "supplement_h5": None if supplement_h5 is None else str(supplement_h5),
        "reject_summary": str(reject_summary_path) if reject_summary_path.exists() else None,
        "geometry": geometry,
        "phi": float(phi),
        "theta": float(theta),
        "input_count": int(len(generated_energy_saved)),
        "target_count": int(len(reference_energy_saved)),
        "missing_count": int(len(missing_indices)),
        "missing_indices": missing_indices.tolist(),
        "missing_energy_GeV": [float(x) for x in missing_energy_internal.tolist()],
        "reject_thresholds": {
            "max_ratio": None if max_ratio is None else float(max_ratio),
            "reject_nonfinite": bool(reject_nonfinite),
        },
        "sampling_overrides": {
            "steps": sampling_cfg.get("steps"),
            "solver": sampling_cfg.get("solver"),
        },
        "rounds": [],
    }

    if len(missing_indices) == 0:
        if output_h5 != generated_h5:
            shutil.copy2(generated_h5, output_h5)
        summary["status"] = "already_complete"
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2))
        LOGGER.info("Sample is already complete. Wrote summary to %s", summary_json)
        return

    resolved_events, resolved_local_indices, unresolved_local_indices, rounds = sample_supplements(
        model=model,
        preprocessor=preprocessor,
        device=device,
        energies_gev=missing_energy_internal,
        geometry=geometry,
        phi=phi,
        theta=theta,
        sampling_cfg=sampling_cfg,
        max_ratio=max_ratio,
        reject_nonfinite=reject_nonfinite,
        batch_size=args.batch_size,
        max_rounds=args.max_rounds,
    )

    summary["rounds"] = rounds
    summary["resolved_count"] = int(len(resolved_local_indices))
    summary["unresolved_count"] = int(len(unresolved_local_indices))
    summary["unresolved_indices"] = [int(missing_indices[idx]) for idx in unresolved_local_indices]
    summary["unresolved_energy_GeV"] = [float(missing_energy_internal[idx]) for idx in unresolved_local_indices]

    if unresolved_local_indices and not args.allow_partial:
        summary["status"] = "failed_unresolved_remaining"
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2))
        raise RuntimeError(
            f"{len(unresolved_local_indices)} supplement event(s) still unresolved after {args.max_rounds} rounds. "
            f"See {summary_json} for details."
        )

    resolved_missing_indices = missing_indices[resolved_local_indices]
    resolved_missing_energy_internal = missing_energy_internal[resolved_local_indices]
    resolved_missing_energy_saved = _energy_to_saved(resolved_missing_energy_internal)

    if supplement_h5 is not None:
        supplement_h5.parent.mkdir(parents=True, exist_ok=True)
        if len(resolved_missing_indices) > 0:
            save_showers(
                resolved_events,
                resolved_missing_energy_internal.reshape(-1, 1),
                float(phi),
                float(theta),
                supplement_h5,
                is_ccd=is_ccd,
            )

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    if len(resolved_missing_indices) == 0:
        if output_h5 != generated_h5:
            shutil.copy2(generated_h5, output_h5)
    else:
        supplement_saved_showers = prepare_saved_showers(resolved_events, is_ccd=is_ccd)
        supplement_phi = np.full((len(resolved_missing_indices),), float(phi), dtype=np.float32)
        supplement_theta = np.full((len(resolved_missing_indices),), float(theta), dtype=np.float32)
        merge_output_h5 = output_h5
        replace_in_place = output_h5 == generated_h5
        if replace_in_place:
            merge_output_h5 = output_h5.with_name(f"{output_h5.stem}.supplement_tmp{output_h5.suffix}")
        write_merged_h5(
            input_h5=generated_h5,
            output_h5=merge_output_h5,
            missing_indices=resolved_missing_indices,
            supplement_saved_showers=supplement_saved_showers,
            supplement_saved_energy=resolved_missing_energy_saved,
            supplement_phi=supplement_phi,
            supplement_theta=supplement_theta,
        )
        if replace_in_place:
            os.replace(merge_output_h5, output_h5)

    summary["status"] = "completed_with_partial" if unresolved_local_indices else "completed"
    summary["resolved_missing_indices"] = resolved_missing_indices.tolist()
    summary["resolved_missing_energy_GeV"] = [float(x) for x in resolved_missing_energy_internal.tolist()]
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2))

    LOGGER.info(
        "Wrote supplemented sample to %s with %d resolved event(s). Summary: %s",
        output_h5,
        len(resolved_missing_indices),
        summary_json,
    )


if __name__ == "__main__":
    main()
