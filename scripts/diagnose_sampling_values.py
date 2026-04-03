#!/usr/bin/env python3
"""Diagnose where sampling-time numerical explosions first appear."""

from __future__ import annotations

import argparse
import json
from heapq import heappush, heappushpop
from pathlib import Path

import rootutils
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, pythonpath=True)

from src.data.dataset import CaloShowerDataset
from src.data.preprocessing import CaloShowerPreprocessor
from src.models.calodit_3drope import FinalLayer as LegacyGatedFinalLayer
from src.models.factory import create_model_from_config
from src.utils import import_class_by_name, set_seed, to_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace sample values before and during inverse preprocessing."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to an exported model with config.")
    parser.add_argument(
        "--condition-file",
        type=Path,
        default=None,
        help="Optional HDF5 file providing the conditioning energies. Defaults to the first test condition file in the saved config.",
    )
    parser.add_argument("--num-showers", type=int, default=10000, help="Number of showers to sample.")
    parser.add_argument("--batch-size", type=int, default=256, help="Sampling batch size.")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Optional override for sampler steps.")
    parser.add_argument("--sampling-solver", type=str, default=None, help="Optional override for sampler solver.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top finite values to keep per stage.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path for the structured diagnostics.",
    )
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda:0 or cpu.")
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


class StageStats:
    def __init__(self, name: str, top_k: int):
        self.name = name
        self.top_k = top_k
        self.tensor_count = 0
        self.event_count = 0
        self.value_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.events_with_nan = 0
        self.events_with_inf = 0
        self.finite_min = None
        self.finite_max = None
        self.first_nan = []
        self.first_inf = []
        self.top_finite = []

    def _append_first(self, bucket, coords, limit=10):
        remaining = limit - len(bucket)
        if remaining <= 0:
            return
        bucket.extend(coords[:remaining])

    def update(self, tensor: torch.Tensor, global_event_offset: int):
        flat = tensor.detach().reshape(tensor.shape[0], -1)
        self.tensor_count += 1
        self.event_count += int(flat.shape[0])
        self.value_count += int(flat.numel())

        nan_mask = torch.isnan(flat)
        inf_mask = torch.isinf(flat)
        finite_mask = torch.isfinite(flat)

        self.nan_count += int(nan_mask.sum().item())
        self.inf_count += int(inf_mask.sum().item())
        self.events_with_nan += int(nan_mask.any(dim=1).sum().item())
        self.events_with_inf += int(inf_mask.any(dim=1).sum().item())

        if len(self.first_nan) < 10 and nan_mask.any():
            coords = torch.nonzero(nan_mask, as_tuple=False).cpu().tolist()
            coords = [
                {"event": global_event_offset + int(row), "cell": int(col)}
                for row, col in coords
            ]
            self._append_first(self.first_nan, coords)

        if len(self.first_inf) < 10 and inf_mask.any():
            coords = torch.nonzero(inf_mask, as_tuple=False).cpu().tolist()
            coords = [
                {"event": global_event_offset + int(row), "cell": int(col)}
                for row, col in coords
            ]
            self._append_first(self.first_inf, coords)

        finite_values = flat[finite_mask]
        if finite_values.numel() > 0:
            batch_min = float(finite_values.min().item())
            batch_max = float(finite_values.max().item())
            self.finite_min = batch_min if self.finite_min is None else min(self.finite_min, batch_min)
            self.finite_max = batch_max if self.finite_max is None else max(self.finite_max, batch_max)

            masked = flat.masked_fill(~finite_mask, -torch.inf).reshape(-1)
            top_count = min(self.top_k, masked.numel())
            if top_count > 0:
                values, indices = torch.topk(masked, k=top_count)
                voxels = flat.shape[1]
                for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
                    if value == float("-inf"):
                        continue
                    row, col = divmod(int(index), voxels)
                    item = {
                        "value": float(value),
                        "event": global_event_offset + row,
                        "cell": col,
                    }
                    heap_item = (item["value"], item["event"], item["cell"], item)
                    if len(self.top_finite) < self.top_k:
                        heappush(self.top_finite, heap_item)
                    elif heap_item > self.top_finite[0]:
                        heappushpop(self.top_finite, heap_item)

    def summary(self):
        return {
            "tensor_count": self.tensor_count,
            "event_count": self.event_count,
            "value_count": self.value_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "events_with_nan": self.events_with_nan,
            "events_with_inf": self.events_with_inf,
            "finite_min": self.finite_min,
            "finite_max": self.finite_max,
            "first_nan": self.first_nan,
            "first_inf": self.first_inf,
            "top_finite": [
                item
                for _, _, _, item in sorted(self.top_finite, reverse=True)
            ],
        }


def default_condition_file(cfg, cli_path: Path | None) -> Path:
    if cli_path is not None:
        return cli_path
    test_conditions = cfg.train.get("test_conditions")
    if not test_conditions:
        raise ValueError("No train.test_conditions found in the saved config; pass --condition-file explicitly.")
    return Path(test_conditions[0][4])


def build_dataset(cfg, condition_file: Path, num_showers: int):
    data_cfg = cfg.data.valid if "valid" in cfg.data else cfg.data.train
    return CaloShowerDataset(
        files=[str(condition_file)],
        use_cond_info=bool(data_cfg.get("use_cond_info", True)),
        max_num_showers=num_showers,
        need_geo_condn=bool(cfg.train.get("need_geo_condn", False)),
        train_on=cfg.train.get("train_on"),
        is_ccd=bool(data_cfg.get("is_ccd", False)),
        ccd_geometry=data_cfg.get("ccd_geometry"),
    )


def main():
    args = parse_args()
    set_seed(args.seed, all_gpus=True)

    model, cfg = load_model(args.model_path)
    preprocessor = CaloShowerPreprocessor(**cfg.preprocessing)
    condition_file = default_condition_file(cfg, args.condition_file).resolve()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    dataset = build_dataset(cfg, condition_file, args.num_showers)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    sampling_cfg = OmegaConf.to_container(cfg.sampling, resolve=True)
    if args.sampling_steps is not None:
        sampling_cfg["steps"] = args.sampling_steps
    if args.sampling_solver is not None:
        sampling_cfg["solver"] = args.sampling_solver

    stages = {
        "sample_output": StageStats("sample_output", args.top_k),
        "after_standardize_inverse": StageStats("after_standardize_inverse", args.top_k),
        "after_log_inverse": StageStats("after_log_inverse", args.top_k),
    }
    class_to_stage = {
        "Standarize": "after_standardize_inverse",
        "LogTransform": "after_log_inverse",
    }

    seen_events = 0
    with torch.inference_mode():
        for _, conditions in dataloader:
            batch_size = int(conditions[0].shape[0])
            _, transformed_conditions = preprocessor.transform(conditions=conditions)
            transformed_conditions = to_device(transformed_conditions, device)

            sample_output = model.sample(
                conditions=transformed_conditions,
                progress=False,
                **sampling_cfg,
            ).squeeze(1)
            stages["sample_output"].update(sample_output, seen_events)

            def trace_fn(class_name: str, tensor: torch.Tensor):
                stage_name = class_to_stage.get(class_name)
                if stage_name is not None:
                    stages[stage_name].update(tensor, seen_events)

            preprocessor.inverse_transform(
                showers=sample_output,
                conditions=transformed_conditions,
                trace_fn=trace_fn,
            )
            seen_events += batch_size

    summary = {
        "model_path": str(args.model_path.resolve()),
        "condition_file": str(condition_file),
        "num_showers": args.num_showers,
        "batch_size": args.batch_size,
        "device": str(device),
        "sampling": sampling_cfg,
        "stages": {name: stats.summary() for name, stats in stages.items()},
    }

    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\njson_written: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
