#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


def _disable_torch_compile() -> None:
    def identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            def decorator(inner_fn):
                return inner_fn

            return decorator
        return fn

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    torch.compile = identity_compile


_disable_torch_compile()

import hydra
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from src.models.factory import create_model_from_config


def _format_human(num: float | int) -> str:
    num = float(num)
    units = ["", "K", "M", "G", "T", "P"]
    for unit in units:
        if abs(num) < 1000.0 or unit == units[-1]:
            return f"{num:.3f}{unit}"
        num /= 1000.0
    return f"{num:.3f}"


def _move_to_device(obj: Any, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _repeat_example_input(example_input, batch_size: int):
    x, c, t = example_input
    x = x.repeat(batch_size, 1, 1, 1, 1)
    c = tuple(cond.repeat(batch_size, 1) for cond in c)
    t = t.repeat(batch_size)
    return (x, c, t)


def _to_ptflops_input(example_input):
    if isinstance(example_input, tuple):
        return tuple(example_input)
    return (example_input,)


def _unwrap_compiled_forward(module: torch.nn.Module) -> int:
    replaced = 0
    for submodule in module.modules():
        eager_forward = getattr(submodule, "_forward", None)
        if eager_forward is None:
            continue
        current_forward = getattr(submodule, "forward", None)
        if current_forward is eager_forward:
            continue
        submodule.forward = eager_forward
        replaced += 1
    return replaced


def _count_params(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _profile_with_deepspeed(
    module: torch.nn.Module,
    example_input,
    profile_step: int = 1,
    module_depth: int = -1,
    top_modules: int = 3,
    detailed: bool = False,
    report_output_file: str | None = None,
):
    from deepspeed.profiling.flops_profiler import FlopsProfiler

    profiler = FlopsProfiler(module)
    profiler.start_profile()
    with torch.no_grad():
        module(*example_input)
    profiler.stop_profile()

    total_flops = profiler.get_total_flops()
    total_macs = profiler.get_total_macs()
    total_params = profiler.get_total_params()

    profiler.print_model_profile(
        profile_step=profile_step,
        module_depth=module_depth,
        top_modules=top_modules,
        detailed=detailed,
        output_file=report_output_file,
    )

    return {
        "flops": int(total_flops),
        "macs": int(total_macs),
        "params": int(total_params),
    }


def _profile_with_ptflops(module: torch.nn.Module, example_input):
    from ptflops import get_model_complexity_info

    inputs = _to_ptflops_input(example_input)

    def input_constructor(_):
        return {"x": inputs[0], "c": inputs[1], "t": inputs[2]}

    macs, params = get_model_complexity_info(
        module,
        input_res=(1,),
        input_constructor=input_constructor,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=True,
        backend="aten",
    )

    return {
        "macs": int(macs),
        "flops_estimate": int(2 * macs),
        "params": int(params),
    }


def _load_cfg(config_name: str | None, config_path: str | None):
    if config_name is not None:
        config_dir = str((Path(__file__).resolve().parent.parent / "configs").resolve())
        with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
            return hydra.compose(config_name=config_name)
    if config_path is not None:
        return OmegaConf.load(config_path)
    raise ValueError("Either --config-name or --config-path must be provided.")


def _profile_one_batch(
    cfg,
    device: torch.device,
    batch_size: int,
    *,
    deepspeed_profile_step: int,
    deepspeed_module_depth: int,
    deepspeed_top_modules: int,
    deepspeed_detailed: bool,
    deepspeed_report_path: str | None,
):
    model = create_model_from_config(cfg.model, cfg.method)
    backbone = model.model
    backbone.eval()

    replaced = _unwrap_compiled_forward(backbone)
    backbone = backbone.to(device)

    example_input = _repeat_example_input(backbone.example_input, batch_size)
    example_input = _move_to_device(example_input, device)

    total_params, trainable_params = _count_params(backbone)

    result = {
        "batch_size": batch_size,
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "compiled_forward_wrappers_replaced": int(replaced),
        "torch_compile_disabled": True,
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        result["deepspeed"] = _profile_with_deepspeed(
            backbone,
            example_input,
            profile_step=deepspeed_profile_step,
            module_depth=deepspeed_module_depth,
            top_modules=deepspeed_top_modules,
            detailed=deepspeed_detailed,
            report_output_file=deepspeed_report_path,
        )
    except Exception as exc:
        result["deepspeed_error"] = f"{type(exc).__name__}: {exc}"
    if device.type == "cuda":
        result["cuda_peak_mem_bytes_after_deepspeed"] = int(torch.cuda.max_memory_allocated(device))

    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        result["ptflops_aten"] = _profile_with_ptflops(backbone, example_input)
    except Exception as exc:
        result["ptflops_error"] = f"{type(exc).__name__}: {exc}"
    if device.type == "cuda":
        result["cuda_peak_mem_bytes_after_ptflops"] = int(torch.cuda.max_memory_allocated(device))

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Profile model FLOPs with torch.compile disabled.")
    parser.add_argument("--config-name", type=str, default=None, help="Hydra config name under configs/.")
    parser.add_argument("--config-path", type=str, default=None, help="Path to a resolved config.yaml file.")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to profile.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("PROFILE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        help="Profiling device.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save JSON results.")
    parser.add_argument(
        "--deepspeed-profile-step",
        type=int,
        default=int(os.environ.get("PROFILE_DEEPSPEED_PROFILE_STEP", "1")),
        help="DeepSpeed profile step value used in the printed report.",
    )
    parser.add_argument(
        "--deepspeed-module-depth",
        type=int,
        default=int(os.environ.get("PROFILE_DEEPSPEED_MODULE_DEPTH", "-1")),
        help="DeepSpeed aggregation depth.",
    )
    parser.add_argument(
        "--deepspeed-top-modules",
        type=int,
        default=int(os.environ.get("PROFILE_DEEPSPEED_TOP_MODULES", "3")),
        help="Number of top modules to show in DeepSpeed summaries.",
    )
    parser.add_argument(
        "--deepspeed-detailed",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("PROFILE_DEEPSPEED_DETAILED", "1") != "0",
        help="Whether to print the full DeepSpeed detailed module report.",
    )
    parser.add_argument(
        "--deepspeed-report-path",
        type=str,
        default=os.environ.get("PROFILE_DEEPSPEED_REPORT_PATH"),
        help="Optional file path for DeepSpeed's text report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = _load_cfg(args.config_name, args.config_path)
    device = torch.device(args.device)

    meta = {
        "device": str(device),
        "torch_compile_disabled": True,
        "config_name": args.config_name,
        "config_path": args.config_path,
        "batch_sizes": list(args.batch_sizes),
        "results": [],
    }

    for batch_size in args.batch_sizes:
        result = _profile_one_batch(
            cfg,
            device,
            batch_size,
            deepspeed_profile_step=args.deepspeed_profile_step,
            deepspeed_module_depth=args.deepspeed_module_depth,
            deepspeed_top_modules=args.deepspeed_top_modules,
            deepspeed_detailed=args.deepspeed_detailed,
            deepspeed_report_path=args.deepspeed_report_path,
        )
        meta["results"].append(result)
        print(json.dumps(result, sort_keys=True))
        if "deepspeed" in result:
            print(
                "DeepSpeed: "
                f"batch={batch_size}, "
                f"MACs={result['deepspeed']['macs']} ({_format_human(result['deepspeed']['macs'])}), "
                f"FLOPs={result['deepspeed']['flops']} ({_format_human(result['deepspeed']['flops'])})"
            )
        if "ptflops_aten" in result:
            print(
                "ptflops(aten): "
                f"batch={batch_size}, "
                f"MACs={result['ptflops_aten']['macs']} ({_format_human(result['ptflops_aten']['macs'])}), "
                f"FLOPs~={result['ptflops_aten']['flops_estimate']} "
                f"({_format_human(result['ptflops_aten']['flops_estimate'])})"
            )

    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(meta, indent=2))
        print(f"Saved JSON to {output_json}")


if __name__ == "__main__":
    sys.exit(main())
