#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

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
    detailed: bool = True,
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


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="experiment/CaloChallenge/flow_ccd2_pred_v_all_lightingdit_freatures_c72_36_36_pixart_sharemod_heun32_latest",
)
def main(cfg: DictConfig):
    device_name = os.environ.get("PROFILE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    output_json_env = os.environ.get("PROFILE_OUTPUT_JSON")
    output_json = Path(output_json_env) if output_json_env else None
    deepspeed_report_path = os.environ.get("PROFILE_DEEPSPEED_REPORT_PATH")
    deepspeed_profile_step = int(os.environ.get("PROFILE_DEEPSPEED_PROFILE_STEP", "1"))
    deepspeed_module_depth = int(os.environ.get("PROFILE_DEEPSPEED_MODULE_DEPTH", "-1"))
    deepspeed_top_modules = int(os.environ.get("PROFILE_DEEPSPEED_TOP_MODULES", "3"))
    deepspeed_detailed = os.environ.get("PROFILE_DEEPSPEED_DETAILED", "1") != "0"

    model = create_model_from_config(cfg.model, cfg.method)
    backbone = model.model
    backbone.eval()

    # Profilers generally behave more predictably on eager forward paths.
    replaced = _unwrap_compiled_forward(backbone)

    device = torch.device(device_name)
    backbone = backbone.to(device)
    example_input = _move_to_device(backbone.example_input, device)

    total_params, trainable_params = _count_params(backbone)

    deepspeed_error = None
    ptflops_error = None
    try:
        deepspeed_results = _profile_with_deepspeed(
            backbone,
            example_input,
            profile_step=deepspeed_profile_step,
            module_depth=deepspeed_module_depth,
            top_modules=deepspeed_top_modules,
            detailed=deepspeed_detailed,
            report_output_file=deepspeed_report_path,
        )
    except Exception as exc:
        deepspeed_results = None
        deepspeed_error = f"{type(exc).__name__}: {exc}"

    try:
        ptflops_results = _profile_with_ptflops(backbone, example_input)
    except Exception as exc:
        ptflops_results = None
        ptflops_error = f"{type(exc).__name__}: {exc}"

    results = {
        "config_name": "resolved_from_hydra",
        "device": str(device),
        "compiled_forward_wrappers_replaced": replaced,
        "parameter_count": {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params,
        },
        "deepspeed": deepspeed_results,
        "deepspeed_error": deepspeed_error,
        "ptflops_aten": ptflops_results,
        "ptflops_error": ptflops_error,
    }

    print("Profile target: backbone model")
    print(f"Device: {device}")
    print(f"Compiled forward wrappers replaced: {replaced}")
    print(
        "Params: "
        f"total={total_params} ({_format_human(total_params)}), "
        f"trainable={trainable_params} ({_format_human(trainable_params)}), "
        f"non_trainable={total_params - trainable_params}"
    )
    if deepspeed_results is not None:
        print(
            "DeepSpeed: "
            f"MACs={deepspeed_results['macs']} ({_format_human(deepspeed_results['macs'])}), "
            f"FLOPs={deepspeed_results['flops']} ({_format_human(deepspeed_results['flops'])}), "
            f"Params={deepspeed_results['params']} ({_format_human(deepspeed_results['params'])})"
        )
    else:
        print(f"DeepSpeed failed: {deepspeed_error}")

    if ptflops_results is not None:
        print(
            "ptflops(aten): "
            f"MACs={ptflops_results['macs']} ({_format_human(ptflops_results['macs'])}), "
            f"FLOPs~={ptflops_results['flops_estimate']} ({_format_human(ptflops_results['flops_estimate'])}), "
            f"Params={ptflops_results['params']} ({_format_human(ptflops_results['params'])})"
        )
    else:
        print(f"ptflops(aten) failed: {ptflops_error}")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, indent=2))
        print(f"Saved JSON to {output_json}")


if __name__ == "__main__":
    sys.exit(main())
