#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import rootutils
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


if "--disable-compile" in sys.argv:
    _disable_torch_compile()


rootutils.setup_root(__file__, pythonpath=True)

from src.models.factory import create_model_from_config


DEFAULT_CONFIG_PATHS = [
    "/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/CaloArt_paper/flow_ccd2_pred_v_all_calodit_h144_heun32_fpd_latest/checkpoints/checkpoint_best/config.yaml",
    "/aifs/user/home/huangzhengkun/work/repositories/CaloFlow/experiments/CaloArt_paper/flow_ccd2_pred_v_all_lightingdit_freatures_c72_36_36_ape_rope_nosharemod_classicdit_heun32_fpd_latest/checkpoints/checkpoint_best/config.yaml",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark steady-state GPU forward latency for CaloFlow backbones."
    )
    parser.add_argument(
        "--config-paths",
        nargs="+",
        default=DEFAULT_CONFIG_PATHS,
        help="Resolved experiment config.yaml paths to benchmark.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 128, 256],
        help="Batch sizes to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations before timing.")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Benchmark device. CUDA is expected for meaningful latency numbers.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["none", "fp16", "bf16"],
        default="none",
        help="Optional autocast dtype for the forward pass.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional JSON output path for benchmark results.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable @torch.compile decorators before model import.",
    )
    return parser.parse_args()


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


def _sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp_dtype]
    return torch.autocast(device_type="cuda", dtype=dtype)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return sorted_values[idx]


def _benchmark_one(module: torch.nn.Module, example_input, device: torch.device, warmup: int, iters: int, amp_dtype: str):
    _sync_if_needed(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        with _autocast_context(device, amp_dtype):
            for _ in range(warmup):
                module(*example_input)
            _sync_if_needed(device)

            start_events = []
            end_events = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                module(*example_input)
                end.record()
                start_events.append(start)
                end_events.append(end)

            _sync_if_needed(device)

    latencies_ms = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
    latencies_sorted = sorted(latencies_ms)
    throughput = [example_input[0].shape[0] * 1000.0 / latency for latency in latencies_ms if latency > 0]

    result = {
        "mean_ms": mean(latencies_ms),
        "median_ms": median(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "p90_ms": _percentile(latencies_sorted, 0.90),
        "p95_ms": _percentile(latencies_sorted, 0.95),
        "throughput_samples_per_s_mean": mean(throughput),
    }
    if device.type == "cuda":
        result["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return result


def _load_backbone(config_path: str, device: torch.device):
    cfg = OmegaConf.load(config_path)
    model = create_model_from_config(cfg.model, cfg.method)
    backbone = model.model.eval().to(device)
    return cfg, backbone


def _benchmark_config(config_path: str, batch_sizes: list[int], device: torch.device, warmup: int, iters: int, amp_dtype: str):
    results = []
    for batch_size in batch_sizes:
        cfg, backbone = _load_backbone(config_path, device)
        example_input = _repeat_example_input(backbone.example_input, batch_size)
        example_input = _move_to_device(example_input, device)
        metrics = _benchmark_one(backbone, example_input, device, warmup, iters, amp_dtype)
        results.append(
            {
                "batch_size": batch_size,
                "metrics": metrics,
                "model_target": cfg.model.architecture.target,
                "run_name": cfg.experiment.run_name,
            }
        )
    return results


def main():
    args = parse_args()
    device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError("This benchmark script is intended for CUDA devices.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    payload = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "disable_compile": bool(args.disable_compile),
        "amp_dtype": args.amp_dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": [],
    }

    for config_path in args.config_paths:
        config_path = str(Path(config_path).resolve())
        config_results = _benchmark_config(
            config_path=config_path,
            batch_sizes=args.batch_sizes,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            amp_dtype=args.amp_dtype,
        )
        payload["results"].append(
            {
                "config_path": config_path,
                "benchmarks": config_results,
            }
        )

        print(f"\nconfig={config_path}")
        for item in config_results:
            metrics = item["metrics"]
            print(
                f"batch={item['batch_size']} "
                f"mean_ms={metrics['mean_ms']:.3f} "
                f"median_ms={metrics['median_ms']:.3f} "
                f"p95_ms={metrics['p95_ms']:.3f} "
                f"throughput={metrics['throughput_samples_per_s_mean']:.2f} samples/s "
                f"peak_mem_mb={metrics['peak_mem_mb']:.1f}"
            )

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON to {output_path}")


if __name__ == "__main__":
    main()
