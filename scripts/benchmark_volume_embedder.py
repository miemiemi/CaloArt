import argparse
import importlib
from statistics import mean

import rootutils
import torch

rootutils.setup_root(__file__, pythonpath=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark VolumeEmbedder conv vs non-conv on GPU.")
    parser.add_argument(
        "--modules",
        nargs="+",
        default=[
            "src.models.layers:VolumeEmbedder",
            "src.models.layers_3drope:VolumeEmbedder",
        ],
        help="Module paths in the form package.module:ClassName",
    )
    parser.add_argument("--input-size", nargs=3, type=int, default=[9, 16, 45])
    parser.add_argument("--patch-size", nargs=3, type=int, default=[3, 2, 3])
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=144)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 16, 64])
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_class(spec: str):
    module_name, class_name = spec.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def correctness_check(embedder_cls, device):
    input_size = (5, 6, 7)
    patch_size = (2, 3, 4)
    in_channels = 3
    out_channels = 11

    torch.manual_seed(0)
    conv = embedder_cls(input_size, patch_size, in_channels, out_channels, use_conv=True).to(device)
    linear = embedder_cls(input_size, patch_size, in_channels, out_channels, use_conv=False).to(device)

    with torch.no_grad():
        linear.proj.weight.copy_(conv.proj.weight.reshape(out_channels, -1))
        linear.proj.bias.copy_(conv.proj.bias)

    x_conv = torch.randn(2, in_channels, *input_size, device=device, requires_grad=True)
    x_linear = x_conv.detach().clone().requires_grad_(True)

    y_conv = conv(x_conv)
    y_linear = linear(x_linear)
    grad_out = torch.randn_like(y_conv)

    y_conv.backward(grad_out)
    y_linear.backward(grad_out)

    return {
        "output_max_abs_diff": (y_conv - y_linear).abs().max().item(),
        "input_grad_max_abs_diff": (x_conv.grad - x_linear.grad).abs().max().item(),
    }


def benchmark_once(module, x, backward: bool):
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()
        start.record()
        if backward:
            loss = module(x).square().mean()
            loss.backward()
        else:
            module(x)
        end.record()
        sync()
        elapsed_ms = start.elapsed_time(end)
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        raise RuntimeError("This benchmark is intended to run on GPU.")

    module.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    return elapsed_ms, peak_mem_mb


def benchmark(embedder_cls, batch_size, input_size, patch_size, in_channels, out_channels, warmup, iters, device):
    results = {}
    for use_conv in (False, True):
        module = embedder_cls(input_size, patch_size, in_channels, out_channels, use_conv=use_conv).to(device)

        x = torch.randn(batch_size, in_channels, *input_size, device=device)
        for _ in range(warmup):
            module(x)
        sync()

        forward_times = []
        forward_mems = []
        for _ in range(iters):
            elapsed_ms, peak_mem_mb = benchmark_once(module, x, backward=False)
            forward_times.append(elapsed_ms)
            forward_mems.append(peak_mem_mb)

        x = torch.randn(batch_size, in_channels, *input_size, device=device, requires_grad=True)
        for _ in range(warmup):
            loss = module(x).square().mean()
            loss.backward()
            module.zero_grad(set_to_none=True)
            x.grad = None
        sync()

        backward_times = []
        backward_mems = []
        for _ in range(iters):
            elapsed_ms, peak_mem_mb = benchmark_once(module, x, backward=True)
            backward_times.append(elapsed_ms)
            backward_mems.append(peak_mem_mb)

        results[use_conv] = {
            "forward_ms": mean(forward_times),
            "forward_peak_mem_mb": mean(forward_mems),
            "forward_backward_ms": mean(backward_times),
            "forward_backward_peak_mem_mb": mean(backward_mems),
        }
    return results


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    print(f"device={torch.cuda.get_device_name(device)}")
    print(
        "config:",
        {
            "input_size": tuple(args.input_size),
            "patch_size": tuple(args.patch_size),
            "in_channels": args.in_channels,
            "out_channels": args.out_channels,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "iters": args.iters,
        },
    )

    for spec in args.modules:
        embedder_cls = load_class(spec)
        print(f"\nmodule={spec}")
        correctness = correctness_check(embedder_cls, device)
        print("correctness:", correctness)

        for batch_size in args.batch_sizes:
            results = benchmark(
                embedder_cls=embedder_cls,
                batch_size=batch_size,
                input_size=tuple(args.input_size),
                patch_size=tuple(args.patch_size),
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            linear = results[False]
            conv = results[True]
            print(
                f"batch_size={batch_size} "
                f"linear.forward_ms={linear['forward_ms']:.4f} "
                f"conv.forward_ms={conv['forward_ms']:.4f} "
                f"conv_forward_speedup={linear['forward_ms'] / conv['forward_ms']:.4f}x"
            )
            print(
                f"batch_size={batch_size} "
                f"linear.forward_backward_ms={linear['forward_backward_ms']:.4f} "
                f"conv.forward_backward_ms={conv['forward_backward_ms']:.4f} "
                f"conv_forward_backward_speedup={linear['forward_backward_ms'] / conv['forward_backward_ms']:.4f}x"
            )
            print(
                f"batch_size={batch_size} "
                f"linear.forward_peak_mem_mb={linear['forward_peak_mem_mb']:.2f} "
                f"conv.forward_peak_mem_mb={conv['forward_peak_mem_mb']:.2f} "
                f"linear.forward_backward_peak_mem_mb={linear['forward_backward_peak_mem_mb']:.2f} "
                f"conv.forward_backward_peak_mem_mb={conv['forward_backward_peak_mem_mb']:.2f}"
            )


if __name__ == "__main__":
    main()
