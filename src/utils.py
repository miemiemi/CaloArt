import importlib
import logging
import math
import os
import random
import sys
from functools import wraps
from typing import Union
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs
from omegaconf import OmegaConf

MIX_PRECISION_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)


def manual_cast(tensor, dtype):
    """
    Cast if autocast is not enabled.
    """
    if not torch.is_autocast_enabled():
        return tensor.type(dtype)
    return tensor

def str_to_dtype(dtype_str: str):
    return {
        'f16': torch.float16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'f32': torch.float32,
        'fp32': torch.float32,
        'float32': torch.float32,
    }[dtype_str]

def convert_module_to(l, dtype):
    """
    Convert primitive modules to the given dtype.
    """
    if isinstance(l, MIX_PRECISION_MODULES):
        for p in l.parameters():
            p.data = p.data.to(dtype)

def get_logger(name: str = None, zero_rank_only: bool = True):
    """Create a logger that writes to stdout"""
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    if name is None:
        name = __name__

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    if zero_rank_only and rank != 0:
        logger.addHandler(logging.NullHandler())  # no-op logger
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_accelerator(cpu: bool = False, mixed_precision: str = "none", single_core: bool = False) -> Accelerator:
    mp.set_start_method("spawn")

    if cpu and single_core:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

    # keep split_batches unchanged; resuming a run with different resources and split_batches=False can modify batches_per_epoch
    # Edit: split_batches won't make a difference if dispatch_batches is True
    # do not remove find_unused_parameters, it is necessary for DDP to work properly: https://github.com/pytorch/pytorch/issues/43259
    accelerator = Accelerator(
        cpu=cpu,
        mixed_precision=mixed_precision,
        dataloader_config=DataLoaderConfiguration(),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        ],
    )

    # if distributed, wait for all processes to join
    if accelerator.use_distributed:
        accelerator.wait_for_everyone()

    return accelerator


def set_seed(seed: int, deterministic: bool = False, all_gpus: bool = False):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if all_gpus:
        torch.cuda.manual_seed_all(seed)
    # can slow down training
    if deterministic:
        torch.use_deterministic_algorithms(True)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cycle(dataloader):
    while True:
        for sample in dataloader:
            yield sample


def identity(x, *args, **kwargs):
    return x


def to_device(x, device: str = "cuda"):
    if isinstance(x, (list, tuple)):
        return tuple(to_device(item, device) for item in x)
    elif isinstance(x, dict):
        return {key: to_device(value, device) for key, value in x.items()}
    else:
        return x.to(device)


def import_class_by_name(class_name: str):
    module_name, class_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_





def flatten_dict(nested_dict: dict, sep="."):
    """Flatten a nested dictionary into a single level dictionary."""

    def _flatten_dict(nested_dict, parent_key=""):
        items = []
        for key, value in nested_dict.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, dict):
                items.extend(_flatten_dict(value, new_key).items())
            else:
                items.append((new_key, value))
        return dict(items)

    return _flatten_dict(nested_dict)


def get_lrs(optimizer):
    return [param_group["lr"] for param_group in optimizer.param_groups]


def get_conditions_str(geometry, energy, phi, theta):
    return f"Geo_{geometry}_E_{energy}_Phi_{phi}_Theta_{theta}"


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def load_config(default_config_path: str):
    """
    Load config from default path, merge it with custom config (if provided) and CLI arguments.
    Then call the decorated function with the config as an only argument.
    """

    def _is_yaml_file(file_path: str) -> bool:
        """Check if the file is a YAML file."""
        return file_path.endswith(".yaml") or file_path.endswith(".yml")

    def decorator(func):
        @wraps(func)
        def wrapper():
            config = OmegaConf.load(default_config_path)

            args = sys.argv[1:]
            if len(args) > 0 and _is_yaml_file(args[0]):
                custom_config_path = args.pop(0)
                print(custom_config_path)
                custom_config = OmegaConf.load(custom_config_path)
                config = OmegaConf.merge(config, custom_config)

            if len(args) > 0:
                cli_config = OmegaConf.from_cli(args)
                config = OmegaConf.merge(config, cli_config)

            func(config)

        return wrapper

    return decorator


def unwrap_ddp(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model
