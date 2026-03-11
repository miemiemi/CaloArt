import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_inverse_sqrt_schedule,
    get_linear_schedule_with_warmup,
)


def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int = None,
    num_stable_steps: int = None,
    num_decay_steps: int = None,
    num_max_steps: int = None,
    min_lr_ratio: float = 0,
    last_epoch: int = -1,
    decay_type: str = "cosine",
):
    """
    Warmup-Stable-Decay schedule. For more details, see https://arxiv.org/pdf/2405.18392

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_stable_steps (`int`):
            The number of steps for the stable phase.
        num_decay_steps (`int`):
            The number of steps for the cosine annealing phase.
        min_lr_ratio (`float`, *optional*, defaults to 0):
            The minimum learning rate as a ratio of the initial learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        decay_type (`str`, *optional*, defaults to "cosine"):
            The type of decay to use. Can be "cosine", "linear", "sqrt", or "square".
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if num_max_steps:
        num_warmup_steps = 0.1 * num_max_steps
        num_stable_steps = 0.5 * num_max_steps
        num_decay_steps = num_max_steps - num_warmup_steps - num_stable_steps

    def _get_wsd_scheduler_lambda(
        current_step: int,
        *,
        num_warmup_steps: int,
        num_stable_steps: int,
        num_decay_steps: int,
        min_lr_ratio: float,
        decay_type: str,
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
            progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
            if decay_type == "cosine":
                value = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            elif decay_type == "linear":
                value = 1.0 - progress
            elif decay_type == "sqrt":
                value = 1.0 - math.sqrt(progress)
            elif decay_type == "square":
                value = 1.0 - progress**2
            else:
                raise ValueError(f"Unknown decay type: {decay_type}")
            return (1.0 - min_lr_ratio) * value + min_lr_ratio
        return min_lr_ratio

    lr_lambda = partial(
        _get_wsd_scheduler_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=min_lr_ratio,
        decay_type=decay_type,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr_scheduler(optimizer, lr_scheduler_type, **lr_scheduler_kwargs):
    if lr_scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, **lr_scheduler_kwargs)
    elif lr_scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, **lr_scheduler_kwargs)
    elif lr_scheduler_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, **lr_scheduler_kwargs)
    elif lr_scheduler_type == "sqrt":
        return get_inverse_sqrt_schedule(optimizer, **lr_scheduler_kwargs)
    elif lr_scheduler_type == "wsd":
        return get_wsd_schedule(optimizer, **lr_scheduler_kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {lr_scheduler_type}")
