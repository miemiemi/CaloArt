"""
ODE samplers for flow matching.
Solve the ODE: dx/dt = v(x, t) from t=0 (noise) to t=1 (data).
"""

import torch
from tqdm import trange


@torch.no_grad()
def euler_ode_sample(model, x, conditions, steps=50, progress=False):
    """Euler method ODE solver for flow matching.
    
    Args:
        model: backbone network that predicts velocity v(z_t, c, t)
        x: initial noise x_0 ~ N(0, I), shape (B, C, R, PHI, Z)
        conditions: tuple of condition tensors
        steps: number of ODE integration steps
        progress: show progress bar
    
    Returns:
        x_1: generated samples (approximation to clean data)
    """
    dt = 1.0 / steps
    for i in trange(steps, disable=not progress):
        t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
        v = model(x, conditions, t)
        x = x + v * dt
    return x


@torch.no_grad()
def heun_ode_sample(model, x, conditions, steps=50, progress=False):
    """Heun's method (2nd-order) ODE solver for flow matching.
    
    Uses predictor-corrector approach:
        1. Predict: x_next = x + v(x, t) * dt  (Euler step)
        2. Correct: x_next = x + 0.5 * (v(x, t) + v(x_next, t+dt)) * dt
    """
    dt = 1.0 / steps
    for i in trange(steps, disable=not progress):
        t_cur = i * dt
        t_next = (i + 1) * dt

        t_cur_batch = torch.full((x.shape[0],), t_cur, device=x.device, dtype=x.dtype)
        v1 = model(x, conditions, t_cur_batch)

        # Euler predictor
        x_pred = x + v1 * dt

        if t_next >= 1.0:
            # Last step: just use Euler
            x = x_pred
        else:
            # Corrector
            t_next_batch = torch.full((x.shape[0],), t_next, device=x.device, dtype=x.dtype)
            v2 = model(x_pred, conditions, t_next_batch)
            x = x + 0.5 * (v1 + v2) * dt

    return x


@torch.no_grad()
def midpoint_ode_sample(model, x, conditions, steps=50, progress=False):
    """Midpoint method (2nd-order) ODE solver for flow matching.
    
    x_next = x + dt * v(x + 0.5*dt*v(x, t), t + 0.5*dt)
    """
    dt = 1.0 / steps
    for i in trange(steps, disable=not progress):
        t_cur = i * dt
        t_mid = t_cur + 0.5 * dt

        t_cur_batch = torch.full((x.shape[0],), t_cur, device=x.device, dtype=x.dtype)
        v1 = model(x, conditions, t_cur_batch)

        # Midpoint evaluation
        x_mid = x + 0.5 * dt * v1
        t_mid_batch = torch.full((x.shape[0],), t_mid, device=x.device, dtype=x.dtype)
        v_mid = model(x_mid, conditions, t_mid_batch)

        x = x + v_mid * dt

    return x
