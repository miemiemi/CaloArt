"""
EDM sampling solvers.
"""

import numpy as np
import torch
from scipy import integrate
from tqdm import trange

from src.utils import append_dims


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


@torch.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    callback=None,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=not progress):
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x


@torch.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    callback=None,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=not progress):
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            x = x + d * dt
        else:
            x_next = x + d * dt
            denoised_next = denoiser(x_next, sigmas[i + 1] * s_in)
            d_next = to_d(x_next, sigmas[i + 1], denoised_next)
            x = x + 0.5 * (d + d_next) * dt
    return x


def linear_multistep_coeff(t, r, n, j):
    if r - 1 > n:
        raise ValueError(f"Order {r} too high for step {n}")

    def l(tau):
        prod = 1.0
        for i in range(r):
            if i != j:
                prod *= (tau - t[n - i]) / (t[n - j] - t[n - i])
        return prod

    return integrate.quad(l, t[n], t[n + 1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_linear_multistep(denoiser, x, sigmas, callback=None, progress=False, order=4):
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    derivatives = []
    for i in trange(len(sigmas) - 1, disable=not progress):
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        derivatives.append(d)
        if len(derivatives) > order:
            derivatives.pop(0)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
        cur_order = min(i + 1, order)
        coeffs = [linear_multistep_coeff(sigmas_cpu, cur_order, i, j) for j in range(cur_order)]
        x = x + sum(coeff * d_item for coeff, d_item in zip(coeffs, reversed(derivatives)))
    return x
