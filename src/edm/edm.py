"""
Implementation of Elucidating the Design Space of Diffusion-Based Generative Models (EDM).
Source: Karras et al., https://arxiv.org/abs/2206.00364
Ported from diffusion4sim, adapted for CaloFlow.
"""

import torch

from src.edm.sampling import sample_euler, sample_heun, sample_linear_multistep
from src.method_base import MethodBase
from src.utils import append_dims, append_zero, mean_flat


class EDM(MethodBase):
    def __init__(
        self,
        model,
        num_timesteps=32,
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        rho=7,
        P_mean=-1.2,
        P_std=1.2,
    ):
        super().__init__()
        self.model = model
        assert model.in_channels == model.out_channels, "input and output channels must match"
        self.input_size = (model.in_channels, *model.input_size)

        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def denoise(self, x_t, x_cond, sigma):
        c_skip, c_out, c_in = [append_dims(c, x_t.ndim) for c in self.get_scalings(sigma)]
        c_noise = 0.25 * torch.log(sigma + 1e-44) * 1000
        model_output = self.model(c_in * x_t, x_cond, c_noise)
        return c_skip * x_t + c_out * model_output

    def noise_distribution(self, batch_size):
        log_sigmas = self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)
        return torch.exp(log_sigmas)

    def loss_weighting(self, sigma):
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    def forward(self, x_0, x_cond):
        batch_size = x_0.shape[0]
        sigmas = self.noise_distribution(batch_size)
        noise = torch.randn_like(x_0)
        x_t = x_0 + noise * append_dims(sigmas, x_0.ndim)
        denoised = self.denoise(x_t, x_cond, sigmas)
        weights = append_dims(self.loss_weighting(sigmas), x_0.ndim)
        losses = mean_flat(weights * (denoised - x_0) ** 2)
        return losses.mean()

    def get_timesteps(self, steps=None):
        num_steps = steps if steps is not None else self.num_timesteps
        i = torch.arange(num_steps, device=self.device)
        sigmas = (
            self.sigma_max ** (1 / self.rho)
            + i / (num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        return append_zero(sigmas)

    @torch.inference_mode()
    def sample(
        self,
        conditions,
        steps=None,
        solver="heun",
        solver_args={},
        clip_denoised=False,
        progress=False,
    ):
        num_samples = len(conditions[0])
        assert all(num_samples == len(c) for c in conditions), "all conditions must have the same batch size"
        x_shape = (num_samples, *self.input_size)

        sigmas = self.get_timesteps(steps)
        x_t = torch.randn(x_shape, device=self.device) * self.sigma_max

        sample_fn = {
            "heun": sample_heun,
            "euler": sample_euler,
            "linear_multistep": sample_linear_multistep,
        }[solver]

        def denoiser(x_cur, sigma):
            denoised = self.denoise(x_cur, conditions, sigma)
            if clip_denoised:
                denoised = denoised.clamp(-1, 1)
            return denoised

        return sample_fn(denoiser, x_t, sigmas, progress=progress, **solver_args)
