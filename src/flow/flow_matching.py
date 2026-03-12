"""
Flow Matching for 3D calorimeter shower generation.

Supports a 3×3 matrix of (predict_mode × loss_target):
    predict_mode: what the model outputs  — 'pred_v', 'pred_x1', 'pred_eps'
    loss_target:  what space to compute loss in — 'v', 'x1', 'eps'

Linear interpolation path: z_t = (1-t)·x₀ + t·x₁
    where x₀ = noise (ε), x₁ = data, v = x₁ - x₀

Conversions (given z_t and t):
    v  → x₁ = z_t + (1-t)·v       v  → ε  = z_t - t·v
    x₁ → v  = (x₁ - z_t)/(1-t)    x₁ → ε  = (z_t - t·x₁)/(1-t)
    ε  → v  = (z_t - ε)/t          ε  → x₁ = (z_t - (1-t)·ε)/t
"""

import torch
import torch.nn as nn

from src.flow.sampler import euler_ode_sample, heun_ode_sample, midpoint_ode_sample
from src.method_base import MethodBase
from src.utils import mean_flat


PREDICT_MODES = ("pred_v", "pred_x1", "pred_eps")
LOSS_TARGETS = ("v", "x1", "eps")


class FlowMatching(MethodBase):
    """Flow Matching wrapper for a backbone network (e.g. CaloLightningDiT).

    Args:
        model:              backbone nn.Module with forward(x, conditions, t)
        predict_mode:       what the model predicts — 'pred_v' | 'pred_x1' | 'pred_eps'
        loss_target:        which space to compute MSE loss in — 'v' | 'x1' | 'eps'
        time_sampler:       how to sample t during training — 'uniform' | 'logit_normal'
        logit_normal_mean:  mean for logit-normal time sampling
        logit_normal_std:   std  for logit-normal time sampling
        noise_scale:        scale of initial noise x₀
        t_eps:              small epsilon to avoid t=0 or t=1 singularities
        num_sample_steps:   default ODE steps for sampling
        solver:             default ODE solver — 'euler' | 'heun' | 'midpoint'
    """

    def __init__(
        self,
        model,
        predict_mode="pred_v",
        loss_target="v",
        time_sampler="uniform",
        logit_normal_mean=0.0,
        logit_normal_std=1.0,
        noise_scale=1.0,
        t_eps=1e-5,
        num_sample_steps=50,
        solver="euler",
    ):
        super().__init__()
        assert predict_mode in PREDICT_MODES, f"predict_mode must be one of {PREDICT_MODES}, got '{predict_mode}'"
        assert loss_target in LOSS_TARGETS, f"loss_target must be one of {LOSS_TARGETS}, got '{loss_target}'"

        self.model = model
        assert model.in_channels == model.out_channels, "input and output channels must match"
        self.input_size = (model.in_channels, *model.input_size)

        self.predict_mode = predict_mode
        self.loss_target = loss_target
        self.time_sampler = time_sampler
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.num_sample_steps = num_sample_steps
        self.solver = solver

    # ------------------------------------------------------------------ #
    #                         Time sampling                                #
    # ------------------------------------------------------------------ #

    def sample_time(self, batch_size, device):
        """Sample timesteps t ∈ [t_eps, 1 - t_eps]."""
        if self.time_sampler == "uniform":
            t = torch.rand(batch_size, device=device) * (1.0 - 2 * self.t_eps) + self.t_eps
        elif self.time_sampler == "logit_normal":
            # t = sigmoid(N(mean, std^2))  — biases sampling towards the middle
            u = torch.randn(batch_size, device=device) * self.logit_normal_std + self.logit_normal_mean
            t = torch.sigmoid(u)
            t = t.clamp(self.t_eps, 1.0 - self.t_eps)
        else:
            raise ValueError(f"Unknown time_sampler: {self.time_sampler}")
        return t

    # ------------------------------------------------------------------ #
    #                   Conversion between representations                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _expand_t(t, ndim):
        """Expand t of shape (B,) to (B, 1, 1, ..., 1) for broadcasting."""
        return t.view(-1, *([1] * (ndim - 1)))

    def _convert(self, pred, z_t, t, from_type, to_type):
        """Convert a prediction from one representation to another.

        Args:
            pred:       model prediction, shape (B, C, R, PHI, Z)
            z_t:        noisy input, shape (B, C, R, PHI, Z)
            t:          timestep (B,)
            from_type:  source representation ('v', 'x1', 'eps')
            to_type:    target representation ('v', 'x1', 'eps')

        Returns:
            converted prediction in to_type space
        """
        if from_type == to_type:
            return pred

        te = self._expand_t(t, pred.ndim)
        one_minus_te = 1.0 - te

        # Step 1: convert to v (canonical intermediate)
        if from_type == "v":
            v = pred
        elif from_type == "x1":
            v = (pred - z_t) / one_minus_te.clamp(min=self.t_eps)
        elif from_type == "eps":
            v = (z_t - pred) / te.clamp(min=self.t_eps)
        else:
            raise ValueError(f"Unknown from_type: {from_type}")

        # Step 2: convert from v to target
        if to_type == "v":
            return v
        elif to_type == "x1":
            return z_t + one_minus_te * v
        elif to_type == "eps":
            return z_t - te * v
        else:
            raise ValueError(f"Unknown to_type: {to_type}")

    def _get_target(self, x_0, x_1, v, target_type):
        """Get the ground-truth target in the specified representation."""
        if target_type == "v":
            return v
        elif target_type == "x1":
            return x_1
        elif target_type == "eps":
            return x_0
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    # ------------------------------------------------------------------ #
    #                         Training (forward)                           #
    # ------------------------------------------------------------------ #

    def forward(self, x_1, x_cond):
        """Compute flow matching training loss.

        Args:
            x_1:    clean data (showers), shape (B, C, R, PHI, Z)
            x_cond: tuple of condition tensors

        Returns:
            scalar loss
        """
        batch_size = x_1.shape[0]

        # 1. Sample time
        t = self.sample_time(batch_size, device=x_1.device)
        te = self._expand_t(t, x_1.ndim)

        # 2. Sample noise x_0 ~ N(0, noise_scale^2 * I)
        x_0 = torch.randn_like(x_1) * self.noise_scale

        # 3. Linear interpolation: z_t = (1-t)·x₀ + t·x₁
        z_t = (1.0 - te) * x_0 + te * x_1

        # 4. Ground truth velocity
        v_target = x_1 - x_0

        # 5. Model prediction (interpretation depends on predict_mode)
        raw_pred = self.model(z_t, x_cond, t)

        # 6. Strip 'pred_' prefix to get the from_type key
        from_type = self.predict_mode.replace("pred_", "")  # 'v', 'x1', or 'eps'

        # 7. Convert prediction to loss_target space
        pred_in_loss_space = self._convert(raw_pred, z_t, t, from_type, self.loss_target)
        target_in_loss_space = self._get_target(x_0, x_1, v_target, self.loss_target)

        # 8. MSE loss
        losses = mean_flat((pred_in_loss_space - target_in_loss_space) ** 2)
        return losses.mean()

    # ------------------------------------------------------------------ #
    #                        Sampling (inference)                           #
    # ------------------------------------------------------------------ #

    def _velocity_fn(self, x, conditions, t):
        """Compute velocity from model prediction, regardless of predict_mode.

        During sampling, we always need the velocity field to integrate the ODE.
        This function converts the raw model output back to velocity.
        """
        # Keep sampling-time model inputs aligned with the training support.
        t = t.clamp(self.t_eps, 1.0 - self.t_eps)
        raw_pred = self.model(x, conditions, t)
        from_type = self.predict_mode.replace("pred_", "")
        if from_type == "v":
            return raw_pred
        else:
            return self._convert(raw_pred, x, t, from_type, "v")

    @torch.inference_mode()
    def sample(
        self,
        conditions,
        steps=None,
        solver=None,
        progress=False,
    ):
        """Generate samples by solving the ODE from t=0 (noise) to t=1 (data).

        Args:
            conditions: tuple of condition tensors
            steps:      number of ODE integration steps (default: self.num_sample_steps)
            solver:     'euler' | 'heun' | 'midpoint' (default: self.solver)
            progress:   show progress bar

        Returns:
            x_1: generated samples, shape (B, C, R, PHI, Z)
        """
        steps = steps or self.num_sample_steps
        solver = solver or self.solver

        num_samples = len(conditions[0])
        assert all(num_samples == len(c) for c in conditions), "all conditions must have the same batch size"
        x_shape = (num_samples, *self.input_size)

        # Initial noise
        x_0 = torch.randn(x_shape, device=self.device) * self.noise_scale

        # Select solver
        sample_fn = {
            "euler": euler_ode_sample,
            "heun": heun_ode_sample,
            "midpoint": midpoint_ode_sample,
        }[solver]

        # Solve ODE
        x_1 = sample_fn(
            model=self._velocity_fn,
            x=x_0,
            conditions=conditions,
            steps=steps,
            progress=progress,
        )

        return x_1
