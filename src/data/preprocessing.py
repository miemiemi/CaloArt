import math

import torch
import numpy as np

from src.utils import append_dims, get_logger, import_class_by_name

logger = get_logger()


def cut_below_noise_level(x, noise_level):
    return torch.where(x < noise_level, 0.0, x)


class CutNoise:
    """Cuts off noise below a certain threshold."""

    def __init__(self, noise_level, both_directions=False):
        self.noise_level = noise_level  # a.k.a. energy readout threshold
        self.both_directions = both_directions

    def transform(self, x, _energy):
        if self.both_directions:
            return cut_below_noise_level(x, self.noise_level)
        else:
            return x

    def inverse_transform(self, x, _energy):
        return cut_below_noise_level(x, self.noise_level)


class AddNoise:
    """Adds one-sided uniform noise during preprocessing only."""

    def __init__(self, noise_level):
        self.noise_level = noise_level

    def transform(self, x, _energy):
        return x + torch.rand_like(x) * self.noise_level

    def inverse_transform(self, x, _energy):
        # Final thresholding should be handled explicitly by CutNoise.
        return x


class ScaleAboveCut:
    """Amplifies voxels above a threshold during preprocessing and undoes it on inverse."""

    def __init__(self, factor, threshold):
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}.")
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}.")
        self.factor = factor
        self.threshold = threshold
        self.scaled_threshold = threshold * factor

    def transform(self, x, _energy):
        return torch.where(x > self.threshold, x * self.factor, x)

    def inverse_transform(self, x, _energy):
        return torch.where(x > self.scaled_threshold, x / self.factor, x)


class LogitTransform:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def transform(self, x, _energy):
        z = self.eps + (1 - 2 * self.eps) * x
        return torch.logit(z)

    def inverse_transform(self, x, _energy):
        z = torch.sigmoid(x)
        return (z - self.eps) / (1 - 2 * self.eps)


class LogTransform:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def transform(self, x, _energy):
        return torch.log(x + self.eps)

    def inverse_transform(self, x, _energy):
        return torch.exp(x) - self.eps


class Log1pTransform:
    """Applies log(1 + x / scale) with a stable inverse."""

    def __init__(self, scale):
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}.")
        self.scale = scale

    def transform(self, x, _energy):
        return torch.log1p(x / self.scale)

    def inverse_transform(self, x, _energy):
        return torch.expm1(x) * self.scale


class AsinhTransform:
    """Applies asinh(x / scale) with a stable inverse."""

    def __init__(self, scale):
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}.")
        self.scale = scale

    def transform(self, x, _energy):
        return torch.asinh(x / self.scale)

    def inverse_transform(self, x, _energy):
        return torch.sinh(x) * self.scale


class Standarize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, x, _energy):
        return (x - self.mean) / self.std

    def inverse_transform(self, x, _energy):
        return (x * self.std) + self.mean


class StandarizeHalf:
    """Standardize to half-scale outputs while keeping inverse self-consistent."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, x, _energy):
        return (x - self.mean) / self.std * 0.5

    def inverse_transform(self, x, _energy):
        return (x * self.std * 2) + self.mean


class ScaleByIncidentEnergy:
    def transform(self, x, energy):
        return x / append_dims(energy, x.ndim)

    def inverse_transform(self, x, energy):
        return x * append_dims(energy, x.ndim)


class ScaleByFactor:
    def __init__(self, factor):
        self.factor = factor

    def transform(self, x, _energy):
        return x / self.factor

    def inverse_transform(self, x, _energy):
        return x * self.factor


class RemoveSamplingFraction:
    def __init__(self, factor, reverse=True):
        self.factor = factor
        self.reverse = reverse

    def transform(self, x, _):
        return x * self.factor

    def inverse_transform(self, x, _):
        if self.reverse:
            return x / self.factor
        # return as it is, can use non sampling fractioned files to plot shower observables
        return x


""" WIP
class Decouple:
    def __init__(self, eps):
        self.eps = eps
    
    def transform(self, x, _energy):
        alpha = x.sum(dim=tuple(range(1, x.ndim)))  # [B, 1]
        p = x / alpha  # [B, ...]
        p = self.eps + (1 - 2 * self.eps) * p
        logits = torch.logit(p)
        return logits
    
    def inverse_transform(self, logits, alpha_hat, _energy):
        p_hat = torch.sigmoid(logits)
        p_hat = p_hat / p_hat.sum(dim=tuple(range(1, x.ndim)), keepdim=True)
        x_hat = alpha_hat * p_hat
        return x_hat
"""

class ShowerPreprocessor:
    def __init__(self, steps):
        self.steps = steps
        self.pipeline = []
        for step in steps:
            class_name, init_args = step['class_name'], step.get('init_args', {})

            if "noise_level" in init_args:
                self.noise_level = init_args["noise_level"]

            class_ = import_class_by_name(class_name)
            self.pipeline.append(class_(**init_args))

            # sanity checks
            if not hasattr(self.pipeline[-1], "transform"):
                raise ValueError(f"Class {class_name} does not have a `transform` method.")
            if not hasattr(self.pipeline[-1], "inverse_transform"):
                raise ValueError(f"Class {class_name} does not have an `inverse_transform` method.")

    def transform(self, showers, energy=None):
        for step in self.pipeline:
            showers = step.transform(showers, energy)

        return showers

    def inverse_transform(self, showers, energy=None):
        for step in reversed(self.pipeline):
            showers = step.inverse_transform(showers, energy)

        return showers

    def inverse_transform_with_trace(self, showers, energy=None, trace_fn=None):
        for step in reversed(self.pipeline):
            showers = step.inverse_transform(showers, energy)
            if trace_fn is not None:
                trace_fn(type(step).__name__, showers)

        return showers


class ConditionsPreprocessor:
    def __init__(
        self,
        keep_condition_components=None,
        energy_encoding="linear",
        energy_min=1.0,
        energy_max=1000.0,
    ):
        self.keep_condition_components = None
        if keep_condition_components is not None:
            self.keep_condition_components = tuple(keep_condition_components)
            valid_components = {"energy", "phi", "theta", "geo"}
            invalid_components = set(self.keep_condition_components) - valid_components
            if invalid_components:
                raise ValueError(f"Unsupported condition components: {sorted(invalid_components)}")
        valid_energy_encodings = {"linear", "log10"}
        if energy_encoding not in valid_energy_encodings:
            raise ValueError(
                f"Unsupported energy_encoding '{energy_encoding}'. "
                f"Expected one of {sorted(valid_energy_encodings)}."
            )
        if energy_min <= 0:
            raise ValueError(f"energy_min must be positive, got {energy_min}.")
        if energy_max <= energy_min:
            raise ValueError(
                f"energy_max must be larger than energy_min, got energy_min={energy_min}, energy_max={energy_max}."
            )
        self.energy_encoding = energy_encoding
        self.energy_min = float(energy_min)
        self.energy_max = float(energy_max)

    def _transform_energy(self, energy):
        if self.energy_encoding == "linear":
            return energy / self.energy_max

        encoded = torch.log10(energy.clamp_min(self.energy_min) / self.energy_min)
        return encoded / math.log10(self.energy_max / self.energy_min)

    def _inverse_transform_energy(self, energy):
        if self.energy_encoding == "linear":
            return energy * self.energy_max

        base = torch.as_tensor(
            self.energy_max / self.energy_min,
            dtype=energy.dtype,
            device=energy.device,
        )
        return self.energy_min * torch.pow(base, energy)

    # [0.0, 3.14] -> [0, 1]
    def _transform_theta(self, theta):
        theta_min = 1e-8
        theta_max = torch.pi
        # return torch.log10(theta / theta_min) / torch.log10(theta_max / theta_min)
        return (theta - theta_min) / (theta_max - theta_min)

    def _inverse_transform_theta(self, theta):
        theta_min = 1e-8
        theta_max = torch.pi
        # theta = theta_min * (theta_max / theta_min) ** theta
        return theta * (theta_max - theta_min) + theta_min

    # [-pi, pi] -> [0, 1]
    def _transform_phi(self, phi):
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        return torch.stack([sin_phi, cos_phi], dim=-1)

    def _inverse_transform_phi(self, phi):
        sin_phi, cos_phi = torch.chunk(phi, 2, dim=-1)
        # sin in [-pi, pi] is invertible
        phi_from_sin = torch.arcsin(sin_phi)
        return phi_from_sin

    def _select_components(self, components):
        if self.keep_condition_components is None:
            return tuple(components.values())
        selected = []
        for key in self.keep_condition_components:
            if key not in components:
                raise ValueError(f"Requested condition component '{key}' is unavailable.")
            selected.append(components[key])
        return tuple(selected)

    def transform(self, conditions):
        geo = None
        if len(conditions)==4:
            energy, phi, theta, geo = conditions
        elif len(conditions) == 1:
            (energy,) = conditions
            phi = theta = None
        else:
            energy, phi, theta = conditions
        components = {
            "energy": self._transform_energy(energy).reshape(-1, 1),
        }
        if phi is not None:
            components["phi"] = self._transform_phi(phi).reshape(-1, 2)
        if theta is not None:
            components["theta"] = self._transform_theta(theta).reshape(-1, 1)
        if geo is not None:
            components["geo"] = geo.reshape(-1, 5)

        return self._select_components(components)

    def inverse_transform(self, conditions):
        components = {}
        include_geo_in_output = self.keep_condition_components is not None and "geo" in self.keep_condition_components
        if self.keep_condition_components is None:
            if len(conditions) == 1:
                component_keys = ("energy",)
            elif len(conditions) == 3:
                component_keys = ("energy", "phi", "theta")
            elif len(conditions) == 4:
                component_keys = ("energy", "phi", "theta", "geo")
            else:
                raise ValueError(f"Cannot infer condition names from {len(conditions)} transformed tensors.")
        else:
            component_keys = self.keep_condition_components
            if len(conditions) != len(component_keys):
                raise ValueError(
                    f"Expected {len(component_keys)} transformed conditions, got {len(conditions)}."
                )
        for key, value in zip(component_keys, conditions):
            components[key] = value

        outputs = []
        if "energy" in components:
            outputs.append(self._inverse_transform_energy(components["energy"]))
        if "phi" in components:
            outputs.append(self._inverse_transform_phi(components["phi"]))
        if "theta" in components:
            outputs.append(self._inverse_transform_theta(components["theta"]))
        if include_geo_in_output and "geo" in components:
            outputs.append(components["geo"])

        return tuple(outputs)


class CaloShowerPreprocessor:
    def __init__(self, steps, keep_condition_components=None, condition_preprocessing=None):
        self.shower_preprocessor = ShowerPreprocessor(steps)
        condition_preprocessing = condition_preprocessing or {}
        self.conditions_preprocessor = ConditionsPreprocessor(
            keep_condition_components=keep_condition_components,
            **condition_preprocessing,
        )

    def transform(self, showers=None, conditions=None):
        if showers is not None and conditions is not None:
            energy, *_ = conditions
            showers = self.shower_preprocessor.transform(showers, energy)
            conditions = self.conditions_preprocessor.transform(conditions)
        elif conditions is not None:
            conditions = self.conditions_preprocessor.transform(conditions)
        elif showers is not None:
            showers = self.shower_preprocessor.transform(showers)
        else:
            raise ValueError("Expected either showers or conditions")

        return showers, conditions

    def inverse_transform(self, showers=None, conditions=None, trace_fn=None):
        if showers is not None and conditions is not None:
            conditions = self.conditions_preprocessor.inverse_transform(conditions)
            energy, *_ = conditions
            if trace_fn is None:
                showers = self.shower_preprocessor.inverse_transform(showers, energy)
            else:
                showers = self.shower_preprocessor.inverse_transform_with_trace(
                    showers,
                    energy,
                    trace_fn=trace_fn,
                )
        elif conditions is not None:
            conditions = self.conditions_preprocessor.inverse_transform(conditions)
        elif showers is not None:
            if trace_fn is None:
                showers = self.shower_preprocessor.inverse_transform(showers)
            else:
                showers = self.shower_preprocessor.inverse_transform_with_trace(
                    showers,
                    trace_fn=trace_fn,
                )
        else:
            raise ValueError("Expected either showers or conditions")

        return showers, conditions
