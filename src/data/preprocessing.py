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
    """Adds uniform noise to the input."""

    def __init__(self, noise_level):
        self.noise_level = noise_level

    def transform(self, x, _energy):
        return x + torch.rand_like(x) * self.noise_level

    def inverse_transform(self, x, _energy):
        return cut_below_noise_level(x, self.noise_level)


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


class Standarize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, x, _energy):
        return (x - self.mean) / self.std

    def inverse_transform(self, x, _energy):
        return (x * self.std) + self.mean


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
    def __init__(self, keep_condition_components=None):
        self.keep_condition_components = None
        if keep_condition_components is not None:
            self.keep_condition_components = tuple(keep_condition_components)
            valid_components = {"energy", "phi", "theta", "geo"}
            invalid_components = set(self.keep_condition_components) - valid_components
            if invalid_components:
                raise ValueError(f"Unsupported condition components: {sorted(invalid_components)}")

    def _transform_energy(self, energy):
        energy_min = 1  # after division by 1000
        energy_max = 1000  # after division by 1000
        # return torch.log10(energy / energy_min) / torch.log10(energy_max / energy_min)
        return energy / energy_max

    def _inverse_transform_energy(self, energy):
        energy_min = 1  # after division by 1000
        energy_max = 1000  # after division by 1000
        # return energy_min * (energy_max / energy_min) ** energy
        return energy * energy_max

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
    def __init__(self, steps, keep_condition_components=None):
        self.shower_preprocessor = ShowerPreprocessor(steps)
        self.conditions_preprocessor = ConditionsPreprocessor(keep_condition_components=keep_condition_components)

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
