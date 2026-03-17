#!/usr/bin/env python3
"""Verify that the updated ConditionsPreprocessor preserves legacy math.

This script compares the current implementation against a copy of the legacy
behavior for the default cases, and also checks that explicit component
selection (e.g. energy-only) is consistent with the corresponding legacy
sub-expressions.
"""

from __future__ import annotations

import argparse

import torch

from src.data.preprocessing import ConditionsPreprocessor


class LegacyConditionsPreprocessor:
    """Reference implementation copied from the pre-selection version."""

    def _transform_energy(self, energy):
        energy_max = 1000
        return energy / energy_max

    def _inverse_transform_energy(self, energy):
        energy_max = 1000
        return energy * energy_max

    def _transform_theta(self, theta):
        theta_min = 1e-8
        theta_max = torch.pi
        return (theta - theta_min) / (theta_max - theta_min)

    def _inverse_transform_theta(self, theta):
        theta_min = 1e-8
        theta_max = torch.pi
        return theta * (theta_max - theta_min) + theta_min

    def _transform_phi(self, phi):
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        return torch.stack([sin_phi, cos_phi], dim=-1)

    def _inverse_transform_phi(self, phi):
        sin_phi, _ = torch.chunk(phi, 2, dim=-1)
        return torch.arcsin(sin_phi)

    def transform(self, conditions):
        geo = None
        if len(conditions) == 4:
            energy, phi, theta, geo = conditions
        else:
            energy, phi, theta = conditions
        energy = self._transform_energy(energy).reshape(-1, 1)
        phi = self._transform_phi(phi).reshape(-1, 2)
        theta = self._transform_theta(theta).reshape(-1, 1)
        if geo is not None:
            geo = geo.reshape(-1, 5)
            return energy, phi, theta, geo
        return energy, phi, theta

    def inverse_transform(self, conditions):
        if len(conditions) == 4:
            energy, phi, theta, _geo = conditions
        else:
            energy, phi, theta = conditions
        energy = self._inverse_transform_energy(energy)
        phi = self._inverse_transform_phi(phi)
        theta = self._inverse_transform_theta(theta)
        return energy, phi, theta


def assert_tuple_close(name: str, actual, expected, atol=1e-7, rtol=1e-7):
    if len(actual) != len(expected):
        raise AssertionError(f"{name}: tuple length mismatch: {len(actual)} != {len(expected)}")
    for idx, (a, e) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(
            a,
            e,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"{name}[{idx}] mismatch: {msg}",
        )


def make_raw_conditions(batch_size: int):
    energy = torch.linspace(1.0, 1000.0, batch_size)
    phi = torch.linspace(-torch.pi + 0.1, torch.pi - 0.1, batch_size)
    theta = torch.linspace(0.05, torch.pi - 0.05, batch_size)
    geo = torch.arange(batch_size * 5, dtype=torch.float32).reshape(batch_size, 5)
    return (energy, phi, theta), (energy, phi, theta, geo)


def verify_default_equivalence():
    current = ConditionsPreprocessor()
    legacy = LegacyConditionsPreprocessor()

    for batch_size in (1, 4, 17):
        raw_3, raw_4 = make_raw_conditions(batch_size)

        transformed_current_3 = current.transform(raw_3)
        transformed_legacy_3 = legacy.transform(raw_3)
        assert_tuple_close("default_transform_3", transformed_current_3, transformed_legacy_3)

        inverted_current_3 = current.inverse_transform(transformed_current_3)
        inverted_legacy_3 = legacy.inverse_transform(transformed_legacy_3)
        assert_tuple_close("default_inverse_3", inverted_current_3, inverted_legacy_3)

        transformed_current_4 = current.transform(raw_4)
        transformed_legacy_4 = legacy.transform(raw_4)
        assert_tuple_close("default_transform_4", transformed_current_4, transformed_legacy_4)

        inverted_current_4 = current.inverse_transform(transformed_current_4)
        inverted_legacy_4 = legacy.inverse_transform(transformed_legacy_4)
        assert_tuple_close("default_inverse_4", inverted_current_4, inverted_legacy_4)


def verify_energy_only_consistency():
    legacy = LegacyConditionsPreprocessor()
    current = ConditionsPreprocessor(keep_condition_components=("energy",))

    for batch_size in (1, 4, 17):
        raw_3, _ = make_raw_conditions(batch_size)
        transformed_legacy = legacy.transform(raw_3)
        transformed_current = current.transform(raw_3)

        if len(transformed_current) != 1:
            raise AssertionError(
                f"energy_only_transform: expected a single condition tensor, got {len(transformed_current)}"
            )
        torch.testing.assert_close(
            transformed_current[0],
            transformed_legacy[0],
            atol=1e-7,
            rtol=1e-7,
        )

        inverted_current = current.inverse_transform(transformed_current)
        if len(inverted_current) != 1:
            raise AssertionError(
                f"energy_only_inverse: expected a single condition tensor, got {len(inverted_current)}"
            )
        torch.testing.assert_close(
            inverted_current[0],
            raw_3[0].reshape(-1, 1),
            atol=1e-7,
            rtol=1e-7,
        )


def main():
    parser = argparse.ArgumentParser(description="Verify ConditionsPreprocessor math against the legacy behavior.")
    parser.add_argument("--verbose", action="store_true", help="Print extra progress information.")
    args = parser.parse_args()

    if args.verbose:
        print("Checking default behavior against legacy implementation...")
    verify_default_equivalence()

    if args.verbose:
        print("Checking energy-only selection against legacy energy branch...")
    verify_energy_only_consistency()

    print("[OK] ConditionsPreprocessor matches legacy math for default behavior and energy-only selection.")


if __name__ == "__main__":
    main()
