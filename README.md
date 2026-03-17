# CaloLightingDiT

Generative calorimeter shower modeling for CaloChallenge-style data, built around Hydra configs, Hugging Face Accelerate, and PyTorch.

This repository contains:

- Flow-matching based calorimeter generation pipelines
- EDM-based CaloDiT
- A 3D LightingDiT-style backbone with RoPE 

## Condition Input Formats

Condition preprocessing and model-side condition embedding support three input formats:

- Four-element tuples for multi calorimeter : `(energy, phi, theta, geo)`
- Single-element tuples for energy-only conditioning: `(energy,)`
- Three-element tuples for continuous conditioning: `(energy, phi, theta)`

This makes it possible to use the same codepath for energy-only experiments, ,  .
