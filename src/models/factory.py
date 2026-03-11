from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.method_base import MethodBase
from src.utils import import_class_by_name


def _to_config(cfg: Any):
    if cfg is None:
        return OmegaConf.create({})
    return OmegaConf.create(cfg)


def _load_saved_config(model_path: str | None):
    if model_path is None:
        return None
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    return state["config"]


def _resolve_model_and_method_config(model_cfg, method_cfg):
    model_cfg = _to_config(model_cfg)
    method_cfg = _to_config(method_cfg)

    model_path = model_cfg.get("model_path")
    saved_root_cfg = _load_saved_config(model_path)

    architecture_cfg = _to_config(model_cfg.get("architecture"))
    resolved_method_cfg = method_cfg

    if saved_root_cfg is not None:
        saved_model_cfg = _to_config(saved_root_cfg.get("model"))
        saved_method_cfg = _to_config(saved_root_cfg.get("method"))
        architecture_cfg = OmegaConf.merge(saved_model_cfg.get("architecture", {}), architecture_cfg)
        resolved_method_cfg = OmegaConf.merge(saved_method_cfg, resolved_method_cfg)

    if architecture_cfg.get("target") is None:
        raise ValueError("Missing model.architecture.target. Provide model.architecture or load from model.model_path.")

    if resolved_method_cfg.get("target") is None:
        raise ValueError("Missing method.target. Provide method config or load from model.model_path.")

    return architecture_cfg, resolved_method_cfg, model_path


def create_model_from_config(model_cfg, method_cfg):
    architecture_cfg, method_cfg, model_path = _resolve_model_and_method_config(model_cfg, method_cfg)

    architecture_cls = import_class_by_name(architecture_cfg["target"])
    assert issubclass(architecture_cls, nn.Module), (
        f"Architecture class {architecture_cls} must be a subclass of nn.Module."
    )

    method_cls = import_class_by_name(method_cfg["target"])

    architecture = architecture_cls(**architecture_cfg.get("init_args", {}))

    assert issubclass(method_cls, MethodBase), (
        f"Method class {method_cls} must be a subclass of MethodBase."
    )

    model = method_cls(model=architecture, **method_cfg.get("init_args", {}))

    if model_path is not None:
        model.load_state(model_path)

    model.eval()

    return model
