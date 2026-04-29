from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


def sanitize_config_for_artifact(config: Any):
    """Clone a config object and remove runtime-only resume pointers."""
    if config is None:
        return None

    cloned = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    train_cfg = cloned.get("train")
    if train_cfg is not None and "resume_from_checkpoint" in train_cfg:
        train_cfg["resume_from_checkpoint"] = None
    return cloned
