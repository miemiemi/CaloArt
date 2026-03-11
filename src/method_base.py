from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torchinfo import summary

from src.utils import get_logger

logger = get_logger()


class MethodBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.model.parameters()).device

    def save_config(self, config: dict) -> None:
        self.config = config

    def save_state(self, save_path: Union[str, Path]) -> None:
        torch.save({"model": self.model.state_dict(), "config": self.config}, save_path)
        logger.info(f"Saved model to {save_path}")

    def load_state(self, load_path: Union[str, Path]) -> None:
        state = torch.load(load_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.config = state["config"]
        logger.info(f"Loaded model from {load_path}")

    def summarize(self):
        summary(self.model, input_data=self.model.example_input, col_names=["input_size", "output_size", "num_params"])
