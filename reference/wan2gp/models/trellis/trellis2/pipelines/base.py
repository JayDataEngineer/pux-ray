# File: trellis2/pipelines/base.py
# trellis2/pipelines/base.py
from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model from a local directory.

        Path must be a local directory containing pipeline.json and model files.
        Models are pre-downloaded by the deployment system (Ray, Docker, etc.).
        """
        import os
        import json

        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Model path not found: {path}\n"
                f"Models must be pre-downloaded before starting inference.\n"
                f"Example: TRELLIS_MODEL_ID=/models/TRELLIS.2-4B"
            )

        config_file = os.path.join(path, "pipeline.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"pipeline.json not found at {config_file}")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            _models[k] = models.from_pretrained(os.path.join(path, v))

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))