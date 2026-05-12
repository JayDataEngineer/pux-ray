"""Base handler contract — every model family implements this.

The handler decomposes a model into a pipe dict of {name: nn.Module}.
mmgp's offload.profile() manages VRAM/CPU/RAM placement for those modules.
The pipeline object knows how to run inference on the decomposed model.

Reference: Wan2GP's family_handler pattern, internalized for Tech Noir.
"""
from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelVariant:
    """A specific model variant within a family (e.g., ace_step_v1_5_turbo)."""
    name: str                           # e.g., "ace_step_v1_5_turbo"
    family: str                         # e.g., "ace_step"
    display_name: str                   # e.g., "ACE-Step 1.5 Turbo"
    vram_estimate_gb: float             # rough estimate for planning
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadResult:
    """Result from load_model() — the decomposition."""
    pipeline: Any                       # knows how to run inference
    pipe: dict[str, torch.nn.Module]    # mmgp manages these
    co_tenants: dict[str, list[str]] = field(default_factory=dict)
    # co_tenants: models that can share VRAM, e.g., {"transformer": ["vae"]}
    # mmgp uses this to decide what stays loaded together


class BaseHandler(ABC):
    """Contract every model handler implements.

    Usage:
        handler = AceStepHandler()
        result = handler.load_model("ace_step_v1_5_turbo", model_path, dtype=torch.bfloat16)
        pipeline = result.pipeline
        pipe = result.pipe  # pass to mmgp offload.profile()
    """

    @abstractmethod
    def supported_types(self) -> list[str]:
        """Return all model type strings this handler supports."""
        ...

    @abstractmethod
    def load_model(
        self,
        model_type: str,
        model_path: Path,
        dtype: torch.dtype = torch.bfloat16,
        quantize_transformer: bool = False,
        **kwargs,
    ) -> LoadResult:
        """Decompose a model into pipeline + pipe dict.

        Args:
            model_type: which variant to load (e.g., "ace_step_v1_5_turbo")
            model_path: directory containing model weights
            dtype: target dtype for model weights
            quantize_transformer: whether to quantize the main transformer

        Returns:
            LoadResult with pipeline object and pipe dict of nn.Modules
        """
        ...

    @abstractmethod
    def get_variant(self, model_type: str) -> ModelVariant:
        """Get metadata for a specific model variant."""
        ...

    def default_settings(self, model_type: str) -> dict[str, Any]:
        """Default inference settings for a model type. Override if needed."""
        variant = self.get_variant(model_type)
        return dict(variant.defaults)
