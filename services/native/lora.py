"""LoRA management via PEFT — replaces mmGP's 560-line load_loras_into_model.

Rules (verified):
  1. Load LoRAs BEFORE group_offload (hooks need adapter params)
  2. PEFT is compile-compatible (mmGP's wasn't)
  3. set_adapters <5ms (in-place scale update)
"""
from __future__ import annotations
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class LoRAManager:
    def __init__(self, pipe):
        self.pipe = pipe
        self._adapters: dict[str, dict] = {}

    def load(self, path: str, name: str = "default", weight_name: str | None = None) -> str:
        if not os.path.exists(path):
            for prefix in ("/models/loras", "/models/wan2gp/loras", "/models"):
                candidate = os.path.join(prefix, path)
                if os.path.exists(candidate):
                    path = candidate
                    break
        kwargs = {"adapter_name": name}
        if weight_name:
            kwargs["weight_name"] = weight_name
        self.pipe.load_lora_weights(path, **kwargs)
        self._adapters[name] = {"path": path, "scale": 1.0}
        logger.info("LoRA loaded: '%s'", name)
        return name

    def set_active(self, names: list[str], scales: list[float] | None = None) -> None:
        if scales is None:
            scales = [1.0] * len(names)
        for n in names:
            if n not in self._adapters:
                raise ValueError(f"LoRA '{n}' not loaded")
        self.pipe.set_adapters(names, adapter_weights=scales)
        for n, s in zip(names, scales):
            self._adapters[n]["scale"] = s

    def set_scale(self, name: str, scale: float) -> None:
        self._adapters[name]["scale"] = scale
        names = list(self._adapters.keys())
        scales = [self._adapters[n]["scale"] for n in names]
        self.pipe.set_adapters(names, adapter_weights=scales)

    def unload(self, name: str | None = None) -> None:
        if name:
            self.pipe.unload_lora_weights(name) if hasattr(self.pipe, "unload_lora_weights") else None
            self._adapters.pop(name, None)
        else:
            if hasattr(self.pipe, "unload_lora_weights"):
                self.pipe.unload_lora_weights()
            self._adapters.clear()

    def list(self) -> dict[str, dict]:
        return dict(self._adapters)
