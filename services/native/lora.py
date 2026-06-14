"""LoRA management via PEFT — replaces mmGP's 560-line load_loras_into_model.

Key rules (from deep research verification):
  1. Load ALL LoRAs BEFORE enabling group_offload (hooks need to see adapter params)
  2. PEFT is compatible with torch.compile (unlike mmGP's monkey-patching)
  3. set_adapters is <5ms (just updates scale multipliers in-place)
  4. load_lora_weights takes 150-850ms (disk/PCIe transfer)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class LoRAManager:
    """Manages LoRA loading, swapping, and fusion for a pipeline."""

    def __init__(self, pipe):
        self.pipe = pipe
        self._loaded_adapters: dict[str, dict] = {}  # name → {path, scale, weight_name}

    def load(self, path: str, adapter_name: str = "default",
             weight_name: Optional[str] = None) -> str:
        """Load a LoRA adapter. Must be called BEFORE group_offload."""
        # Resolve path
        if not os.path.exists(path):
            # Try common LoRA directories
            for prefix in ("/models/wan2gp/loras", "/models/loras", "/models"):
                candidate = os.path.join(prefix, path)
                if os.path.exists(candidate):
                    path = candidate
                    break

        try:
            kwargs = {"adapter_name": adapter_name}
            if weight_name:
                kwargs["weight_name"] = weight_name

            self.pipe.load_lora_weights(path, **kwargs)
            self._loaded_adapters[adapter_name] = {
                "path": path,
                "scale": 1.0,
                "weight_name": weight_name,
            }
            logger.info("LoRA: loaded '%s' from %s", adapter_name, path)
            return adapter_name
        except Exception as e:
            logger.error("LoRA: failed to load '%s' from %s: %s", adapter_name, path, e)
            raise

    def set_active(self, adapter_names: list[str], scales: list[float] | None = None) -> None:
        """Activate specific LoRAs with independent weight scales."""
        if scales is None:
            scales = [1.0] * len(adapter_names)

        # Validate all adapters are loaded
        for name in adapter_names:
            if name not in self._loaded_adapters:
                raise ValueError(f"LoRA '{name}' not loaded. Call load() first.")

        self.pipe.set_adapters(adapter_names, adapter_weights=scales)

        for name, scale in zip(adapter_names, scales):
            self._loaded_adapters[name]["scale"] = scale

        logger.info("LoRA: active=%s scales=%s", adapter_names, scales)

    def set_scale(self, adapter_name: str, scale: float) -> None:
        """Dynamically adjust a single LoRA's strength (<5ms)."""
        if adapter_name not in self._loaded_adapters:
            raise ValueError(f"LoRA '{adapter_name}' not loaded")

        self._loaded_adapters[adapter_name]["scale"] = scale
        # Re-set all adapters with updated scales
        names = list(self._loaded_adapters.keys())
        scales = [self._loaded_adapters[n]["scale"] for n in names]
        self.pipe.set_adapters(names, adapter_weights=scales)

    def unload(self, adapter_name: str | None = None) -> None:
        """Unload one or all LoRA adapters."""
        if adapter_name:
            self.pipe.unload_lora_weights(adapter_name)
            self._loaded_adapters.pop(adapter_name, None)
            logger.info("LoRA: unloaded '%s'", adapter_name)
        else:
            self.pipe.unload_lora_weights()
            self._loaded_adapters.clear()
            logger.info("LoRA: unloaded all")

    def list_adapters(self) -> dict[str, dict]:
        """Return info about all loaded adapters."""
        return dict(self._loaded_adapters)

    def fuse(self, adapter_names: list[str] | None = None, lora_scale: float = 1.0) -> None:
        """Fuse LoRA weights into base model (faster inference, no adapter overhead)."""
        names = adapter_names or list(self._loaded_adapters.keys())
        if not names:
            return

        self.pipe.fuse_lora(adapter_names=names, lora_scale=lora_scale)
        logger.info("LoRA: fused %s into base model", names)
