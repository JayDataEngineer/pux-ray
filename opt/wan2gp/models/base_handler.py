"""Base class for Wan2GP family handlers.

Each handler only needs to set class attributes for metadata and implement
load_model(). The boilerplate query_* and update_default_settings methods
are provided by this base.

Handlers can also export HANDLER_META to declare their interface and
provide hooks for service-layer integration (device patching, bf16
autocast, kwarg remapping, etc.).
"""
from __future__ import annotations

import base64
from typing import Any


class BaseFamilyHandler:
    """Base class providing the standard family_handler interface.

    Subclasses set class attributes and implement load_model().

    Class attributes to override:
        SUPPORTED_TYPES: list[str] — model type names
        FAMILY: str — family identifier
        FAMILY_INFOS: dict[str, tuple[int, str]] — {type: (id, display_name)}
        MODEL_DEF: dict — {"audio_only": bool, "image_outputs": bool}
        DEFAULTS: dict — merged into ui_defaults by update_default_settings
    """

    SUPPORTED_TYPES: list[str] = []
    FAMILY: str = ""
    FAMILY_INFOS: dict = {}
    MODEL_DEF: dict = {"audio_only": True, "image_outputs": False}
    DEFAULTS: dict = {}


def _make_handler_cls(subclass):
    """Patch static methods on a subclass to use its class attributes.

    Wan2GP calls methods as staticmethods on the class, so we need to
    bind the class attributes to staticmethod returns.
    """
    subclass.query_supported_types = staticmethod(lambda: subclass.SUPPORTED_TYPES)
    subclass.query_family_maps = staticmethod(lambda: ({}, {}))
    subclass.query_model_family = staticmethod(lambda: subclass.FAMILY)
    subclass.query_family_infos = staticmethod(lambda: subclass.FAMILY_INFOS)
    subclass.query_model_def = staticmethod(lambda bmt, md: dict(subclass.MODEL_DEF))
    subclass.update_default_settings = staticmethod(
        lambda bmt, md, ui: ui.update(subclass.DEFAULTS)
    )
    return subclass


def audio_response(data: bytes, media_type: str = "audio/wav") -> dict:
    """Build a standard audio/base64 response dict."""
    return {"status": "success", "data": base64.b64encode(data).decode(),
            "media_type": media_type}


# ─── Handler Hooks Protocol ────────────────────────────────────────────────────

class HandlerHooks:
    """Lifecycle hooks for handler-specific behavior.

    Handlers export a HANDLER_META dict with a ``hooks`` instance.
    The service layer calls these at the appropriate points so per-handler
    logic lives in the handler, not in the service's infer() method.
    """

    # Whether to wrap model.generate() in torch.amp.autocast("cuda", dtype=bfloat16)
    needs_bf16_autocast: bool = False

    # Whether to monkey-patch type(pipeline).device to return cuda
    # (workaround for mmgp keeping weights on CPU between forward passes)
    needs_device_patch: bool = False

    def pre_import(self) -> None:
        """Called before the handler module is imported.

        Use for source file patching that must happen before Python
        imports the module (e.g. anigen decoder scatter ops).
        """

    def on_loaded(self, pipeline: Any, pipe: dict, base_model_type: str) -> None:
        """Called after load_model() and mmgp profile application.

        Use for: attention backend config, rembg patching, moving
        non-module objects to GPU, injecting runtime state.
        """

    def before_generate(self, pipeline: Any, kwargs: dict) -> dict:
        """Called before model.generate(). Receives the normalized kwargs.

        Use for: kwarg remapping, temp file creation, seed clamping,
        inner device patching (e.g. MOSS patches model.model.device).

        Returns the (possibly modified) kwargs dict.
        """
        return kwargs


# ─── Handler Meta Registry ─────────────────────────────────────────────────────

_HANDLER_META_REGISTRY: dict[str, dict] = {}


def register_handler_meta(base_model_type: str, meta: dict) -> None:
    """Register HANDLER_META for a base_model_type."""
    _HANDLER_META_REGISTRY[base_model_type] = meta


def get_handler_meta(base_model_type: str) -> dict | None:
    """Look up registered HANDLER_META for a base_model_type."""
    return _HANDLER_META_REGISTRY.get(base_model_type)
