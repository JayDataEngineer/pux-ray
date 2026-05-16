"""Base class for Wan2GP family handlers.

Each handler only needs to set class attributes for metadata and implement
load_model(). The boilerplate query_* and update_default_settings methods
are provided by this base.
"""
from __future__ import annotations

import base64


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
