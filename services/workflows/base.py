"""Shared workflow helpers — service singleton + standard response format.

When running inside Forge (set_forge_core called), get_service() returns
a ForgeProxy that routes load/infer through the Forge's VRAM ledger.
Otherwise returns a bare NativeService (SGLang HTTP client) for standalone testing.

All models are now served through the native diffusers service.
No Wan2GP, no mmGP.
"""
from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.forge import ForgeCore
    from services.forge_proxy import ForgeProxy
    from services.native.service import NativeService

logger = logging.getLogger(__name__)

_svc: NativeService | ForgeProxy | None = None
_forge_core: ForgeCore | None = None


def get_service() -> NativeService | ForgeProxy:
    """Get the service singleton for workflow execution.

    Inside forge: returns ForgeProxy (VRAM-tracked).
    Standalone: returns bare NativeService.
    """
    global _svc
    if _svc is None:
        if _forge_core is not None:
            from services.forge_proxy import ForgeProxy
            _svc = ForgeProxy(_forge_core)
        else:
            from services.native.service import NativeService
            _svc = NativeService()
    return _svc


def set_forge_core(forge: ForgeCore) -> None:
    """Inject ForgeCore for VRAM-aware workflow execution."""
    global _forge_core, _svc
    _forge_core = forge
    _svc = None


def clear_forge_core() -> None:
    """Remove ForgeCore — next get_service() returns bare NativeService."""
    global _forge_core, _svc
    _forge_core = None
    _svc = None


def reset_service() -> None:
    global _svc
    if _svc is not None:
        try:
            _svc.unload()
        except Exception:
            pass
        _svc = None


def decode_image(image_b64: str) -> bytes:
    return base64.b64decode(image_b64)


def encode_output(data: bytes, media_type: str = "image/png") -> dict[str, Any]:
    return {
        "status": "ok",
        "data": base64.b64encode(data).decode(),
        "media_type": media_type,
    }


def error_response(message: str) -> dict[str, Any]:
    return {"status": "error", "error": message}
