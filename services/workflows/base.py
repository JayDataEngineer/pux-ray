"""Shared workflow helpers — service singleton + standard response format.

When running inside Forge (set_forge_core called), get_service() returns
a ForgeProxy that routes load/infer through the Forge's VRAM ledger.
Otherwise returns a bare Wan2GPService (for standalone testing).

Use get_native_service() for the native diffusers service (replaces Wan2GP
for models with native diffusers pipeline support).
"""
from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from services.wan2gp.deployment import Wan2GPService

if TYPE_CHECKING:
    from services.forge import ForgeCore
    from services.forge_proxy import ForgeProxy

logger = logging.getLogger(__name__)

_svc: Wan2GPService | ForgeProxy | None = None
_native_svc = None  # Lazy-loaded NativeDiffusersService
_forge_core: ForgeCore | None = None


def get_service() -> Wan2GPService | ForgeProxy:
    global _svc
    if _svc is None:
        if _forge_core is not None:
            from services.forge_proxy import ForgeProxy
            _svc = ForgeProxy(_forge_core)
        else:
            _svc = Wan2GPService()
    return _svc


def set_forge_core(forge: ForgeCore) -> None:
    """Inject ForgeCore for VRAM-aware workflow execution."""
    global _forge_core, _svc
    _forge_core = forge
    _svc = None


def clear_forge_core() -> None:
    """Remove ForgeCore — next get_service() returns bare Wan2GPService."""
    global _forge_core, _svc
    _forge_core = None
    _svc = None


def reset_service() -> None:
    global _svc, _native_svc
    if _svc is not None:
        try:
            _svc.unload()
        except Exception:
            pass
        _svc = None
    if _native_svc is not None:
        try:
            _native_svc.unload()
        except Exception:
            pass
        _native_svc = None


def get_native_service():
    """Get the NativeDiffusersService singleton.

    For standalone testing (outside forge). When inside forge,
    use forge.invoke("native", payload) instead.
    """
    global _native_svc
    if _native_svc is None:
        from services.native.service import NativeDiffusersService
        _native_svc = NativeDiffusersService()
    return _native_svc


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
