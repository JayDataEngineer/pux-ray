"""Shared workflow helpers — Wan2GPService singleton, standard response format."""
from __future__ import annotations

import base64
import logging
from typing import Any

from services.wan2gp.deployment import Wan2GPService

logger = logging.getLogger(__name__)

_svc: Wan2GPService | None = None


def get_service() -> Wan2GPService:
    global _svc
    if _svc is None:
        _svc = Wan2GPService()
    return _svc


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
