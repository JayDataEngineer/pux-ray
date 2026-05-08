"""Playground — interactive web UI for all Ray Serve services.

Provides a unified TNAP-driven interface where users can browse services
by category, dynamically generate input forms from service metadata, send
TNAP requests, and see rendered results (audio, image, 3D, text).

Endpoints:
    GET  /playground              — serve the playground HTML page
    GET  /playground/api/services — all services with UI metadata
"""

from __future__ import annotations

import logging
from pathlib import Path

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from gateway.dashboard import query_service_status
from services.registry import get_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Playground metadata — input fields per service (drives the dynamic form UI)
# ---------------------------------------------------------------------------
# Each field has:
#   key        — the TNAP input field name
#   label      — human-readable label
#   type       — text, textarea, number, range, image, audio, select, checkbox
#   required   — whether the field is required
#   default    — default value
#   placeholder
#   min/max/step   — for range/number
#   options        — for select
#   help           — help text shown below field

PLAYGROUND_META: dict[str, dict] = {
    "kokoro": {
        "category": "TTS",
        "gpu": False,
        "route": "/tts/kokoro",
        "format": "tnap",
        "input_fields": [
            {"key": "text", "label": "Text to speak", "type": "textarea", "required": True, "placeholder": "Enter text to synthesize..."},
            {"key": "voice", "label": "Voice", "type": "text", "default": "af_bella", "help": "Available: af_bella, af_heart, af_nicole, am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis"},
            {"key": "speed", "label": "Speed", "type": "range", "min": 0.5, "max": 2.0, "step": 0.1, "default": 1.0},
        ],
    },
    "espeak": {
        "category": "TTS",
        "gpu": False,
        "route": "/tts/espeak",
        "format": "tnap",
        "input_fields": [
            {"key": "text", "label": "Text to speak", "type": "textarea", "required": True, "placeholder": "Enter text to synthesize..."},
            {"key": "voice", "label": "Voice / Language", "type": "text", "default": "en", "placeholder": "en, fr, de, es, ja, etc."},
            {"key": "speed", "label": "Speed", "type": "range", "min": 0.5, "max": 2.0, "step": 0.1, "default": 1.0},
        ],
    },
    "faster_whisper": {
        "category": "ASR",
        "gpu": False,
        "route": "/asr/whisper",
        "format": "tnap",
        "input_fields": [
            {"key": "audio", "label": "Audio file", "type": "audio", "required": True, "help": "Upload audio to transcribe (WAV, MP3, M4A, etc.)"},
            {"key": "language", "label": "Language hint", "type": "text", "default": "", "placeholder": "en, fr, de, or leave empty for auto-detect"},
        ],
    },
    "index_tts": {
        "category": "TTS",
        "gpu": True,
        "route": "/tts/index-tts",
        "format": "tnap",
        "input_fields": [
            {"key": "text", "label": "Text to speak", "type": "textarea", "required": True, "placeholder": "Enter text to synthesize..."},
            {"key": "voice", "label": "Voice", "type": "text", "default": "default"},
        ],
    },
    "vibevoice": {
        "category": "TTS",
        "gpu": True,
        "route": "/tts/vibevoice-cpp",
        "format": "tnap",
        "input_fields": [
            {"key": "text", "label": "Text to speak", "type": "textarea", "required": True, "placeholder": "Enter text to synthesize..."},
            {"key": "voice", "label": "Voice", "type": "text", "default": "default"},
        ],
    },
    "moss_soundeffect": {
        "category": "Audio",
        "gpu": True,
        "route": "/forge",
        "format": "forge",
        "input_fields": [
            {"key": "prompt", "label": "Sound description", "type": "textarea", "required": True, "placeholder": "e.g., thunder and heavy rain, ocean waves crashing, footsteps on gravel..."},
        ],
    },
    "ace_step": {
        "category": "Music",
        "gpu": True,
        "route": "/forge",
        "format": "forge",
        "input_fields": [
            {"key": "prompt", "label": "Music description", "type": "textarea", "required": True, "placeholder": "e.g., upbeat electronic dance music with a driving bass line"},
            {"key": "duration", "label": "Duration (seconds)", "type": "number", "default": 30, "min": 5, "max": 120},
            {"key": "bpm", "label": "BPM", "type": "number", "default": 120, "min": 40, "max": 200},
            {"key": "instrumental", "label": "Instrumental only", "type": "checkbox", "default": True},
        ],
    },
    "llm": {
        "category": "LLM",
        "gpu": True,
        "route": "/llm",
        "format": "openai",
        "input_fields": [
            {"key": "system", "label": "System prompt", "type": "textarea", "required": False, "placeholder": "You are a helpful assistant...", "default": ""},
            {"key": "messages", "label": "Message", "type": "textarea", "required": True, "placeholder": "Enter your message..."},
            {"key": "temperature", "label": "Temperature", "type": "range", "min": 0.0, "max": 2.0, "step": 0.1, "default": 0.7},
            {"key": "max_tokens", "label": "Max tokens", "type": "number", "min": 64, "max": 32768, "default": 2048},
        ],
    },
    "trellis": {
        "category": "3D",
        "gpu": True,
        "route": "/forge",
        "format": "forge",
        "input_fields": [
            {"key": "image", "label": "Source image", "type": "image", "required": True, "help": "Upload an image to convert to 3D mesh"},
            {"key": "seed", "label": "Seed", "type": "number", "default": 1},
            {"key": "steps", "label": "Steps", "type": "number", "default": 12, "min": 1, "max": 50},
            {"key": "guidance", "label": "Guidance scale", "type": "range", "min": 1.0, "max": 20.0, "step": 0.5, "default": 7.5},
        ],
    },
    "anigen": {
        "category": "3D",
        "gpu": True,
        "route": "/forge",
        "format": "forge",
        "input_fields": [
            {"key": "image", "label": "Anime image", "type": "image", "required": True, "help": "Upload an anime-style image to convert to 3D"},
            {"key": "seed", "label": "Seed", "type": "number", "default": -1},
            {"key": "steps", "label": "Steps", "type": "number", "default": 12, "min": 1, "max": 50},
            {"key": "guidance", "label": "Guidance scale", "type": "range", "min": 1.0, "max": 20.0, "step": 0.5, "default": 7.0},
        ],
    },
    "see_through": {
        "category": "Creative",
        "gpu": True,
        "route": "/forge",
        "format": "forge",
        "input_fields": [
            {"key": "image", "label": "Anime image", "type": "image", "required": True, "help": "Upload an anime image for layer decomposition"},
        ],
    },
    "phi4mm": {
        "category": "Vision",
        "gpu": True,
        "route": None,
        "format": "tnap",
        "input_fields": [
            {"key": "image", "label": "Image (optional, for multimodal)", "type": "image"},
            {"key": "text", "label": "Text prompt", "type": "textarea", "required": True, "placeholder": "Describe the image or ask a question..."},
            {"key": "max_tokens", "label": "Max tokens", "type": "number", "default": 1024, "min": 64, "max": 8192},
        ],
    },
}


# Endpoints

_PLAYGROUND_HTML = Path(__file__).parent / "playground.html"


async def playground_page(request: Request) -> HTMLResponse:
    """GET /playground — serve the playground UI."""
    return HTMLResponse(_PLAYGROUND_HTML.read_text())


async def playground_services(request: Request) -> JSONResponse:
    """GET /playground/api/services — all services with UI metadata."""
    deploy_status = {s["name"]: s for s in query_service_status()}

    services = []
    for name, meta in PLAYGROUND_META.items():
        entry = get_service(name)
        dep = deploy_status.get(entry.deployment if entry else name, {})
        services.append({
            "name": name,
            "label": entry.label if entry else name,
            "category": meta["category"],
            "gpu": meta["gpu"],
            "route": meta.get("route"),
            "format": meta.get("format", "tnap"),
            "input_fields": meta["input_fields"],
            "output_type": entry.output_type if entry else "unknown",
            "description": entry.description if entry else "",
            "status": dep.get("status", "UNKNOWN"),
            "running_replicas": dep.get("running_replicas", 0),
        })

    return JSONResponse(services)
