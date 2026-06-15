"""Standalone MOSS audio server with model switching.

Wraps MossSoundEffectPipeline as a simple HTTP API.
Runs in a separate Docker container — no Wan2GP, no diffusers.

Supports multiple MOSS model variants (SoundEffect, TTS, etc.)
with automatic load/unload/switch.

POST /generate
  { "prompt": "rain on tin roof", "model": "moss-soundeffect", "seconds": 10, "seed": 0 }
  → { "audio": "<base64 wav>", "sample_rate": 48000 }

POST /release
  → Unloads current model, frees VRAM

GET /health
  → { "status": "ok", "loaded_model": "moss-soundeffect" }
"""
from __future__ import annotations

import base64
import gc
import io
import logging
import os
import sys
import time

import torch
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("moss-server")

# Add vendored MOSS code to path
MOSS_CODE_PATH = os.environ.get("MOSS_CODE_PATH", "/opt/moss")
if os.path.exists(MOSS_CODE_PATH):
    sys.path.insert(0, MOSS_CODE_PATH)

PORT = int(os.environ.get("MOSS_PORT", "8081"))

# Model registry: name → path on disk
MOSS_MODELS = {
    "moss-soundeffect": "/models/audio/moss-soundeffect",
    "moss-soundeffect-v2": "/models/wan2gp/moss_soundeffect_v2",
    "moss-tts": "/models/audio/moss-tts",
}

# Currently loaded state
_pipeline = None
_loaded_model: str | None = None


def load_model(model_name: str):
    """Load a MOSS pipeline. Switches if a different model is loaded."""
    global _pipeline, _loaded_model

    # Already loaded — return immediately
    if _pipeline is not None and _loaded_model == model_name:
        return _pipeline

    # Different model loaded — unload first
    if _pipeline is not None:
        logger.info("MOSS: switching from '%s' to '%s'", _loaded_model, model_name)
        unload_model()

    # Resolve path
    model_path = MOSS_MODELS.get(model_name)
    if model_path is None:
        raise ValueError(f"Unknown MOSS model '{model_name}'. Available: {list(MOSS_MODELS.keys())}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MOSS model not found at {model_path}")

    logger.info("MOSS: loading '%s' from %s", model_name, model_path)

    from pipeline_moss_soundeffect import MossSoundEffectPipeline

    _pipeline = MossSoundEffectPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    _loaded_model = model_name
    logger.info("MOSS: '%s' loaded successfully", model_name)
    return _pipeline


def unload_model():
    """Unload the current model and free VRAM."""
    global _pipeline, _loaded_model

    if _pipeline is not None:
        logger.info("MOSS: unloading '%s'", _loaded_model)
        del _pipeline
        _pipeline = None
        _loaded_model = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    logger.info("MOSS: model unloaded, VRAM freed")


def generate_sound(model: str, prompt: str, seconds: float = 10.0,
                   seed: int = 0, steps: int = 100, cfg: float = 4.0) -> dict:
    """Generate sound effect from text prompt."""
    pipe = load_model(model)

    t0 = time.perf_counter()
    output = pipe(
        prompt=prompt,
        seconds=seconds,
        num_inference_steps=steps,
        cfg_scale=cfg,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0

    # Get audio tensor
    if hasattr(output, "audios"):
        audio = output.audios
    else:
        audio = output

    # Convert to WAV bytes
    audio_np = audio.cpu().float().numpy()
    if audio_np.ndim == 3:
        audio_np = audio_np[0]  # Remove batch dim
    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, 0)

    sr = getattr(pipe, "sample_rate", 48000)
    wav_bytes = _to_wav(audio_np, sr)

    return {
        "audio": base64.b64encode(wav_bytes).decode(),
        "sample_rate": sr,
        "duration_s": seconds,
        "generation_time_s": round(elapsed, 2),
        "model": model,
        "prompt": prompt,
    }


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float numpy array to WAV bytes."""
    import wave

    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        n_channels = audio_int16.shape[0]
        wav.setnchannels(n_channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.T.tobytes())

    return buf.getvalue()


class MossHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "loaded_model": _loaded_model,
                "available_models": list(MOSS_MODELS.keys()),
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            if self.path == "/generate":
                prompt = body.get("prompt", "")
                if not prompt:
                    self._send_json(400, {"error": "no prompt"})
                    return

                model = body.get("model", "moss-soundeffect")
                result = generate_sound(
                    model=model,
                    prompt=prompt,
                    seconds=float(body.get("seconds", 10.0)),
                    seed=int(body.get("seed", 0)),
                    steps=int(body.get("steps", 100)),
                    cfg=float(body.get("cfg", 4.0)),
                )
                self._send_json(200, result)

            elif self.path == "/release":
                unload_model()
                self._send_json(200, {"status": "released"})

            elif self.path == "/load":
                model = body.get("model", "")
                load_model(model)  # Pre-load a model
                self._send_json(200, {"status": "loaded", "model": model})

            else:
                self._send_json(404, {"error": "not found"})

        except Exception as e:
            logger.exception("Request failed")
            self._send_json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)


if __name__ == "__main__":
    logger.info("MOSS audio server starting on port %d", PORT)
    logger.info("Code path: %s", MOSS_CODE_PATH)
    logger.info("Available models: %s", list(MOSS_MODELS.keys()))
    server = HTTPServer(("0.0.0.0", PORT), MossHandler)
    server.serve_forever()
