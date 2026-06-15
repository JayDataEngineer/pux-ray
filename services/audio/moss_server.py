"""Standalone MOSS audio server.

Wraps MossSoundEffectPipeline as a simple HTTP API.
Runs in a separate Docker container — no Wan2GP, no diffusers.

POST /generate
  { "prompt": "rain on tin roof", "seconds": 10, "seed": 0 }
  → { "audio": "<base64 wav>", "sample_rate": 48000 }

POST /release
  → Unloads model, frees VRAM

GET /health
  → { "status": "ok" }
"""
from __future__ import annotations

import base64
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

MODEL_PATH = os.environ.get("MOSS_MODEL_PATH", "/models/audio/moss-soundeffect")
PORT = int(os.environ.get("MOSS_PORT", "8081"))

_pipeline = None


def load_model():
    """Load the MOSS pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading MOSS pipeline from %s", MODEL_PATH)

    from pipeline_moss_soundeffect import MossSoundEffectPipeline

    _pipeline = MossSoundEffectPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    logger.info("MOSS pipeline loaded")
    return _pipeline


def generate_sound(prompt: str, seconds: float = 10.0, seed: int = 0,
                   steps: int = 100, cfg: float = 4.0) -> dict:
    """Generate sound effect from text prompt."""
    pipe = load_model()

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

    sr = getattr(_pipeline, "sample_rate", 48000)
    wav_bytes = _to_wav(audio_np, sr)

    return {
        "audio": base64.b64encode(wav_bytes).decode(),
        "sample_rate": sr,
        "duration_s": seconds,
        "generation_time_s": round(elapsed, 2),
        "prompt": prompt,
    }


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float numpy array to WAV bytes."""
    import wave
    import struct

    # Normalize to int16
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
            self._send_json(200, {"status": "ok", "loaded": _pipeline is not None})
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

                result = generate_sound(
                    prompt=prompt,
                    seconds=float(body.get("seconds", 10.0)),
                    seed=int(body.get("seed", 0)),
                    steps=int(body.get("steps", 100)),
                    cfg=float(body.get("cfg", 4.0)),
                )
                self._send_json(200, result)

            elif self.path == "/release":
                global _pipeline
                _pipeline = None
                torch.cuda.empty_cache()
                self._send_json(200, {"status": "released"})

            else:
                self._send_json(404, {"error": "not found"})

        except Exception as e:
            logger.exception("Request failed")
            self._send_json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)


if __name__ == "__main__":
    logger.info("MOSS audio server starting on port %d", PORT)
    logger.info("Model path: %s", MODEL_PATH)
    logger.info("Code path: %s", MOSS_CODE_PATH)
    server = HTTPServer(("0.0.0.0", PORT), MossHandler)
    server.serve_forever()
