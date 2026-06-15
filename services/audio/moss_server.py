"""Standalone MOSS audio server with model switching.

Wraps MossSoundEffectPipeline as a simple HTTP API.
Runs in a separate Docker container — no Wan2GP.

POST /generate
  { "prompt": "rain on tin roof", "model": "moss-soundeffect-v2", "seconds": 3, "seed": 42 }
  → { "audio": "<base64 wav>", "sample_rate": 48000 }

POST /release  → Unloads model, frees VRAM
POST /load     → Pre-load a specific model
GET /health    → { "status": "ok", "loaded_model": "..." }
"""
from __future__ import annotations

import base64, gc, io, logging, os, sys, time
import torch, numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("moss-server")

# MOSS pipeline code — parent dir so package imports work
MOSS_CODE_PARENT = os.environ.get("MOSS_CODE_PARENT", "/opt/wan2gp")
if os.path.exists(MOSS_CODE_PARENT):
    sys.path.insert(0, MOSS_CODE_PARENT)

PORT = int(os.environ.get("MOSS_PORT", "8081"))

# Model registry: name → (path, pipeline_class)
# All models use MossSoundEffectPipeline unless noted.
# Paths match config/model_registry.yaml — sync when adding models.
MOSS_MODELS = {
    # v2 — self-contained 1.3B DiT + Qwen3-1.7B + DAC (tested, works)
    "moss-soundeffect-v2": ("/models/audio/moss-soundeffect-v2/", "MossSoundEffectPipeline"),

    # v1 — 8B DiT + Qwen3.6 + DAC. Requires moss-audio-tokenizer.
    "moss-soundeffect": ("/models/audio/moss-soundeffect/bf16/", "MossSoundEffectPipeline"),
    "moss-tts": ("/models/audio/moss-tts/", "MossSoundEffectPipeline"),
    "moss-ttsd": ("/models/audio/moss-ttsd/", "MossSoundEffectPipeline"),

    # Voice generator — 7GB, same pipeline family
    "moss-voicegenerator": ("/models/audio/moss-voicegenerator/", "MossSoundEffectPipeline"),

    # Nano — GPT2-based, DIFFERENT pipeline (will fail with MossSoundEffectPipeline)
    "moss-tts-nano": ("/models/audio/moss-tts-nano/", "MossSoundEffectPipeline"),

    # Realtime and local-transformer — may use different pipelines
    "moss-tts-realtime": ("/models/audio/moss-tts-realtime/", "MossSoundEffectPipeline"),
    "moss-tts-local-transformer": ("/models/audio/moss-tts-local-transformer/", "MossSoundEffectPipeline"),
}

_pipeline = None
_loaded_model: str | None = None


def load_model(model_name: str):
    """Load a MOSS pipeline. Switches if a different model is loaded."""
    global _pipeline, _loaded_model

    if _pipeline is not None and _loaded_model == model_name:
        return _pipeline

    if _pipeline is not None:
        logger.info("MOSS: switching from '%s' to '%s'", _loaded_model, model_name)
        unload_model()

    model_path, pipe_cls_name = MOSS_MODELS.get(model_name, (None, None))
    if model_path is None:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MOSS_MODELS.keys())}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {model_path}.\n"
            f"  Download with: python3 scripts/download_moss_models.py --only {model_name.split('-')[-1]}\n"
            f"  Or: snapshot_download('OpenMOSS-Team/MOSS-SoundEffect-v2.0', local_dir='{model_path}')"
        )

    logger.info("MOSS: loading '%s' from %s", model_name, model_path)

    from moss_soundeffect_v2.pipeline_moss_soundeffect import MossSoundEffectPipeline

    _pipeline = MossSoundEffectPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    _loaded_model = model_name
    logger.info("MOSS: '%s' loaded (%dMB VRAM)", model_name,
                torch.cuda.memory_allocated(0) // (1024*1024))
    return _pipeline


def unload_model():
    """Unload current model and free VRAM."""
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
    logger.info("MOSS: VRAM freed")


def generate_sound(model: str, prompt: str, seconds: float = 3.0,
                   seed: int = 0, steps: int = 50, cfg: float = 4.0) -> dict:
    """Generate sound effect from text prompt."""
    pipe = load_model(model)
    t0 = time.perf_counter()
    output = pipe(prompt=prompt, seconds=seconds, num_inference_steps=steps,
                  cfg_scale=cfg, seed=seed)
    elapsed = time.perf_counter() - t0

    audio = output.audios if hasattr(output, "audios") else output
    audio_np = audio.cpu().float().numpy()
    if audio_np.ndim == 3: audio_np = audio_np[0]
    if audio_np.ndim == 1: audio_np = np.expand_dims(audio_np, 0)

    sr = getattr(pipe, "sample_rate", 48000)
    wav_bytes = _to_wav(audio_np, sr)
    return {
        "audio": base64.b64encode(wav_bytes).decode(),
        "sample_rate": sr, "duration_s": seconds,
        "generation_time_s": round(elapsed, 2),
        "model": model, "prompt": prompt,
    }


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    import wave
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(audio_int16.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.T.tobytes())
    return buf.getvalue()


class MossHandler(BaseHTTPRequestHandler):
    def _send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "loaded_model": _loaded_model,
                                  "available_models": list(MOSS_MODELS.keys())})
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
                    model=body.get("model", "moss-soundeffect-v2"),
                    prompt=prompt,
                    seconds=float(body.get("seconds", 3.0)),
                    seed=int(body.get("seed", 0)),
                    steps=int(body.get("steps", 50)),
                    cfg=float(body.get("cfg", 4.0)),
                )
                self._send_json(200, result)

            elif self.path == "/release":
                unload_model()
                self._send_json(200, {"status": "released"})

            elif self.path == "/load":
                load_model(body.get("model", ""))
                self._send_json(200, {"status": "loaded", "model": body.get("model", "")})
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as e:
            logger.exception("Request failed")
            self._send_json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)


if __name__ == "__main__":
    logger.info("MOSS server starting on port %d", PORT)
    logger.info("Code parent: %s", MOSS_CODE_PARENT)
    logger.info("Available models: %s", list(MOSS_MODELS.keys()))
    server = HTTPServer(("0.0.0.0", PORT), MossHandler)
    server.serve_forever()
