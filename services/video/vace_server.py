"""Standalone VACE video server — DiffSynth-Studio pipeline + TeaCache + SageAttention.

Why this exists (and doesn't use SGLang):
  SGLang's diffusion engine compiles a rigid CUDA graph for standard T2V/I2V tensor
  shapes. VACE requires a Video Condition Unit (VCU): masked source-video latents +
  alpha mask channel + reference layout frames. SGLang has no input slots for those
  extra channels. DiffSynth-Studio handles VCU natively AND ships TeaCache, tiled
  VAE, and AutoWrappedModule (mmGP-equivalent block-level VRAM streaming).

Architecture mirrors moss_server.py:
  - stdlib http.server (no FastAPI dep)
  - one pipeline per process (Python GIL — single-request-at-a-time)
  - /health, /load, /release, /generate endpoints

Optimization levers (env vars):
  VACE_FP8=1                 → FP8 e4m3fn CPU storage + BF16 GPU compute (~50% RAM/PCIe cut)
  VACE_TEACACHE_THRESH=0.15  → TeaCache L1 threshold (0 disables; 0.10-0.20 typical)
  VACE_TEACACHE_MODEL_ID=…   → coefficient set: Wan2.1-T2V-14B | Wan2.1-I2V-14B-480P | ...
  VACE_ATTENTION=sage_attention | flash_attention_3 | flash_attention_2 | torch
  VACE_VRAM_LIMIT_GB=…       → cap VRAM usage (default: free - 2GB)
  VACE_TILED=1               → tiled VAE decode (default on — set 0 to disable)
  VACE_TILE_SIZE=30,52       → tile (frames, spatial) for VAE
  VACE_DEFAULT_STEPS=18      → full MoE base; do NOT distill below 12

POST /generate:
  {
    "prompt": "...", "negative_prompt": "...",
    "vace_video": "<url or base64 mp4>",        # depth/pose/layout control video
    "vace_video_mask": "<url or base64 mp4>",   # alpha mask for inpainting/outpainting
    "vace_reference_image": "<url or base64>",  # subject reference for identity preservation
    "vace_scale": 1.0,                          # control strength (0.5-1.0 typical)
    "width": 832, "height": 480, "num_frames": 81,
    "seed": -1, "steps": 18, "cfg": 5.0,
    "tea_cache_l1_thresh": 0.15
  }
  → { "video": "<base64 mp4>", "fps": 15, "metrics": {...} }
"""
from __future__ import annotations

import base64, gc, io, logging, os, sys, time
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("vace-server")

# ── Configuration ────────────────────────────────────────────────────────────
PORT = int(os.environ.get("VACE_PORT", "8082"))
MODELS_ROOT = os.environ.get("VACE_MODELS_ROOT", "/models/video")
TOKENIZER_ROOT = os.environ.get("VACE_TOKENIZER_ROOT", "/models/video/wan-vace-tokenizer")

# Optimization env vars (defaults calibrated for RTX 4090 24GB)
FP8_ENABLED = os.environ.get("VACE_FP8", "0") == "1"
TEACACHE_THRESH = float(os.environ.get("VACE_TEACACHE_THRESH", "0.15"))
TEACACHE_MODEL_ID = os.environ.get("VACE_TEACACHE_MODEL_ID", "Wan2.1-T2V-14B")
ATTENTION_IMPL = os.environ.get("VACE_ATTENTION", "")  # empty = auto
VRAM_LIMIT_GB = float(os.environ.get("VACE_VRAM_LIMIT_GB", "0"))  # 0 = auto
TILED = os.environ.get("VACE_TILED", "1") == "1"
TILE_SIZE = tuple(int(x) for x in os.environ.get("VACE_TILE_SIZE", "30,52").split(","))
DEFAULT_STEPS = int(os.environ.get("VACE_DEFAULT_STEPS", "18"))

# Allow SageAttention / FlashAttn override before DiffSynth imports
if ATTENTION_IMPL:
    os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = ATTENTION_IMPL

# Model registry: short-name → (local_path, tokenizer_path, pipeline_layout)
# Layout matches scripts/download_vace_models.py.
VACE_MODELS = {
    # Wan2.2 VACE-Fun A14B — modular MoE (primary production target)
    "wan-vace-fun-a14b": {
        "model_paths": [
            ("high_noise_model", "diffusion_pytorch_model*.safetensors"),
            ("low_noise_model",  "diffusion_pytorch_model*.safetensors"),
            (".",                "models_t5_umt5-xxl-enc-bf16.pth"),
            (".",                "Wan2.1_VAE.pth"),
        ],
        "model_id_template": "alibaba-pai/Wan2.2-VACE-Fun-A14B",  # for TeaCache coeffs
        "is_moe": True,
    },
    # Wan2.1 VACE 14B monolithic (alternative)
    "wan-vace-14b": {
        "model_paths": [
            (".", "diffusion_pytorch_model*.safetensors"),
            (".", "models_t5_umt5-xxl-enc-bf16.pth"),
            (".", "Wan2.1_VAE.pth"),
        ],
        "model_id_template": "Wan-AI/Wan2.1-VACE-14B",
        "is_moe": False,
    },
    # Wan2.1 VACE 1.3B efficiency tier
    "wan-vace-1.3b": {
        "model_paths": [
            (".", "diffusion_pytorch_model*.safetensors"),
            (".", "models_t5_umt5-xxl-enc-bf16.pth"),
            (".", "Wan2.1_VAE.pth"),
        ],
        "model_id_template": "Wan2.1-T2V-1.3B",  # closest TeaCache coeff set
        "is_moe": False,
    },
}

# ── Pipeline singleton ───────────────────────────────────────────────────────
_pipe = None
_loaded_model: str | None = None


def _build_vram_config() -> dict:
    """Build the DiffSynth VRAM config from env vars.

    Default: BF16 CPU offload (matches DiffSynth official low_vram example).
    VACE_FP8=1: switches offload/onload to FP8 e4m3fn — halves CPU RAM and PCIe
    bandwidth at the cost of dequantization overhead during block streaming.
    """
    base = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    if FP8_ENABLED:
        # Store offloaded weights as FP8 e4m3fn on CPU. Compute stays BF16
        # (FP8 compute requires TransformerEngine / Hopper-specific kernels
        # and the diffusers Wan DiT doesn't expose enable_layerwise_casting
        # for true FP8 GEMM on Ada Lovelace).
        base["offload_dtype"] = torch.float8_e4m3fn
        base["onload_dtype"] = torch.float8_e4m3fn
        logger.info("VRAM: FP8 e4m3fn CPU storage enabled (~50%% RAM/PCIe reduction)")
    return base


def load_model(model_name: str):
    """Load a VACE pipeline. Switches if a different model is loaded."""
    global _pipe, _loaded_model

    if _pipe is not None and _loaded_model == model_name:
        return _pipe

    if _pipe is not None:
        logger.info("VACE: switching from '%s' to '%s'", _loaded_model, model_name)
        unload_model()

    if model_name not in VACE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(VACE_MODELS.keys())}"
        )

    cfg = VACE_MODELS[model_name]
    # Directory name under /models/video/ matches the registry key verbatim
    # (see scripts/download_vace_models.py VACE_MODELS mapping).
    local_root = os.path.join(MODELS_ROOT, model_name)

    if not os.path.exists(local_root):
        # Map registry name → download-script short-name for the helpful hint
        short = model_name.replace("wan-vace-", "").replace("wan-", "") or "fun-a14b"
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {local_root}.\n"
            f"  Download with: python3 scripts/download_vace_models.py --only {short} tokenizer"
        )

    logger.info("VACE: loading '%s' from %s", model_name, local_root)
    logger.info("  FP8=%s  TeaCache=%s (thresh=%.3f)  Tiled=%s  Attention=%s",
                FP8_ENABLED, TEACACHE_THRESH > 0, TEACACHE_THRESH, TILED,
                ATTENTION_IMPL or "auto")

    # DiffSynth-Studio import — fail clearly if not installed in the image
    try:
        from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    except ImportError as e:
        raise RuntimeError(
            "DiffSynth-Studio not installed. Add to Dockerfile.vace:\n"
            "  pip install diffsynth   OR   pip install git+https://github.com/modelscope/DiffSynth-Studio"
        ) from e

    vram_config = _build_vram_config()
    model_configs = [
        ModelConfig(
            model_id=local_root,
            origin_file_pattern=os.path.join(sub, pattern),
            **vram_config,
        )
        for sub, pattern in cfg["model_paths"]
    ]

    # VRAM limit: cap GPU memory (free - 2GB) or env override
    vram_limit = VRAM_LIMIT_GB
    if vram_limit <= 0 and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(0)
        vram_limit = free / (1024 ** 3) - 2.0
    vram_limit = max(vram_limit, 4.0)  # never below 4GB floor

    _pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=ModelConfig(
            model_id=TOKENIZER_ROOT,
            origin_file_pattern="google/umt5-xxl/",
        ) if os.path.exists(TOKENIZER_ROOT) else None,
        vram_limit=vram_limit,
    )
    _loaded_model = model_name
    vram_mb = torch.cuda.memory_allocated(0) // (1024 * 1024) if torch.cuda.is_available() else 0
    logger.info("VACE: '%s' loaded (%dMB VRAM, limit=%.1fGB)",
                model_name, vram_mb, vram_limit)
    return _pipe


def unload_model():
    """Unload current model and free VRAM."""
    global _pipe, _loaded_model
    if _pipe is not None:
        logger.info("VACE: unloading '%s'", _loaded_model)
        del _pipe
        _pipe = None
        _loaded_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    logger.info("VACE: VRAM freed")


def _resolve_input(spec, kind: str):
    """Resolve an input spec (URL or base64) to a numpy array / PIL Image / VideoData.

    For vace_video / vace_video_mask: returns a diffsynth VideoData.
    For vace_reference_image: returns a PIL Image.
    None in → None out.
    """
    if spec is None:
        return None
    from PIL import Image
    from diffsynth.utils.data import VideoData

    if isinstance(spec, str) and spec.startswith(("http://", "https://")):
        url = spec
        if kind == "image":
            import urllib.request
            with urllib.request.urlopen(url, timeout=30) as r:
                return Image.open(io.BytesIO(r.read()))
        # video: download to temp file, hand to VideoData
        import tempfile, urllib.request
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            with urllib.request.urlopen(url, timeout=120) as r:
                tf.write(r.read())
            return VideoData(tf.name, height=480, width=832)  # resized inside
    elif isinstance(spec, str):
        # base64-encoded
        raw = base64.b64decode(spec)
        if kind == "image":
            return Image.open(io.BytesIO(raw))
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(raw)
        return VideoData(tf.name, height=480, width=832)
    return spec  # already an image/video object


def generate_video(payload: dict) -> dict:
    """Generate a VACE video from a payload."""
    pipe = load_model(payload.get("model", _loaded_model or "wan-vace-fun-a14b"))

    from diffsynth.utils.data import save_video

    prompt = payload.get("prompt", "")
    if not prompt:
        raise ValueError("no prompt")

    # Resolve multimodal inputs (lazy — only if provided)
    vace_video = _resolve_input(payload.get("vace_video"), "video")
    vace_video_mask = _resolve_input(payload.get("vace_video_mask"), "video")
    vace_ref = _resolve_input(payload.get("vace_reference_image"), "image")

    width = int(payload.get("width", 832))
    height = int(payload.get("height", 480))
    num_frames = int(payload.get("num_frames", 81))
    seed = int(payload.get("seed", -1))
    steps = int(payload.get("steps", payload.get("sampling_steps", DEFAULT_STEPS)))
    cfg = float(payload.get("cfg", payload.get("guide_scale", 5.0)))
    neg = payload.get("negative_prompt") or payload.get("n_prompt", "")
    vace_scale = float(payload.get("vace_scale", 1.0))

    # Per-request TeaCache override (server default applies if not set)
    tc_thresh = float(payload.get("tea_cache_l1_thresh", TEACACHE_THRESH))

    t0 = time.perf_counter()
    logger.info(
        "VACE: generate model=%s prompt=%r vc=%s vm=%s vr=%s steps=%d cfg=%.2f frames=%d tc=%.3f",
        _loaded_model, prompt[:60], vace_video is not None,
        vace_video_mask is not None, vace_ref is not None, steps, cfg, num_frames, tc_thresh,
    )

    video = pipe(
        prompt=prompt,
        negative_prompt=neg,
        vace_video=vace_video,
        vace_video_mask=vace_video_mask,
        vace_reference_image=vace_ref,
        vace_scale=vace_scale,
        width=width,
        height=height,
        num_frames=num_frames,
        seed=seed,
        num_inference_steps=steps,
        guidance_scale=cfg,
        tea_cache_l1_thresh=tc_thresh if tc_thresh > 0 else None,
        tea_cache_model_id=TEACACHE_MODEL_ID,
        tiled=TILED,
        tile_size=TILE_SIZE,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0

    # Encode frames → mp4 bytes
    fps = int(payload.get("fps", 15))
    mp4_bytes = _frames_to_mp4(video, fps=fps)
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0

    return {
        "video": base64.b64encode(mp4_bytes).decode(),
        "fps": fps,
        "model": _loaded_model,
        "prompt": prompt,
        "metrics": {
            "latency_s": round(elapsed, 2),
            "vram_peak_mb": int(peak_vram),
            "steps": steps,
            "num_frames": num_frames,
            "fp8": FP8_ENABLED,
            "teacache_thresh": tc_thresh,
            "tiled": TILED,
        },
    }


def _frames_to_mp4(video, fps: int = 15, quality: int = 5) -> bytes:
    """Encode video frames (list of PIL or tensor) to MP4 bytes via DiffSynth save_video."""
    from diffsynth.utils.data import save_video
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        path = tf.name
    try:
        save_video(video, path, fps=fps, quality=quality)
        with open(path, "rb") as f:
            return f.read()
    finally:
        try: os.unlink(path)
        except OSError: pass


# ── HTTP handler ─────────────────────────────────────────────────────────────
class VaceHandler(BaseHTTPRequestHandler):
    def _send_json(self, code, data):
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
                "available_models": list(VACE_MODELS.keys()),
                "config": {
                    "fp8": FP8_ENABLED,
                    "teacache_thresh": TEACACHE_THRESH,
                    "teacache_model_id": TEACACHE_MODEL_ID,
                    "attention": ATTENTION_IMPL or "auto",
                    "tiled": TILED,
                    "default_steps": DEFAULT_STEPS,
                },
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            if self.path == "/generate":
                result = generate_video(body)
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
    logger.info("VACE server starting on port %d", PORT)
    logger.info("Models root: %s", MODELS_ROOT)
    logger.info("Tokenizer root: %s", TOKENIZER_ROOT)
    logger.info("Available models: %s", list(VACE_MODELS.keys()))
    logger.info("Optimizations: fp8=%s teacache=%.3f attention=%s tiled=%s",
                FP8_ENABLED, TEACACHE_THRESH, ATTENTION_IMPL or "auto", TILED)
    server = HTTPServer(("0.0.0.0", PORT), VaceHandler)
    server.serve_forever()
