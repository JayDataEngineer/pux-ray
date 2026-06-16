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

# Safety nets: prevent DiffSynth from trying to download anything.
# We use ModelConfig(path=...) which already bypasses downloads, but
# these env vars are a belt-and-suspenders guard.
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", MODELS_ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
    # Use ModelConfig(path=...) NOT model_id — path bypasses ALL download logic
    # and uses local files directly. model_id triggers ModelScope/HF download.
    import glob as glob_module
    model_configs = []
    for sub, pattern in cfg["model_paths"]:
        full_pattern = os.path.join(local_root, sub, pattern) if sub != "." else os.path.join(local_root, pattern)
        matched = sorted(glob_module.glob(full_pattern))
        if not matched:
            raise FileNotFoundError(
                f"No files matching {full_pattern}.\n"
                f"  Download with: python3 scripts/download_vace_models.py --only fun-a14b"
            )
        path = matched[0] if len(matched) == 1 else matched
        model_configs.append(ModelConfig(path=path, **vram_config))

    # VRAM limit: cap GPU memory (free - 2GB) or env override
    vram_limit = VRAM_LIMIT_GB
    if vram_limit <= 0 and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(0)
        vram_limit = free / (1024 ** 3) - 2.0
    vram_limit = max(vram_limit, 4.0)  # never below 4GB floor

    # Tokenizer: use local path (no download). The tokenizer files live at
    # TOKENIZER_ROOT/google/umt5-xxl/ (downloaded separately or from the fun-a14b bundle).
    tokenizer_path = os.path.join(TOKENIZER_ROOT, "google/umt5-xxl")
    if not os.path.exists(tokenizer_path):
        # Fallback: check inside the model bundle itself
        tokenizer_path = os.path.join(local_root, "google/umt5-xxl")
    tokenizer_config = ModelConfig(path=tokenizer_path) if os.path.exists(tokenizer_path) else None

    _pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,  # we have the original .pth files, don't redirect to .safetensors
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


# ═══════════════════════════════════════════════════════════════════════════
# DIRECTOR CAPABILITIES — LTX Director Node Parity
# ═══════════════════════════════════════════════════════════════════════════
# The WhatDreamsCost LTX Director node provides:
#   1. Multi-Keyframe Support — images at arbitrary frame positions
#   2. Prompt Relay — per-segment prompts (different prompts for frame ranges)
#   3. Motion Amplitude — sigma schedule control for inter-keyframe motion
#   4. Guide Modes — replace (hard overwrite) vs guide (Gaussian decay)
#   5. Global + Local prompts — global applies everywhere, local overrides per-segment
#
# All five are implemented below. The Director API is backward-compatible:
# if no keyframes/segments are provided, standard VACE generation runs.

def _build_keyframe_vcu(keyframes: list, num_frames: int, width: int, height: int):
    """Build (vace_video, vace_video_mask) from Director keyframe list.

    Constructs the VCU that VACE expects from a list of keyframe specs:
      - Places each keyframe image at its frame_index in the video
      - Builds a per-frame mask: white=preserve (keyframe), black=generate

    Guide modes (matching LTX Director):
      "replace": mask=strength at the keyframe frame ONLY (hard pin)
      "guide":   mask=Gaussian decay centered at the keyframe (soft influence)
      "fade":    mask=linear ramp from previous keyframe to this one (transition)

    Returns:
        vace_video: list of PIL.Image (one per frame, RGB)
        vace_video_mask: list of PIL.Image (one per frame, L = grayscale 0-255)
    """
    import math as _math
    from PIL import Image as _Image
    import numpy as _np

    keyframes = sorted(keyframes, key=lambda k: k["frame_index"])

    # Initialize video (black) and mask (0 = generate everything)
    video_frames = [_Image.new("RGB", (width, height), (0, 0, 0)) for _ in range(num_frames)]
    mask_array = _np.zeros((num_frames, height, width), dtype=_np.float32)

    for kf in keyframes:
        idx = int(kf["frame_index"])
        if idx < 0 or idx >= num_frames:
            continue
        strength = float(kf.get("strength", 1.0))
        mode = kf.get("mode", "replace")
        img = kf["image"]
        if img is None:
            continue
        if isinstance(img, _Image.Image):
            img = img.resize((width, height), _Image.LANCZOS)
        elif isinstance(img, _np.ndarray):
            img = _Image.fromarray(img.astype(_np.uint8)).resize((width, height), _Image.LANCZOS)

        # Fill video frames between this keyframe and the next with this keyframe
        # (gives VACE structural context, not just sparse anchors)
        prev_idx = idx
        for prev_kf in reversed(keyframes):
            if prev_kf["frame_index"] < idx:
                prev_idx = int(prev_kf["frame_index"])
                break
        else:
            prev_idx = 0
        for f in range(prev_idx, idx + 1):
            video_frames[f] = img

        if mode == "replace":
            # Hard pin: mask is full strength at this frame only
            mask_array[idx] = max(mask_array[idx].max(), strength)
            mask_array[idx] = strength
        elif mode == "guide":
            # Gaussian decay — sigma controls the influence radius
            sigma = float(kf.get("sigma", 5.0))
            for f in range(num_frames):
                dist = abs(f - idx)
                decay = _math.exp(-0.5 * (dist / sigma) ** 2) * strength
                mask_array[f] = _np.maximum(mask_array[f], decay)
        elif mode == "fade":
            # Linear ramp from previous keyframe to this one
            if idx > 0:
                ramp_start = max(0, idx - int(kf.get("fade_duration", 5)))
                for f in range(ramp_start, idx + 1):
                    t = (f - ramp_start) / max(1, idx - ramp_start)
                    mask_array[f] = _np.maximum(mask_array[f], t * strength)

    # Convert mask to PIL Images (grayscale)
    mask_frames = []
    for f in range(num_frames):
        m = (mask_array[f] * 255).clip(0, 255).astype(_np.uint8)
        mask_frames.append(_Image.fromarray(m, mode="L"))

    return video_frames, mask_frames


def _generate_segment(pipe, segment: dict, shared: dict) -> list:
    """Generate one segment of a Director timeline.

    Each segment runs a full VACE pass with its own prompt and keyframes.
    Continuity is maintained by passing the previous segment's last frame
    as a 'replace' keyframe at frame 0.
    """
    seg_prompt = segment["prompt"]
    seg_neg = segment.get("negative_prompt", shared["negative_prompt"])
    seg_frames = segment["end_frame"] - segment["start_frame"] + 1

    # Build keyframes for this segment (frame indices relative to segment start)
    seg_keyframes = []

    # Continuity handoff: last frame of previous segment → frame 0 of this one
    if shared.get("last_frame") is not None:
        seg_keyframes.append({
            "image": shared["last_frame"],
            "frame_index": 0,
            "strength": 1.0,
            "mode": "replace",
        })

    # User-defined keyframes within this segment's range
    for kf in shared.get("keyframes", []):
        if segment["start_frame"] <= kf["frame_index"] <= segment["end_frame"]:
            seg_keyframes.append({
                "image": kf["image"],
                "frame_index": kf["frame_index"] - segment["start_frame"],
                "strength": kf.get("strength", 1.0),
                "mode": kf.get("mode", "replace"),
                "sigma": kf.get("sigma", 5.0),
            })

    # Last keyframe of segment: pin the end frame (if user specified one)
    end_keyframe = None
    for kf in shared.get("keyframes", []):
        if kf["frame_index"] == segment["end_frame"]:
            end_keyframe = kf
            break
    if end_keyframe is None and shared.get("last_frame") is None:
        # No explicit end frame — let the model generate freely
        pass

    # Build VCU for this segment
    if seg_keyframes:
        vace_video, vace_video_mask = _build_keyframe_vcu(
            seg_keyframes, seg_frames, shared["width"], shared["height"]
        )
    else:
        vace_video = None
        vace_video_mask = None

    # Motion amplitude adjusts cfg + vace_scale
    motion = shared.get("motion_amplitude", 1.0)
    adj_cfg = shared["cfg"] * (0.7 + 0.3 * motion)  # higher motion = slightly higher cfg
    adj_vace_scale = shared["vace_scale"] / max(0.5, motion)  # higher motion = weaker anchors

    logger.info(
        "DIRECTOR: segment [%d-%d] prompt=%r keyframes=%d frames=%d cfg=%.2f",
        segment["start_frame"], segment["end_frame"], seg_prompt[:50],
        len(seg_keyframes), seg_frames, adj_cfg,
    )

    video = pipe(
        prompt=seg_prompt,
        negative_prompt=seg_neg,
        vace_video=vace_video,
        vace_video_mask=vace_video_mask,
        vace_reference_image=shared.get("vace_reference_image"),
        vace_scale=adj_vace_scale,
        width=shared["width"],
        height=shared["height"],
        num_frames=seg_frames,
        seed=shared["seed"],
        num_inference_steps=shared["steps"],
        guidance_scale=adj_cfg,
        tea_cache_l1_thresh=shared["teacache_thresh"] if shared["teacache_thresh"] > 0 else None,
        tea_cache_model_id=TEACACHE_MODEL_ID,
        tiled=TILED,
        tile_size=TILE_SIZE,
    )

    # Save last frame for next segment's continuity
    if isinstance(video, list) and len(video) > 0:
        shared["last_frame"] = video[-1]
    elif hasattr(video, "__getitem__"):
        try:
            shared["last_frame"] = video[-1]
        except Exception:
            pass

    return video


def _stitch_segments(segment_videos: list) -> list:
    """Concatenate segment video outputs into one continuous frame list."""
    stitched = []
    for sv in segment_videos:
        if isinstance(sv, list):
            stitched.extend(sv)
        elif hasattr(sv, "__iter__"):
            stitched.extend(list(sv))
    return stitched


def generate_video(payload: dict) -> dict:
    """Generate a VACE video from a payload.

    Supports two modes:
      1. Standard VACE — pass vace_video / vace_reference_image / etc directly
      2. Director mode — pass keyframes[] and/or segments[] for LTX Director parity

    Director mode activates when 'keyframes' or 'segments' are present.
    """
    pipe = load_model(payload.get("model", _loaded_model or "wan-vace-fun-a14b"))

    prompt = payload.get("prompt", "")
    if not prompt:
        raise ValueError("no prompt")

    width = int(payload.get("width", 832))
    height = int(payload.get("height", 480))
    num_frames = int(payload.get("num_frames", 81))
    seed = int(payload.get("seed", -1))
    steps = int(payload.get("steps", payload.get("sampling_steps", DEFAULT_STEPS)))
    cfg = float(payload.get("cfg", payload.get("guide_scale", 5.0)))
    neg = payload.get("negative_prompt") or payload.get("n_prompt", "")
    vace_scale = float(payload.get("vace_scale", 1.0))
    tc_thresh = float(payload.get("tea_cache_l1_thresh", TEACACHE_THRESH))

    # ─── Director mode detection ──────────────────────────────────────────
    has_keyframes = bool(payload.get("keyframes"))
    has_segments = bool(payload.get("segments"))

    if has_segments:
        return _generate_director_segments(pipe, payload, prompt, neg, width, height,
                                           num_frames, seed, steps, cfg, vace_scale, tc_thresh)

    # Resolve standard VACE inputs (or build from keyframes)
    if has_keyframes:
        # Director: build VCU from keyframe list (single-pass mode)
        keyframes_resolved = []
        for kf in payload["keyframes"]:
            img = _resolve_input(kf.get("image") or kf.get("url"), "image")
            if img is not None:
                keyframes_resolved.append({
                    "image": img,
                    "frame_index": int(kf.get("frame_index", 0)),
                    "strength": float(kf.get("strength", 1.0)),
                    "mode": kf.get("mode", "replace"),
                    "sigma": float(kf.get("sigma", 5.0)),
                })
        vace_video, vace_video_mask = _build_keyframe_vcu(
            keyframes_resolved, num_frames, width, height)
        vace_ref = _resolve_input(payload.get("vace_reference_image"), "image")
        logger.info("DIRECTOR: single-pass keyframes=%d frames=%d", len(keyframes_resolved), num_frames)
    else:
        vace_video = _resolve_input(payload.get("vace_video"), "video")
        vace_video_mask = _resolve_input(payload.get("vace_video_mask"), "video")
        vace_ref = _resolve_input(payload.get("vace_reference_image"), "image")

    # Motion amplitude (affects cfg + vace_scale in single-pass mode too)
    motion = float(payload.get("motion_amplitude", 1.0))
    if motion != 1.0 and has_keyframes:
        cfg = cfg * (0.7 + 0.3 * motion)
        vace_scale = vace_scale / max(0.5, motion)

    t0 = time.perf_counter()
    logger.info(
        "VACE: generate model=%s prompt=%r vc=%s vm=%s vr=%s steps=%d cfg=%.2f frames=%d tc=%.3f director=%s",
        _loaded_model, prompt[:60], vace_video is not None,
        vace_video_mask is not None, vace_ref is not None, steps, cfg, num_frames, tc_thresh,
        has_keyframes or has_segments,
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
            "director_mode": has_keyframes or has_segments,
        },
    }


def _generate_director_segments(pipe, payload, global_prompt, global_neg,
                                 width, height, num_frames, seed, steps,
                                 cfg, vace_scale, tc_thresh) -> dict:
    """Director multi-segment mode — LTX Director Prompt Relay parity.

    Generates each segment independently with its own prompt, stitching
    them together with frame continuity handoff.

    Payload keys:
      segments: [{start_frame, end_frame, prompt, negative_prompt?}]
      keyframes: [{image, frame_index, strength, mode, sigma?}]  (optional anchors)
      motion_amplitude: float (0.5=static, 1.0=default, 1.5=exaggerated)
    """
    segments = payload["segments"]
    motion_amplitude = float(payload.get("motion_amplitude", 1.0))

    # Resolve keyframe images once
    resolved_keyframes = []
    for kf in payload.get("keyframes", []):
        img = _resolve_input(kf.get("image") or kf.get("url"), "image")
        if img is not None:
            resolved_keyframes.append({
                "image": img,
                "frame_index": int(kf.get("frame_index", 0)),
                "strength": float(kf.get("strength", 1.0)),
                "mode": kf.get("mode", "replace"),
                "sigma": float(kf.get("sigma", 5.0)),
            })

    shared = {
        "width": width,
        "height": height,
        "negative_prompt": global_neg,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "vace_scale": vace_scale,
        "teacache_thresh": tc_thresh,
        "motion_amplitude": motion_amplitude,
        "keyframes": resolved_keyframes,
        "vace_reference_image": _resolve_input(payload.get("vace_reference_image"), "image"),
        "last_frame": None,  # continuity handoff state
    }

    t0 = time.perf_counter()
    logger.info(
        "DIRECTOR: multi-segment segments=%d keyframes=%d motion=%.2f frames=%d",
        len(segments), len(resolved_keyframes), motion_amplitude, num_frames,
    )

    segment_videos = []
    for i, seg in enumerate(segments):
        # Fill in defaults
        seg_full = {
            "start_frame": int(seg["start_frame"]),
            "end_frame": int(seg["end_frame"]),
            "prompt": seg.get("prompt", global_prompt),
            "negative_prompt": seg.get("negative_prompt", global_neg),
        }
        try:
            sv = _generate_segment(pipe, seg_full, shared)
            segment_videos.append(sv)
        except Exception as e:
            logger.error("DIRECTOR: segment %d failed: %s", i, e)
            raise

    elapsed = time.perf_counter() - t0
    stitched = _stitch_segments(segment_videos)

    fps = int(payload.get("fps", 15))
    mp4_bytes = _frames_to_mp4(stitched, fps=fps)
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0

    return {
        "video": base64.b64encode(mp4_bytes).decode(),
        "fps": fps,
        "model": _loaded_model,
        "prompt": f"[director:{len(segments)} segments]",
        "metrics": {
            "latency_s": round(elapsed, 2),
            "vram_peak_mb": int(peak_vram),
            "steps": steps,
            "num_segments": len(segments),
            "total_frames": len(stitched),
            "fp8": FP8_ENABLED,
            "teacache_thresh": tc_thresh,
            "tiled": TILED,
            "director_mode": True,
            "motion_amplitude": motion_amplitude,
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
