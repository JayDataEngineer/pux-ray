"""Media Analysis MCP Server settings.

All settings use MEDIA_ prefix for environment variables.

Profiles (MEDIA_PROFILE):
  - minimal: No ML models. ColorThief, barcode, EXIF, PySceneDetect, Chromaprint only.
  - standard: + Florence-2, YOLOv8, WD14, Parakeet ASR, video. (default)
  - full: + InsightFace, NudeNet, PANNs, SAM 2, Grounding DINO, Kosmos-2.5. (heavy models)
  - all: Everything including Phi-4 vision, Pyannote (needs HF token).

Individual MEDIA_*_ENABLED flags override profile defaults.
"""

from pydantic_settings import BaseSettings


# Profile definitions: which services are enabled by default
PROFILES = {
    "minimal": {
        "color": True, "barcode": True, "exif": True,
        "scene": True, "fingerprint": True,
        "vision": False, "yolo": False, "tagger": False,
        "asr": False, "video": False,
        "face": False, "nsfw": False, "audio_classify": False,
        "segment": False, "pyannote": False,
        "grounding_dino": False, "phi4_vision": False, "kosmos": False,
    },
    "standard": {
        "color": True, "barcode": True, "exif": True,
        "scene": True, "fingerprint": True,
        "vision": True, "yolo": True, "tagger": True,
        "asr": True, "video": True,
        "face": False, "nsfw": False, "audio_classify": False,
        "segment": False, "pyannote": False,
        "grounding_dino": False, "phi4_vision": False, "kosmos": False,
    },
    "full": {
        "color": True, "barcode": True, "exif": True,
        "scene": True, "fingerprint": True,
        "vision": True, "yolo": True, "tagger": True,
        "asr": True, "video": True,
        "face": True, "nsfw": True, "audio_classify": True,
        "segment": True, "pyannote": False,
        "grounding_dino": True, "phi4_vision": False, "kosmos": True,
    },
    "all": {
        "color": True, "barcode": True, "exif": True,
        "scene": True, "fingerprint": True,
        "vision": True, "yolo": True, "tagger": True,
        "asr": True, "video": True,
        "face": True, "nsfw": True, "audio_classify": True,
        "segment": True, "pyannote": True,
        "grounding_dino": True, "phi4_vision": True, "kosmos": True,
    },
}


def _profile_enabled(profile: str, service: str) -> bool:
    """Look up whether a service is enabled in the given profile."""
    p = PROFILES.get(profile, PROFILES["standard"])
    return p.get(service, False)


class Settings(BaseSettings):
    model_config = {"env_prefix": "MEDIA_"}

    # Server
    host: str = "0.0.0.0"
    port: int = 8001

    # Device: "cpu" or "cuda" — controls all model inference
    device: str = "cpu"

    # Tool profile: "minimal", "standard", "full", "all"
    # Sets defaults for all *_enabled flags below.
    # Individual flags override the profile.
    profile: str = "standard"

    # Idle timeout: auto-unload models after this many seconds of no use.
    # Set to 0 to disable auto-unload.
    idle_timeout: int = 1800  # 30 minutes

    # Router (FunctionGemma) — always loaded at startup, not affected by profile
    router_enabled: bool = True
    router_model: str = "unsloth/functiongemma-270m-it-GGUF"
    router_filename: str = "functiongemma-270m-it-Q4_K_M.gguf"
    router_n_ctx: int = 512
    router_n_threads: int = 4
    router_timeout: float = 10.0

    # --- Service enables (profile sets defaults, env vars override) ---
    # Each uses a classmethod default that reads the profile.

    # Vision (Florence-2, ~900MB)
    vision_enabled: bool | None = None
    vision_model: str = "microsoft/Florence-2-base"
    vision_max_new_tokens: int = 1024
    vision_inference_timeout: float = 120.0

    # Object Detection (YOLOv8, ~6MB)
    yolo_enabled: bool | None = None
    yolo_model: str = "yolov8n.pt"

    # Image Tagging (WD14, ~300MB)
    tagger_enabled: bool | None = None
    tagger_model: str = "SmilingWolf/wd-v1-4-moat-tagger-v2"
    tagger_threshold: float = 0.35

    # ASR (Parakeet TDT v3, ~300MB)
    asr_enabled: bool | None = None
    asr_model: str = "nemo-parakeet-tdt-0.6b-v3"

    # Video (FFmpeg + SSIM, no model)
    video_enabled: bool | None = None
    video_max_frames: int = 10
    video_ssim_threshold: float = 0.85

    # Color extraction (ColorThief, no model)
    color_enabled: bool | None = None

    # Barcode / QR detection (pyzbar, no model)
    barcode_enabled: bool | None = None

    # EXIF metadata extraction (Pillow, no model)
    exif_enabled: bool | None = None

    # Scene detection (PySceneDetect, no model)
    scene_enabled: bool | None = None

    # Audio fingerprinting (Chromaprint, no model)
    fingerprint_enabled: bool | None = None

    # Face detection + recognition (InsightFace, ~350MB)
    face_enabled: bool | None = None
    face_model: str = "buffalo_l"

    # NSFW detection (NudeNet, ~100MB)
    nsfw_enabled: bool | None = None

    # Audio event classification (PANNs, ~200MB)
    audio_classify_enabled: bool | None = None

    # Image segmentation (SAM 2, ~200MB)
    segment_enabled: bool | None = None
    segment_model: str = "sam2_hiera_small"

    # Speaker diarization (Pyannote 3.1, ~1GB) — always requires token
    pyannote_enabled: bool | None = None
    pyannote_token: str = ""

    # Grounding DINO — open-vocabulary object detection (~180M, ~1.3GB VRAM)
    grounding_dino_enabled: bool | None = None
    grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny"

    # Gemma 4 E4B vision — visual reasoning via llama-cpp-python GGUF (4.84GB IQ4_NL, CPU-only)
    phi4_vision_enabled: bool | None = None
    phi4_vision_model: str = "unsloth/gemma-4-E4B-it-GGUF"
    phi4_vision_filename: str = "gemma-4-E4B-it-IQ4_NL.gguf"
    phi4_vision_mmproj: str = "mmproj-BF16.gguf"
    phi4_vision_n_threads: int = 8

    # Kosmos-2.5 — document OCR and markdown (1.3B, ~3-4GB VRAM)
    kosmos_enabled: bool | None = None
    kosmos_model: str = "microsoft/kosmos-2.5"

    # Model cache
    model_cache_dir: str = "/app/models"

    def is_enabled(self, service: str) -> bool:
        """Check if a service is enabled, respecting profile + override.

        Resolution order:
        1. If MEDIA_<SERVICE>_ENABLED is explicitly set (not None), use it
        2. Otherwise, fall back to the current profile
        """
        attr = f"{service}_enabled"
        override = getattr(self, attr, None)
        if override is not None:
            return override
        return _profile_enabled(self.profile, service)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_device() -> str:
    """Return the compute device string ('cpu' or 'cuda'), validated."""
    d = get_settings().device.lower()
    if d not in ("cpu", "cuda"):
        raise ValueError(f"Invalid MEDIA_DEVICE '{d}', must be 'cpu' or 'cuda'")
    return d
