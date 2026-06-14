"""Model registry — all models served through native diffusers.

Each entry maps a model name to its diffusers pipeline class, HuggingFace
repo, and generation defaults. No Wan2GP handlers, no translation layer.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelEntry:
    """Configuration for a single model."""
    name: str
    pipeline: str           # diffusers class name (e.g. "ZImagePipeline")
    repo: str               # HF repo or local path
    task: str = "text2image" # text2image | text2video | image2video | edit
    steps: int = 20
    guidance: float = 4.0
    width: int = 1024
    height: int = 1024
    license: str = ""
    notes: str = ""


# ─── Image Models ──────────────────────────────────────────────────────────────

IMAGE_MODELS: dict[str, ModelEntry] = {

    "z-image": ModelEntry(
        name="z-image",
        pipeline="ZImagePipeline",
        repo="Tongyi-MAI/Z-Image",
        steps=30,
        guidance=4.0,
        license="Apache-2.0",
        notes="Z-Image Base — full quality T2I",
    ),
    "z-image-turbo": ModelEntry(
        name="z-image-turbo",
        pipeline="ZImagePipeline",
        repo="Tongyi-MAI/Z-Image-Turbo",
        steps=8,
        guidance=3.5,
        license="Apache-2.0",
        notes="Z-Image Turbo — 8-step distilled",
    ),
    "flux-schnell": ModelEntry(
        name="flux-schnell",
        pipeline="FluxPipeline",
        repo="black-forest-labs/FLUX.1-schnell",
        steps=4,
        guidance=0.0,
        license="Apache-2.0",
        notes="FLUX.1-schnell — 4-step, Apache-2.0",
    ),
    "flux-dev": ModelEntry(
        name="flux-dev",
        pipeline="FluxPipeline",
        repo="black-forest-labs/FLUX.1-dev",
        steps=20,
        guidance=3.5,
        license="Non-commercial",
        notes="FLUX.1-dev — 20-step, higher quality",
    ),
    "flux2-klein-4b": ModelEntry(
        name="flux2-klein-4b",
        pipeline="Flux2KleinPipeline",
        repo="black-forest-labs/FLUX.2-klein-4B",
        steps=8,
        guidance=3.0,
        license="Apache-2.0",
        notes="FLUX.2 Klein 4B — step-distilled, small and fast",
    ),
    "anima": ModelEntry(
        name="anima",
        pipeline="ModularPipeline",
        repo="circlestone-labs/Anima-Base-v1.0-Diffusers",
        steps=30,
        guidance=4.0,
        license="CircleStone Labs",
        notes="Anima Base v1.0 — Cosmos-Predict2 anime/illustration. Uses Qwen3-0.6B encoder.",
    ),
    "qwen-image": ModelEntry(
        name="qwen-image",
        pipeline="QwenImagePipeline",
        repo="Qwen/Qwen-Image",
        steps=30,
        guidance=4.0,
        license="Apache-2.0",
        notes="Qwen-Image — bilingual T2I from Alibaba",
    ),
}

# ─── Video Models ──────────────────────────────────────────────────────────────

VIDEO_MODELS: dict[str, ModelEntry] = {

    "ltx-video": ModelEntry(
        name="ltx-video",
        pipeline="LTXPipeline",
        repo="Lightricks/LTX-Video",
        task="text2video",
        steps=50,
        guidance=6.0,
        width=768,
        height=512,
        license="Apache-2.0",
        notes="LTX-Video 2B — fast video generation",
    ),
    "wan-t2v": ModelEntry(
        name="wan-t2v",
        pipeline="WanPipeline",
        repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        task="text2video",
        steps=30,
        guidance=5.0,
        width=832,
        height=480,
        license="Apache-2.0",
        notes="Wan 2.1 T2V 14B — text-to-video",
    ),
    "wan-i2v": ModelEntry(
        name="wan-i2v",
        pipeline="WanImageToVideoPipeline",
        repo="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        task="image2video",
        steps=30,
        guidance=5.0,
        width=832,
        height=480,
        license="Apache-2.0",
        notes="Wan 2.1 I2V 14B — image-to-video at 480p",
    ),
}

# ─── Combined registry ─────────────────────────────────────────────────────────

ALL_MODELS = {**IMAGE_MODELS, **VIDEO_MODELS}


def get_model(name: str) -> ModelEntry | None:
    """Look up a model by name."""
    return ALL_MODELS.get(name)


def list_models() -> list[str]:
    """List all registered model names."""
    return list(ALL_MODELS.keys())


def is_video(name: str) -> bool:
    """Check if a model is a video model."""
    return name in VIDEO_MODELS
