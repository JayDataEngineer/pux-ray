"""Model registry for native diffusers service.

Each model declares: pipeline class, repo path, components, defaults,
and VRAM/optimization hints for the adaptive loader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ComponentConfig:
    """A single pipeline component (transformer, text_encoder, vae, etc.)."""
    type: str                    # "transformer" | "text_encoder" | "vae" | "scheduler"
    quantizable: bool = False    # Can we quantize this without visible quality loss?
    precision_critical: bool = False  # Is this precision-sensitive? (transformer = True)
    subfolder: Optional[str] = None    # Subfolder in the repo
    always_bf16: bool = False   # Force BF16 regardless of format selection


@dataclass
class ModelConfig:
    """Configuration for a single model served through native diffusers."""
    name: str                           # Registry key (e.g. "z-image-turbo")
    pipeline_class: str                 # e.g. "ZImagePipeline"
    repo: str                           # HF repo or local path
    repo_diffusers: Optional[str] = None  # Diffusers-format repo (if different)
    task: str = "text2image"            # text2image | text2video | image2video | image_edit
    default_steps: int = 20
    default_guidance: float = 4.0
    default_size: tuple[int, int] = (1024, 1024)
    max_size: tuple[int, int] = (1536, 1536)
    components: dict[str, ComponentConfig] = field(default_factory=dict)
    license: str = "Apache-2.0"
    notes: str = ""

    def __post_init__(self):
        """Set default components if not specified."""
        if not self.components:
            self.components = {
                "text_encoder": ComponentConfig(type="text_encoder", quantizable=True),
                "transformer": ComponentConfig(type="transformer", quantizable=True, precision_critical=True),
                "vae": ComponentConfig(type="vae", always_bf16=True),
            }


# ─── Model Registry ───────────────────────────────────────────────────────────

MODELS: dict[str, ModelConfig] = {

    # ── Z-Image (Alibaba) ──────────────────────────────────────────────────────
    "z-image": ModelConfig(
        name="z-image",
        pipeline_class="ZImagePipeline",
        repo="/models/z-image",
        repo_diffusers="Tongyi-MAI/Z-Image",
        task="text2image",
        default_steps=30,
        default_guidance=4.0,
        license="Apache-2.0",
        notes="Z-Image Base — full quality T2I, 30 steps for best results",
    ),
    "z-image-turbo": ModelConfig(
        name="z-image-turbo",
        pipeline_class="ZImagePipeline",
        repo="/models/z-image-turbo",
        repo_diffusers="Tongyi-MAI/Z-Image-Turbo",
        task="text2image",
        default_steps=8,
        default_guidance=3.5,
        license="Apache-2.0",
        notes="Z-Image Turbo — 8 step distilled, sub-second generation",
    ),

    # ── Anima (CircleStone Labs) ───────────────────────────────────────────────
    "anima": ModelConfig(
        name="anima",
        pipeline_class="ModularPipeline",
        repo="/models/anima",
        repo_diffusers="circlestone-labs/Anima-Base-v1.0-Diffusers",
        task="text2image",
        default_steps=30,
        default_guidance=4.0,
        default_size=(1024, 1024),
        components={
            "text_encoder": ComponentConfig(type="text_encoder", quantizable=True,
                                            subfolder="text_encoder"),
            "transformer": ComponentConfig(type="transformer", quantizable=True,
                                           precision_critical=True,
                                           subfolder="transformer"),
            "vae": ComponentConfig(type="vae", always_bf16=True,
                                   subfolder="vae"),
        },
        license="CircleStone Labs — verify",
        notes="Anima Base v1.0 — Cosmos-Predict2 anime/illustration. Uses Qwen3-0.6B text encoder. ModularPipeline loads from config.",
    ),

    # ── FLUX.1 (Black Forest Labs) ─────────────────────────────────────────────
    "flux-schnell": ModelConfig(
        name="flux-schnell",
        pipeline_class="FluxPipeline",
        repo="/models/flux-schnell",
        repo_diffusers="black-forest-labs/FLUX.1-schnell",
        task="text2image",
        default_steps=4,
        default_guidance=0.0,
        license="Apache-2.0",
        notes="FLUX.1-schnell — 4 step, Apache-2.0. Fastest FLUX variant.",
    ),
    "flux-dev": ModelConfig(
        name="flux-dev",
        pipeline_class="FluxPipeline",
        repo="/models/flux-dev",
        repo_diffusers="black-forest-labs/FLUX.1-dev",
        task="text2image",
        default_steps=20,
        default_guidance=3.5,
        license="Non-commercial",
        notes="FLUX.1-dev — 20 step, higher quality. Non-commercial license.",
    ),

    # ── FLUX.2 Klein (Black Forest Labs) ───────────────────────────────────────
    "flux2-klein-4b": ModelConfig(
        name="flux2-klein-4b",
        pipeline_class="Flux2KleinPipeline",
        repo="/models/flux2-klein-4b",
        repo_diffusers="black-forest-labs/FLUX.2-klein-4B",
        task="text2image",
        default_steps=8,
        default_guidance=3.0,
        license="Apache-2.0",
        notes="FLUX.2 Klein 4B — step-distilled, Apache-2.0. Small enough for max-speed recipe.",
    ),

    # ── Wan 2.1/2.2 (Alibaba) ──────────────────────────────────────────────────
    "wan-t2v-14b": ModelConfig(
        name="wan-t2v-14b",
        pipeline_class="WanPipeline",
        repo="/models/wan-t2v-14b",
        repo_diffusers="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        task="text2video",
        default_steps=30,
        default_guidance=5.0,
        default_size=(832, 480),
        components={
            "text_encoder": ComponentConfig(type="text_encoder", quantizable=True),
            "transformer": ComponentConfig(type="transformer", quantizable=True,
                                           precision_critical=True),
            "vae": ComponentConfig(type="vae", always_bf16=True),
        },
        license="Apache-2.0",
        notes="Wan 2.1 T2V 14B — text-to-video. Large model, needs group_offload on 24GB.",
    ),
    "wan-i2v-14b": ModelConfig(
        name="wan-i2v-14b",
        pipeline_class="WanImageToVideoPipeline",
        repo="/models/wan-i2v-14b",
        repo_diffusers="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        task="image2video",
        default_steps=30,
        default_guidance=5.0,
        default_size=(832, 480),
        license="Apache-2.0",
        notes="Wan 2.1 I2V 14B — image-to-video at 480p.",
    ),

    # ── LTX-Video / LTX-2 (Lightricks) ─────────────────────────────────────────
    "ltx-video": ModelConfig(
        name="ltx-video",
        pipeline_class="LTXPipeline",
        repo="/models/ltx-video",
        repo_diffusers="Lightricks/LTX-Video",
        task="text2video",
        default_steps=50,
        default_guidance=6.0,
        default_size=(768, 512),
        components={
            "text_encoder": ComponentConfig(type="text_encoder", quantizable=True),
            "transformer": ComponentConfig(type="transformer", quantizable=True,
                                           precision_critical=True),
            "vae": ComponentConfig(type="vae", always_bf16=True),
        },
        license="Apache-2.0",
        notes="LTX-Video 2B — fast video gen. Small enough for resident on 24GB.",
    ),
    "ltx-2": ModelConfig(
        name="ltx-2",
        pipeline_class="LTX2Pipeline",
        repo="/models/ltx-2",
        repo_diffusers="Lightricks/LTX-2",
        task="text2video",
        default_steps=40,
        default_guidance=6.0,
        default_size=(768, 512),
        license="Lightricks — verify",
        notes="LTX-2 — two-stage pipeline (base + latent upscaler). Use ltx2_two_stage mode.",
    ),

    # ── Qwen-Image (Alibaba) ───────────────────────────────────────────────────
    "qwen-image": ModelConfig(
        name="qwen-image",
        pipeline_class="QwenImagePipeline",
        repo="/models/qwen-image",
        repo_diffusers="Qwen/Qwen-Image",
        task="text2image",
        default_steps=30,
        default_guidance=4.0,
        license="Apache-2.0",
        notes="Qwen-Image — bilingual T2I from Alibaba.",
    ),
}


def get_model_config(name: str) -> ModelConfig | None:
    """Look up model config by name."""
    return MODELS.get(name)


def list_models() -> list[str]:
    """List all registered model names."""
    return list(MODELS.keys())
