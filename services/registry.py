"""Centralized service registry — single source of truth for all deployments.

Every model service routes through the Forge (VRAM-aware subprocess manager) or
directly to its deployment handle. The registry maps public service names to
metadata for API docs, dashboard, and routing.

params_schema defines the inputs a service accepts. The web UI uses this to
auto-generate forms without hardcoding per-service parameters.

Each param entry:
  - type: "text" | "number" | "select" | "file" | "textarea" | "json" | "bool"
  - label: Human-readable field label
  - required: Whether the field is required
  - default: Default value
  - placeholder: Placeholder text
  - description: Help text
  - options: For "select" type, list of choices
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParamSpec:
    """Parameter specification for auto-generated UI forms."""
    type: str = "text"
    label: str = ""
    required: bool = False
    default: str | int | float | bool | None = None
    placeholder: str = ""
    description: str = ""
    options: list[str] | None = None


@dataclass
class ServiceEntry:
    deployment: str
    """Ray Serve deployment name."""

    app: str
    """Ray Serve application name."""

    label: str
    """Human-readable name for dashboard display."""

    category: str
    """Service category: tts, asr, audio, creative, llm, image, 3d, motion."""

    needs_gpu: bool
    """Whether this service uses GPU."""

    default_model: str
    """Default model name."""

    output_type: str
    """Primary output format: audio, json, model_3d, image, proxy, video, motion."""

    model_aliases: dict[str, str] = field(default_factory=dict)
    description: str = ""
    params_schema: list[ParamSpec] = field(default_factory=list)
    """Input parameters for auto-generating UI forms."""


SERVICE_REGISTRY: dict[str, ServiceEntry] = {
    # ── Forge Services (subprocess-based) ──────────────────────────────────────
    "comfyui": ServiceEntry(
        deployment="forge", app="forge",
        label="ComfyUI", category="image",
        needs_gpu=True, default_model="comfyui",
        output_type="proxy",
        description="ComfyUI — node-based image/video generation pipeline.",
    ),
    "llm": ServiceEntry(
        deployment="forge", app="forge",
        label="LLM (llama.cpp)", category="llm",
        needs_gpu=True, default_model="qwen3.6-27b-q5_k_s-32k",
        output_type="json",
        description="llama.cpp LLM server — OpenAI-compatible chat completions.",
        params_schema=[
            ParamSpec(type="textarea", label="Message", required=True, placeholder="Enter your prompt..."),
            ParamSpec(type="number", label="Max Tokens", default=2048, placeholder="2048"),
            ParamSpec(type="number", label="Temperature", default=0.7, placeholder="0.7"),
        ],
    ),

    # ── Model Engine services ────────────────────────────────────────────────────
    "kokoro": ServiceEntry(
        deployment="diffusers", app="kokoro",   # routes via the diffusers Tier D pool
        label="Kokoro TTS", category="tts",
        needs_gpu=False, default_model="kokoro",
        output_type="audio",
        model_aliases={"tts-01-kokoro": "kokoro"},
        description="Kokoro 82M via sherpa-onnx — fast CPU TTS, 53 voices (EN+ZH). Standalone inference-kokoro docker on port 8060.",
        params_schema=[
            ParamSpec(type="textarea", label="Text", required=True, placeholder="Hello world"),
            ParamSpec(type="select", label="Voice", default="af_bella", options=[
                "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
                "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
                "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
                "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
                "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
                "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
                "ef_dora", "em_alex", "em_santa", "ff_siwis",
                "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
                "if_sara", "im_nicola",
                "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
                "jm_kumo", "pf_dora", "pm_alex", "pm_santa",
                "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
            ]),
        ],
    ),
    "espeak": ServiceEntry(
        deployment="forge", app="forge",
        label="eSpeak TTS", category="tts",
        needs_gpu=False, default_model="espeak",
        output_type="audio",
        model_aliases={"tts-01-espeak": "espeak"},
        description="eSpeak-NG — lightweight phoneme TTS, many languages.",
        params_schema=[
            ParamSpec(type="textarea", label="Text", required=True, placeholder="Hello world"),
            ParamSpec(type="select", label="Language", default="en", options=[
                "en", "fr", "de", "es", "it", "ja", "zh", "ko", "ru", "pt",
            ]),
        ],
    ),
    "faster_whisper": ServiceEntry(
        deployment="forge", app="forge",
        label="Faster-Whisper", category="asr",
        needs_gpu=False, default_model="faster_whisper",
        output_type="json",
        model_aliases={"whisper-1": "faster_whisper"},
        description="Faster-Whisper distil-large-v3 — fast CPU ASR.",
        params_schema=[
            ParamSpec(type="file", label="Audio File", required=True, placeholder="Upload audio to transcribe"),
            ParamSpec(type="select", label="Language", default="auto", options=["auto", "en", "fr", "de", "es", "ja"]),
        ],
    ),
    "moss_soundeffect": ServiceEntry(
        deployment="forge", app="forge",
        label="MOSS-SoundEffect", category="audio",
        needs_gpu=True, default_model="moss/moss-soundeffect",
        output_type="audio",
        description="MOSS-SoundEffect 8B — text-to-sound effects.",
        params_schema=[
            ParamSpec(type="text", label="Description", required=True, placeholder="rain and thunder, footsteps on gravel..."),
            ParamSpec(type="number", label="Duration (s)", default=5, placeholder="5"),
        ],
    ),
    "ace_step": ServiceEntry(
        deployment="forge", app="forge",
        label="ACE-Step", category="audio",
        needs_gpu=True, default_model="ace_step/v1_5",
        output_type="audio",
        description="ACE-STEP 1.5 — text-to-music generation.",
        params_schema=[
            ParamSpec(type="textarea", label="Prompt", required=True, placeholder="epic cinematic orchestral, dark synthwave..."),
            ParamSpec(type="number", label="Duration (s)", default=30, placeholder="30"),
        ],
    ),
    "vibevoice_asr": ServiceEntry(
        deployment="forge", app="forge",
        label="VibeVoice ASR", category="asr",
        needs_gpu=True, default_model="vibevoice_asr",
        output_type="json",
        description="VibeVoice ASR — speech recognition with diarization.",
        params_schema=[
            ParamSpec(type="file", label="Audio File", required=True, placeholder="Upload audio to transcribe"),
        ],
    ),

    # ── Additional Forge-managed services ──────────────────────────────────────
    "trellis": ServiceEntry(
        deployment="forge", app="forge",
        label="TRELLIS 3D", category="3d",
        needs_gpu=True, default_model="trellis",
        output_type="model_3d",
        description="TRELLIS.2 4B — image-to-3D mesh generation.",
        params_schema=[
            ParamSpec(type="file", label="Input Image", required=True, placeholder="Upload an image to convert to 3D"),
            ParamSpec(type="select", label="Quality", default="standard", options=["turbo", "standard", "high"]),
        ],
    ),
    "anigen": ServiceEntry(
        deployment="forge", app="forge",
        label="AniGen 3D", category="3d",
        needs_gpu=True, default_model="anigen",
        output_type="model_3d",
        description="AniGen — anime image-to-rigged-3D generation.",
        params_schema=[
            ParamSpec(type="file", label="Anime Image", required=True, placeholder="Upload anime character image"),
        ],
    ),
    "z_image": ServiceEntry(
        deployment="forge", app="forge",
        label="Z-Image Generation", category="image",
        needs_gpu=True, default_model="z_image",
        output_type="image",
        description="Z-Image generation (Flux, SD variants) via Forge.",
        params_schema=[
            ParamSpec(type="textarea", label="Prompt", required=True, placeholder="A cyberpunk samurai in a neon-lit alleyway..."),
            ParamSpec(type="text", label="Negative Prompt", required=False, placeholder="blurry, low quality, distorted..."),
            ParamSpec(type="select", label="Quality", default="turbo", options=["turbo", "standard", "high"]),
            ParamSpec(type="number", label="Width", default=1024, placeholder="1024"),
            ParamSpec(type="number", label="Height", default=1024, placeholder="1024"),
        ],
    ),
    # faster_qwen3_tts / qwen3_tts service removed — superseded by
    # OpenMOSS VoiceGenerator (instruction-following + multilingual TTS)
    # and sherpa-onnx Kokoro (CPU-fast TTS). The 4.3 GB Qwen3-TTS-12Hz
    # weights, faster_qwen3_tts handler, and opt/wan2gp/models/TTS/qwen3/
    # tree have been deleted.
    "moss_voicegenerator": ServiceEntry(
        deployment="forge", app="forge",
        label="MOSS Voice Design", category="tts",
        needs_gpu=True, default_model="moss-voicegenerator",
        output_type="audio",
        description="MOSS VoiceGenerator — design a voice from text description.",
        params_schema=[
            ParamSpec(type="textarea", label="Voice Description", required=True,
                      placeholder="A warm female voice with a gentle southern accent...",
                      description="Describe the voice you want to generate."),
            ParamSpec(type="select", label="Language", default="English",
                      options=["English", "Chinese", "Japanese", "Korean"]),
        ],
    ),
    "moss_tts": ServiceEntry(
        deployment="forge", app="forge",
        label="MOSS TTS", category="tts",
        needs_gpu=True, default_model="moss-tts",
        output_type="audio",
        description="MOSS TTS — text-to-speech with voice cloning via reference audio.",
        params_schema=[
            ParamSpec(type="textarea", label="Text", required=True, placeholder="Hello world"),
            ParamSpec(type="textarea", label="Instruction", required=False,
                      placeholder="warm, friendly, slightly husky",
                      description="Optional emotion/style instruction"),
            ParamSpec(type="file", label="Reference Audio", required=True,
                      placeholder="Upload reference audio for voice cloning",
                      description="Audio sample to clone the voice from"),
        ],
    ),
    "hy_motion": ServiceEntry(
        deployment="forge", app="forge",
        label="HY-Motion", category="motion",
        needs_gpu=True, default_model="hy_motion",
        output_type="motion",
        description="HY-Motion 1.0 — text-to-3D motion generation.",
        params_schema=[
            ParamSpec(type="textarea", label="Motion Description", required=True, placeholder="A person walking forward, arms swinging naturally..."),
            ParamSpec(type="select", label="Quality", default="low", options=["low", "standard", "high"]),
        ],
    ),
    "see_through": ServiceEntry(
        deployment="forge", app="forge",
        label="See-Through", category="creative",
        needs_gpu=True, default_model="see_through",
        output_type="image",
        description="See-Through — anime layer decomposition.",
        params_schema=[
            ParamSpec(type="file", label="Anime Image", required=True, placeholder="Upload an anime image to decompose"),
        ],
    ),
    "nvidia_upscale": ServiceEntry(
        deployment="forge", app="forge",
        label="GPU Upscale", category="image",
        needs_gpu=True, default_model="nvidia_upscale",
        output_type="video",
        description="GPU-accelerated Lanczos upscaling for images and video.",
        params_schema=[
            ParamSpec(type="file", label="Input Media", required=True, placeholder="Upload image or video to upscale"),
            ParamSpec(type="number", label="Scale Factor", default=2, placeholder="2"),
        ],
    ),
    "dwpose": ServiceEntry(
        deployment="forge", app="forge",
        label="DWPose Detection", category="image",
        needs_gpu=False, default_model="dwpose",
        output_type="json",
        description="Mediapipe face mesh keypoint detection + face cropping.",
        params_schema=[
            ParamSpec(type="file", label="Image", required=True, placeholder="Upload an image for pose detection"),
        ],
    ),
    "body_mesh": ServiceEntry(
        deployment="forge", app="forge",
        label="BodyMesh Renderer", category="3d",
        needs_gpu=False, default_model="body_mesh",
        output_type="image",
        description="Skeleton wireframe renderer from joint rotation parameters.",
        params_schema=[
            ParamSpec(type="json", label="Joint Rotations", required=True, placeholder='{"spine": [0,0,0], "arm_l": [0.5,0,0], ...}'),
        ],
    ),

    # ── Kimodo Motion (Forge-managed, standalone) ────────────────────────────
    "kimodo": ServiceEntry(
        deployment="forge", app="forge",
        label="Kimodo Motion", category="motion",
        needs_gpu=True, default_model="kimodo-soma-rp",
        output_type="motion",
        description="Kimodo text-to-3D motion (NPZ output, SOMA 77-joint skeleton).",
        params_schema=[
            ParamSpec(type="textarea", label="Motion Description", required=True, placeholder="Describe the motion..."),
            ParamSpec(type="number", label="Duration (s)", default=5, placeholder="5"),
        ],
    ),
    "lance": ServiceEntry(
        deployment="forge", app="forge",
        label="Lance Multimodal", category="creative",
        needs_gpu=True, default_model="lance/lance-video",
        output_type="video",
        model_aliases={"lance-t2v": "lance/lance-video", "lance-t2i": "lance/lance-image"},
        description="Lance 3B — ByteDance unified multimodal: t2i, t2v, image/video edit, understanding.",
        params_schema=[
            ParamSpec(type="textarea", label="Prompt", required=True, placeholder="Describe what to generate..."),
            ParamSpec(type="select", label="Mode", default="t2v", options=["t2v", "t2i", "edit", "understand"]),
            ParamSpec(type="number", label="Duration (s)", default=5, placeholder="5"),
        ],
    ),

    "kimodo_demo": ServiceEntry(
        deployment="forge", app="forge",
        label="Kimodo Demo", category="motion",
        needs_gpu=True, default_model="kimodo-soma-rp",
        output_type="proxy",
        description="Kimodo Viser demo — interactive 3D motion authoring.",
    ),

    "gemx": ServiceEntry(
        deployment="forge", app="forge",
        label="GEM-X Pose Estimator", category="motion",
        needs_gpu=True, default_model="gem_soma",
        output_type="json",
        description="GEM-X — NVIDIA video-based SOMA 77-joint pose estimation (Apache 2.0).",
        params_schema=[
            ParamSpec(type="file", label="Video File", required=True, placeholder="Upload a video for pose estimation"),
        ],
    ),

    "kohya": ServiceEntry(
        deployment="forge", app="forge",
        label="kohya_ss LoRA Trainer", category="training",
        needs_gpu=True, default_model="lance-poseedit",
        output_type="json",
        model_aliases={
            "lance-poseedit": "lance-poseedit",
            "klein-poseedit": "klein-poseedit",
        },
        description="kohya_ss sd-scripts — LoRA training for Lance 3B / Klein 4B (Apache 2.0).",
        params_schema=[
            ParamSpec(type="select", label="Base Model", default="lance-poseedit", options=["lance-poseedit", "klein-poseedit"]),
            ParamSpec(type="file", label="Training Images", required=True, placeholder="Upload training dataset (zip)"),
            ParamSpec(type="text", label="Instance Prompt", required=True, placeholder="A photo of X style..."),
            ParamSpec(type="number", label="Training Steps", default=1000, placeholder="1000"),
        ],
    ),

    # ── Avatar Pipeline (Forge-managed, staged VRAM) ──────────────────────────
    "avatar": ServiceEntry(
        deployment="forge", app="forge",
        label="Avatar Pipeline", category="avatar",
        needs_gpu=True, default_model="gem_smpl",
        output_type="video",
        description="Text-to-avatar: GEM gesture gen + SOMA body model + FluxRT rendering.",
        params_schema=[
            ParamSpec(type="textarea", label="Avatar Description", required=True, placeholder="Describe the character and motion..."),
            ParamSpec(type="select", label="Style", default="photorealistic", options=["photorealistic", "anime", "stylized", "cartoon"]),
        ],
    ),

    # ── Anima (anime/illustration image generation) ──────────────────────────────
    "anima": ServiceEntry(
        deployment="forge", app="forge",
        label="Anima", category="image",
        needs_gpu=True, default_model="anima",
        output_type="image",
        description="Anima — anime/illustration image generation via Forge.",
        params_schema=[
            ParamSpec(type="textarea", label="Prompt", required=True, placeholder="A beautiful anime girl..."),
            ParamSpec(type="number", label="Width", default=1024, placeholder="1024"),
            ParamSpec(type="number", label="Height", default=1024, placeholder="1024"),
            ParamSpec(type="number", label="Steps", default=30, placeholder="30"),
            ParamSpec(type="number", label="CFG Scale", default=4.0, placeholder="4.0"),
        ],
    ),

    # ── Native — diffusers model classes + manual denoise ───────────────────────
    "native": ServiceEntry(
        deployment="forge", app="forge",
        label="Native Models", category="image",
        needs_gpu=True, default_model="z-image-turbo",
        output_type="image",
        description="Native diffusers — Z-Image, FLUX, LTX Video, and more through adaptive VRAM optimization.",
        params_schema=[
            ParamSpec(type="text", label="Model", required=False, default="z-image-turbo",
                      placeholder="z-image-turbo", description="Model name from native registry"),
            ParamSpec(type="textarea", label="Prompt", required=True, placeholder="Describe what to generate..."),
            ParamSpec(type="text", label="Negative Prompt", required=False, placeholder="things to avoid..."),
            ParamSpec(type="number", label="Steps", default=4, placeholder="4"),
            ParamSpec(type="number", label="CFG Scale", default=3.5, placeholder="3.5"),
            ParamSpec(type="number", label="Width", default=1024, placeholder="1024"),
            ParamSpec(type="number", label="Height", default=1024, placeholder="1024"),
            ParamSpec(type="number", label="Seed", default=-1, placeholder="-1 (random)"),
        ],
    ),
}


def get_service(name: str) -> Optional[ServiceEntry]:
    return SERVICE_REGISTRY.get(name)


def resolve_model(model_name: str) -> Optional[tuple[str, ServiceEntry]]:
    for key, entry in SERVICE_REGISTRY.items():
        if model_name in entry.model_aliases:
            return key, entry
        if model_name == key:
            return key, entry
    return None


def get_all_services() -> dict[str, ServiceEntry]:
    return dict(SERVICE_REGISTRY)


def list_all_models(category: str | None = None) -> list[dict]:
    """Return all service entries as model dicts for API discovery."""
    models = []
    for key, entry in SERVICE_REGISTRY.items():
        if category and entry.category != category:
            continue
        models.append({
            "id": key,
            "label": entry.label,
            "category": entry.category,
            "needs_gpu": entry.needs_gpu,
            "output_type": entry.output_type,
            "default_model": entry.default_model,
            "description": entry.description,
            "model_aliases": list(entry.model_aliases.keys()),
            "params_schema": [
                {
                    "name": p.label.lower().replace(" ", "_"),
                    "type": p.type,
                    "label": p.label,
                    "required": p.required,
                    "default": p.default,
                    "placeholder": p.placeholder,
                    "description": p.description,
                    "options": p.options,
                }
                for p in entry.params_schema
            ],
        })
    return models
