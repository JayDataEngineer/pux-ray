"""Deploy all Ray Serve applications.

Ray-native IaC: deploys GPU scheduler, ComfyUI extension manager,
and all AI service deployments via Ray Serve.

GPU services run inside Ray-managed containers. Ray handles the
container lifecycle, GPU scheduling, and networking.

Usage:
    python -m scripts.deploy_services
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray import serve

# Clean old Ray session logs from tmpfs to prevent disk exhaustion.
_now = time.time()
_tmp_ray = Path("/tmp/ray")
if _tmp_ray.exists():
    for session_dir in _tmp_ray.glob("session_*"):
        try:
            if session_dir.is_dir() and (_now - session_dir.stat().st_mtime) > 3600:
                import shutil
                shutil.rmtree(session_dir)
                print(f"  Purged old session: {session_dir.name}")
        except Exception:
            pass

# Start Ray if not already running
if not ray.is_initialized():
    ray.init(address="auto")

# Configure Serve HTTP proxy (only if not already running)
try:
    serve.status()
    print("Ray Serve already running, skipping serve.start()")
except Exception:
    serve.start(http_options={"host": "0.0.0.0", "port": 18800})
    print("Ray Serve started on port 18800")

# Deploy GPU scheduler as a named actor
from gateway.gpu_scheduler import GPUScheduler
try:
    scheduler = ray.get_actor("gpu_scheduler")
    print("GPU scheduler already running")
except ValueError:
    scheduler = GPUScheduler.options(
        name="gpu_scheduler", lifetime="detached"
    ).remote()
    print("GPU scheduler deployed")

# Deploy ComfyUI Extension Manager
from gateway.comfyui_manager import ComfyUIExtensionManager
try:
    ext_manager = ray.get_actor("comfyui_ext_manager")
    print("ComfyUI extension manager already running")
except ValueError:
    ext_manager = ComfyUIExtensionManager.options(
        name="comfyui_ext_manager", lifetime="detached"
    ).remote()
    print("ComfyUI extension manager deployed")

# --- Container runtime_env helper ---
from registry.config import Config

_config = Config()
_models_root = _config.models_root


def _container(image: str, extra_mounts: list[str] | None = None) -> dict:
    """Build runtime_env container config with models mount."""
    run_options = [
        "-v", f"{_models_root}:/models:ro",
        "--shm-size", "16g",
    ]
    if extra_mounts:
        run_options.extend(extra_mounts)
    return {
        "container": {
            "image": image,
            "run_options": run_options,
        }
    }


# --- Deploy services ---

print("Deploying LLM...")
from services.llm.deployment import LLMDeployment
serve.run(LLMDeployment.bind(), name="llm", route_prefix="/llm")

print("Deploying CPU TTS...")
from services.tts.espeak import EspeakTTS
serve.run(EspeakTTS.bind(), name="espeak_tts", route_prefix="/tts/espeak")

from services.tts.kokoro import KokoroTTS
serve.run(KokoroTTS.bind(), name="kokoro_tts", route_prefix="/tts/kokoro")

print("Deploying CPU ASR...")
from services.asr.faster_whisper import FasterWhisperASR
serve.run(FasterWhisperASR.bind(), name="faster_whisper", route_prefix="/asr/whisper")

print("Deploying GPU TTS...")
from services.tts.gpu_tts import IndexTTSDeployment
serve.run(IndexTTSDeployment.bind(), name="index_tts", route_prefix="/tts/index-tts")

from services.tts.qwen_tts import QwenTTSDeployment
serve.run(QwenTTSDeployment.bind(), name="qwen_tts", route_prefix="/tts/qwen-tts")

from services.tts.vibe_voice import VibeVoiceDeployment
serve.run(
    VibeVoiceDeployment.options(
        ray_actor_options={
            "num_gpus": 1.0, "num_cpus": 0.5,
            "runtime_env": _container("tech-noir/vibevoice:latest"),
        }
    ).bind(),
    name="vibevoice",
    route_prefix="/tts/vibevoice",
)

from services.tts.gpt_sovits import GPTSoVITSDeployment
serve.run(
    GPTSoVITSDeployment.options(
        ray_actor_options={
            "num_gpus": 1.0, "num_cpus": 0.5,
            "runtime_env": _container("tech-noir/gptsovits:latest"),
        }
    ).bind(),
    name="gpt_sovits",
    route_prefix="/tts/gpt-sovits",
)

print("Deploying GPU ASR...")
from services.asr.gpu_asr import VibeVoiceASRDeployment, QwenASRDeployment
serve.run(VibeVoiceASRDeployment.bind(), name="vibevoice_asr", route_prefix="/asr/vibevoice")
serve.run(QwenASRDeployment.bind(), name="qwen_asr", route_prefix="/asr/qwen")

print("Deploying ComfyUI...")
from services.image.comfyui import ComfyUIDeployment
serve.run(
    ComfyUIDeployment.options(
        ray_actor_options={
            "num_gpus": 1.0, "num_cpus": 0.5,
            "runtime_env": _container("tech-noir/comfyui:latest"),
        }
    ).bind(),
    name="comfyui",
    route_prefix="/comfyui",
)

print("Deploying Creative services...")
from services.creative.trellis import TRELLISDeployment
serve.run(TRELLISDeployment.bind(), name="trellis", route_prefix="/3d/trellis")

from services.creative.anigen import AniGenDeployment
serve.run(
    AniGenDeployment.options(
        ray_actor_options={
            "num_gpus": 1.0, "num_cpus": 0.5,
            "runtime_env": _container("tech-noir/anigen:latest"),
        }
    ).bind(),
    name="anigen",
    route_prefix="/3d/anigen",
)

from services.creative.hy_motion import HYMotionDeployment
serve.run(
    HYMotionDeployment.options(
        ray_actor_options={
            "num_gpus": 1.0, "num_cpus": 0.5,
            "runtime_env": _container("tech-noir/hymotion:latest"),
        }
    ).bind(),
    name="hy_motion",
    route_prefix="/3d/hy-motion",
)

from services.creative.see_through import SeeThroughDeployment
serve.run(
    SeeThroughDeployment.options(
        ray_actor_options={
            "num_gpus": 1.0, "num_cpus": 0.5,
            "runtime_env": _container("tech-noir/seethrough:latest"),
        }
    ).bind(),
    name="see_through",
    route_prefix="/creative/see-through",
)

from services.creative.ace_step import ACEStepDeployment
serve.run(
    ACEStepDeployment.options(
        ray_actor_options={
            "num_gpus": 1.0, "num_cpus": 0.5,
            "runtime_env": _container("tech-noir/acestep:latest"),
        }
    ).bind(),
    name="ace_step",
    route_prefix="/music/ace-step",
)

print("")
print("All services deployed!")
print("  Dashboard: http://localhost:18265")
print("  API:       http://localhost:18800")
print("")
print("Routes:")
print("  /llm/*            - LLM (llama.cpp)")
print("  /tts/kokoro/*     - Kokoro TTS (CPU)")
print("  /tts/espeak/*     - eSpeak TTS (CPU)")
print("  /tts/index-tts/*  - IndexTTS (GPU)")
print("  /tts/qwen-tts/*   - Qwen3-TTS (GPU)")
print("  /tts/vibevoice/*  - VibeVoice TTS (GPU, container)")
print("  /tts/gpt-sovits/* - GPT-SoVITS (GPU, container)")
print("  /asr/whisper/*    - Faster-Whisper (CPU)")
print("  /asr/vibevoice/*  - VibeVoice ASR (GPU)")
print("  /asr/qwen/*       - Qwen ASR (GPU)")
print("  /comfyui/*        - ComfyUI (GPU, container)")
print("  /3d/trellis/*     - TRELLIS.2 (GPU)")
print("  /3d/anigen/*      - AniGen (GPU, container)")
print("  /3d/hy-motion/*   - HY-Motion (GPU, container)")
print("  /creative/see-through/* - See-Through (GPU, container)")
print("  /music/ace-step/* - ACE-STEP music (GPU, container)")
