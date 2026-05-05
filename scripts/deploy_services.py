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

# Configure Serve HTTP proxy.
# Ray Serve persists HTTP options across serve.start() calls — if a stale
# session is running with wrong host/port, detect and restart.
_SERVE_HOST = "0.0.0.0"
_SERVE_PORT = 18800


def _serve_proxy_healthy() -> bool:
    """Check if Serve proxy is listening on the expected port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex(("127.0.0.1", _SERVE_PORT)) == 0


try:
    serve.status()
    if _serve_proxy_healthy():
        print(f"Ray Serve already running on {_SERVE_HOST}:{_SERVE_PORT}")
    else:
        print(f"Ray Serve running but proxy not on port {_SERVE_PORT} — restarting")
        serve.shutdown()
        serve.start(http_options={"host": _SERVE_HOST, "port": _SERVE_PORT})
        print(f"Ray Serve restarted on {_SERVE_HOST}:{_SERVE_PORT}")
except Exception:
    serve.start(http_options={"host": _SERVE_HOST, "port": _SERVE_PORT})
    print(f"Ray Serve started on {_SERVE_HOST}:{_SERVE_PORT}")

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
    from services.base import container_runtime
    mounts = {k: v for d in (extra_mounts or []) for k, v in [d.split(":", 1)]} if extra_mounts else None
    return container_runtime(image, extra_mounts=mounts)


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

# TODO: needs ray installed in image
# from services.tts.qwen_tts import QwenTTSDeployment
# serve.run(QwenTTSDeployment.bind(), name="qwen_tts", route_prefix="/tts/qwen-tts")

# from services.tts.vibe_voice import VibeVoiceDeployment
# serve.run(
#     VibeVoiceDeployment.options(
#         ray_actor_options={
#             "num_gpus": 0, "num_cpus": 0.5,
#             "runtime_env": _container("tech-noir/vibevoice:latest"),
#         }
#     ).bind(),
#     name="vibevoice",
#     route_prefix="/tts/vibevoice",
# )

# TODO: needs ray installed in image
# from services.tts.gpt_sovits import GPTSoVITSDeployment
# serve.run(
#     GPTSoVITSDeployment.options(
#         ray_actor_options={
#             "num_gpus": 0, "num_cpus": 0.5,
#             "runtime_env": _container("tech-noir/gptsovits:latest"),
#         }
#     ).bind(),
#     name="gpt_sovits",
#     route_prefix="/tts/gpt-sovits",
# )

print("Deploying GPU ASR...")
from services.asr.gpu_asr import VibeVoiceASRDeployment, QwenASRDeployment
serve.run(VibeVoiceASRDeployment.bind(), name="vibevoice_asr", route_prefix="/asr/vibevoice")
serve.run(QwenASRDeployment.bind(), name="qwen_asr", route_prefix="/asr/qwen")

print("Deploying ComfyUI...")
from services.image.comfyui import ComfyUIDeployment
serve.run(ComfyUIDeployment.bind(), name="comfyui", route_prefix="/comfyui")

print("Deploying Multimodal services...")
from services.multimodal.phi4mm import Phi4MMDeployment
serve.run(
    Phi4MMDeployment.options(
        ray_actor_options={
            "num_gpus": 0, "num_cpus": 0.5,
            "runtime_env": _container("tech-noir/phi4mm:latest"),
        }
    ).bind(),
    name="phi4mm",
    route_prefix="/multimodal/phi4mm",
)

print("Deploying Vision services...")
from services.vision.florence2 import Florence2Deployment
serve.run(
    Florence2Deployment.options(
        ray_actor_options={
            "num_gpus": 0, "num_cpus": 0.5,
            "runtime_env": _container("tech-noir/florence2:latest"),
        }
    ).bind(),
    name="florence2",
    route_prefix="/vision/florence2",
)

print("Deploying Audio generation services...")
from services.audio.moss_soundeffect import MossSoundEffectDeployment
serve.run(
    MossSoundEffectDeployment.options(
        ray_actor_options={
            "num_gpus": 0, "num_cpus": 0.5,
            "runtime_env": _container("tech-noir/moss-sfx:latest"),
        }
    ).bind(),
    name="moss_soundeffect",
    route_prefix="/audio/moss-soundeffect",
)

from services.audio.tangoflux import TangoFluxDeployment
serve.run(
    TangoFluxDeployment.options(
        ray_actor_options={
            "num_gpus": 0, "num_cpus": 0.5,
            "runtime_env": _container("tech-noir/tangoflux:latest"),
        }
    ).bind(),
    name="tangoflux",
    route_prefix="/audio/tangoflux",
)

print("Deploying Creative services...")
from services.creative.trellis import TRELLISDeployment
serve.run(TRELLISDeployment.bind(), name="trellis", route_prefix="/3d/trellis")

# TODO: needs ray installed in image
# from services.creative.anigen import AniGenDeployment
# serve.run(
#     AniGenDeployment.options(
#         ray_actor_options={
#             "num_gpus": 0, "num_cpus": 0.5,
#             "runtime_env": _container("tech-noir/anigen:latest"),
#         }
#     ).bind(),
#     name="anigen",
#     route_prefix="/3d/anigen",
# )

from services.creative.hy_motion import HYMotionDeployment
serve.run(HYMotionDeployment.bind(), name="hy_motion", route_prefix="/3d/hy-motion")

# TODO: needs ray installed in image
# from services.creative.see_through import SeeThroughDeployment
# serve.run(
#     SeeThroughDeployment.options(
#         ray_actor_options={
#             "num_gpus": 0, "num_cpus": 0.5,
#             "runtime_env": _container("tech-noir/seethrough:latest"),
#         }
#     ).bind(),
#     name="see_through",
#     route_prefix="/creative/see-through",
# )

# TODO: needs ray installed in image
# from services.creative.ace_step import ACEStepDeployment
# serve.run(
#     ACEStepDeployment.options(
#         ray_actor_options={
#             "num_gpus": 0, "num_cpus": 0.5,
#             "runtime_env": _container("tech-noir/acestep:latest"),
#         }
#     ).bind(),
#     name="ace_step",
#     route_prefix="/music/ace-step",
# )

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
# print("  /tts/vibevoice/*  - VibeVoice TTS (GPU, container) -- disabled")
print("  /tts/gpt-sovits/* - GPT-SoVITS (GPU, container)")
print("  /asr/whisper/*    - Faster-Whisper (CPU)")
print("  /asr/vibevoice/*  - VibeVoice ASR (GPU)")
print("  /asr/qwen/*       - Qwen ASR (GPU)")
print("  /comfyui/*        - ComfyUI (GPU, container)")
print("  /multimodal/chat  - Phi-4 Multimodal (GPU, container)")
print("  /vision/florence2 - Florence-2 Vision (GPU, container)")
print("  /audio/soundeffect - MOSS SoundEffect (GPU, container)")
print("  /audio/tangoflux  - TangoFlux Text-to-Audio (GPU, container)")
print("  /3d/trellis/*     - TRELLIS.2 (GPU)")
print("  /3d/anigen/*      - AniGen (GPU, container)")
print("  /3d/hy-motion/*   - HY-Motion (GPU, container)")
print("  /creative/see-through/* - See-Through (GPU, container)")
print("  /music/ace-step/* - ACE-STEP music (GPU, container)")
