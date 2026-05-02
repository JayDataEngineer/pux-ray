"""Deploy all Ray Serve applications.

Usage:
    python -m scripts.deploy_services
"""

from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray import serve

# Start Ray if not already running
if not ray.is_initialized():
    ray.init(address="auto")

# Configure Serve HTTP proxy
serve.start(http_options={"host": "0.0.0.0", "port": 18800})

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

# Deploy JobManager as a named actor (tracks queued generation jobs)
from gateway.jobs import JobManager
try:
    job_manager = ray.get_actor("job_manager")
    print("JobManager already running")
except ValueError:
    job_manager = JobManager.options(
        name="job_manager", lifetime="detached"
    ).remote()
    print("JobManager deployed")

# Deploy Git Sidecar (polls all infra git repos and reloads on push)
from gateway.git_sidecar import GitSidecar
try:
    sidecar = ray.get_actor("git_sidecar")
    print("Git sidecar already running")
except ValueError:
    sidecar = GitSidecar.options(
        name="git_sidecar", lifetime="detached"
    ).remote()
    sidecar.start.remote()
    print("Git sidecar deployed")

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
serve.run(VibeVoiceDeployment.bind(), name="vibevoice", route_prefix="/tts/vibevoice")

from services.tts.gpt_sovits import GPTSoVITSDeployment
serve.run(GPTSoVITSDeployment.bind(), name="gpt_sovits", route_prefix="/tts/gpt-sovits")

print("Deploying GPU ASR...")
from services.asr.gpu_asr import VibeVoiceASRDeployment, QwenASRDeployment
serve.run(VibeVoiceASRDeployment.bind(), name="vibevoice_asr", route_prefix="/asr/vibevoice")
serve.run(QwenASRDeployment.bind(), name="qwen_asr", route_prefix="/asr/qwen")

print("Deploying ComfyUI...")
from services.image.comfyui import ComfyUIDeployment
serve.run(ComfyUIDeployment.bind(), name="comfyui", route_prefix="/comfyui")

print("Deploying Creative services...")
from services.creative.trellis import TRELLISDeployment
serve.run(TRELLISDeployment.bind(), name="trellis", route_prefix="/3d/trellis")

from services.creative.anigen import AniGenDeployment
serve.run(AniGenDeployment.bind(), name="anigen", route_prefix="/3d/anigen")

from services.creative.see_through import SeeThroughDeployment
serve.run(SeeThroughDeployment.bind(), name="see_through", route_prefix="/creative/see-through")

from services.creative.ace_step import ACEStepDeployment
serve.run(ACEStepDeployment.bind(), name="ace_step", route_prefix="/music/ace-step")

print("MCP servers are persistent processes — start with scripts/start_mcp.sh")

print("")
print("All services deployed!")
print("  GPU Dashboard: http://localhost:18800/dashboard")
print("  Ray Dashboard: http://localhost:18265")
print("  API:           http://localhost:18800")
print("")
print("Routes:")
print("  /llm/*            - LLM (llama.cpp)")
print("  /tts/kokoro/*     - Kokoro TTS (CPU)")
print("  /tts/espeak/*     - eSpeak TTS (CPU)")
print("  /tts/index-tts/*  - IndexTTS (GPU)")
print("  /tts/qwen-tts/*   - Qwen3-TTS (GPU)")
print("  /tts/vibevoice/*  - VibeVoice TTS (GPU)")
print("  /tts/gpt-sovits/* - GPT-SoVITS (GPU)")
print("  /asr/whisper/*    - Faster-Whisper (CPU)")
print("  /asr/vibevoice/*  - VibeVoice ASR (GPU)")
print("  /asr/qwen/*       - Qwen ASR (GPU)")
print("  /comfyui/*        - ComfyUI (GPU, WebUI)")
print("  /3d/trellis/*     - TRELLIS.2 (GPU)")
print("  /3d/anigen/*      - AniGen (GPU)")
print("  /creative/see-through/* - See-Through (GPU)")
print("  /mcp/web/*             - Local Web MCP (persistent)")
print("  /mcp/media/*           - Media Analysis MCP (persistent)")
print("")
print("Job Routes (queued, async):")
print("  POST /jobs/{type}     - Submit job (trellis, anigen, ace_step, comfyui)")
print("  GET  /jobs/{id}       - Get job status")
print("  GET  /jobs/{id}/result- Get job result (binary)")
print("  GET  /jobs            - List all jobs")
