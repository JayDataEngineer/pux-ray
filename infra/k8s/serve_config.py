"""KubeRay Serve configuration — bound deployment graphs.

Each attribute is referenced by import_path in the RayService serveConfigV2.
All imports are safe at module level (heavy deps loaded lazily in _load()).

Tier 1 services are deployed by default. Tier 2/3 are commented out.

The Forge replaces the old Master Router + GPU Governor with a single
VRAM-aware GPU manager.
"""

# ─── Tier 1: First-Class Citizens (Native Ray, auto-deployed) ───────────────

# CPU TTS (head node)
from services.tts.kokoro import KokoroTTS
from services.tts.espeak import EspeakTTS

kokoro_tts = KokoroTTS.bind()
espeak_tts = EspeakTTS.bind()

# CPU ASR (head node)
from services.asr.faster_whisper import FasterWhisperASR

faster_whisper = FasterWhisperASR.bind()

# Lightweight GPU TTS (can coexist on GPU)
from services.tts.faster_qwen3_tts import FasterQwen3TTSDeployment
from services.tts.gpu_tts import IndexTTSDeployment

faster_qwen3_tts = FasterQwen3TTSDeployment.bind()
index_tts = IndexTTSDeployment.bind()

# vibevoice.cpp (subprocess — TTS + ASR via quantized GGUF)
from services.tts.vibevoice_cpp import VibeVoiceCppGpuDeployment, VibeVoiceCppCpuDeployment

vibevoice_cpp_gpu = VibeVoiceCppGpuDeployment.bind()
vibevoice_cpp_cpu = VibeVoiceCppCpuDeployment.bind()

# ─── The Forge — VRAM-aware GPU manager ─────────────────────────────────────
# Replaces the old Master Router + GPU Governor.
# Claims num_gpus: 1.0, tracks VRAM inline, swaps models as needed.
from services.forge import forge

# Playground UI (serves interactive HTML page + service metadata API)
from gateway.playground_deployment import PlaygroundDeployment

playground = PlaygroundDeployment.bind()

# API Ingress — catch-all gateway for /health, /v1/*, /dashboard, /studio, etc.
# Must be LAST — Ray Serve matches most-specific route_prefix first, so
# /tts/kokoro, /asr/whisper, /forge, /playground bypass this deployment.
from gateway.ingress_deployment import APIIngressDeployment

api_ingress = APIIngressDeployment.bind()

# ─── Tier 2: Available via forge master router (uncomment to enable) ──────
# These services are available through the Forge on demand.
# To enable, add the service to SERVICE_MAP in services/forge.py.

# Wan2GP Pool — video generation, many model variants, mmgp-managed VRAM.
# Already in SERVICE_MAP in services/forge.py.

# VibeVoice Microsoft — microsoft/VibeVoice-ASR 7B with native diarization
# VibeVoice Community TTS — vibevoice/VibeVoice-7B long-form multi-speaker synthesis
# Phi-4-multimodal — 5.6B omni model (text+vision+speech -> text, 24GB)

# ─── Tier 3: Blocked — needs Docker image changes ────────────────

# GPT-SoVITS — needs GPT_SoVITS package in Docker image
# from services.tts.gpt_sovits import GPTSoVITSDeployment
# gpt_sovits = GPTSoVITSDeployment.bind()
