"""KubeRay Serve configuration — bound deployment graphs.

Each attribute is referenced by import_path in the RayService serveConfigV2.
All imports are safe at module level (heavy deps loaded lazily in _load()).

Tier 1 services are deployed by default. Tier 2/3 are commented out.

GPUGovernor actor is created on import to ensure it's available before
any service tries to acquire a lease.
"""

# Ensure GPUGovernor actor exists before any deployment tries to use it
import ray
try:
    ray.get_actor("gpu_governor")
except ValueError:
    from gateway.gpu_governor import GPUGovernor
    GPUGovernor.options(name="gpu_governor", lifetime="detached").remote()
    print("GPUGovernor actor created.")

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

# Heavy GPU services — routed through Master Router (exclusive GPU access)
# The master router claims num_gpus: 1.0 and swaps models explicitly,
# preventing VRAM collisions on the single RTX 4090.
from services.creative.master_router import master_router as forge

# Playground UI (serves interactive HTML page + service metadata API)
from gateway.playground_deployment import PlaygroundDeployment

playground = PlaygroundDeployment.bind()

# ─── Tier 2: Available via forge master router (uncomment to enable) ──────
# These services route through /forge with exclusive GPU access.
# To enable, add the service name to HEAVY_SERVICES in master_router.py
# and add LOAD_KWARGS if _load() needs arguments.

# VibeVoice Microsoft — microsoft/VibeVoice-ASR 7B with native diarization
# from services.asr.gpu_asr import VibeVoiceMicrosoftDeployment
# vibevoice_microsoft = VibeVoiceMicrosoftDeployment.bind()

# VibeVoice Community TTS — vibevoice/VibeVoice-7B long-form multi-speaker synthesis (18.7GB)
# from services.tts.vibe_voice import VibeVoiceCommunityTTSDeployment
# vibevoice_community_tts = VibeVoiceCommunityTTSDeployment.bind()

# Phi-4-multimodal — 5.6B omni model (text+vision+speech -> text, 24GB)
# Needs model download: task models:pull multimodal.phi4-multimodal
# from services.multimodal.phi4mm import Phi4MMDeployment
# phi4mm = Phi4MMDeployment.bind()

# ─── Tier 3: Blocked — needs Docker image changes ────────────────

# GPT-SoVITS — needs GPT_SoVITS package in Docker image
# from services.tts.gpt_sovits import GPTSoVITSDeployment
# gpt_sovits = GPTSoVITSDeployment.bind()
