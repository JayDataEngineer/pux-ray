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
from services.tts.vibevoice_cpp import VibeVoiceCppDeployment

vibevoice_cpp = VibeVoiceCppDeployment.bind()

# Heavy GPU services — routed through Master Router (exclusive GPU access)
# The master router claims num_gpus: 1.0 and swaps models explicitly,
# preventing VRAM collisions on the single RTX 4090.
from services.creative.master_router import master_router as forge

# Playground UI (serves interactive HTML page + service metadata API)
from gateway.playground_deployment import PlaygroundDeployment

playground = PlaygroundDeployment.bind()

# ─── Tier 3: Third-Class Citizens — not auto-deployed ──────────
# moss_soundeffect handled by Master Router via /forge (not standalone).
# Uncomment below to deploy individually for debugging.

# from services.tts.gpt_sovits import GPTSoVITSDeployment
# gpt_sovits = GPTSoVITSDeployment.bind()

# from services.asr.gpu_asr import VibeVoiceASRDeployment, QwenASRDeployment
# vibevoice_asr = VibeVoiceASRDeployment.bind()
# qwen_asr = QwenASRDeployment.bind()

# from services.tts.vibe_voice import VibeVoiceDeployment
# vibevoice = VibeVoiceDeployment.bind()


# from services.multimodal.phi4mm import Phi4MMDeployment
# phi4mm = Phi4MMDeployment.bind()


# from services.tts.qwen_tts_legacy import QwenTTSDeployment
# qwen_tts = QwenTTSDeployment.bind()
