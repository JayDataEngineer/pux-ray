"""KubeRay Serve configuration — bound deployment graphs.

Each attribute is referenced by import_path in the RayService serveConfigV2.
All imports are safe at module level (heavy deps loaded lazily in _load()).

Tier 1 services are deployed by default. Tier 2/3 are commented out.
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
from services.tts.vibevoice_cpp import VibeVoiceCppDeployment

vibevoice_cpp = VibeVoiceCppDeployment.bind()

# Heavy GPU services via Master Router (exclusive GPU, explicit model swapping)
# trellis, ace_step, comfyui, hy_motion all go through this single deployment
from services.creative.master_router import master_router

# LLM via llama.cpp (subprocess — GGUF inference on GPU)
from services.llm.deployment import LLMDeployment

llm = LLMDeployment.bind()

# ─── Tier 2: Second-Class Citizens (standalone, not auto-deployed) ──────────
# florence2 — works but needs compat patches

# ─── Tier 3: Third-Class Citizens (broken / experimental / reference only) ──
# Uncomment to deploy individually for debugging.

# from services.tts.gpt_sovits import GPTSoVITSDeployment
# gpt_sovits = GPTSoVITSDeployment.bind()

# from services.audio.moss_soundeffect import MossSoundEffectDeployment
# moss_sfx = MossSoundEffectDeployment.bind()

# from services.asr.gpu_asr import VibeVoiceASRDeployment, QwenASRDeployment
# vibevoice_asr = VibeVoiceASRDeployment.bind()
# qwen_asr = QwenASRDeployment.bind()

# from services.tts.vibe_voice import VibeVoiceDeployment
# vibevoice = VibeVoiceDeployment.bind()

# from services.vision.florence2 import Florence2Deployment
# florence2 = Florence2Deployment.bind()

# from services.multimodal.phi4mm import Phi4MMDeployment
# phi4mm = Phi4MMDeployment.bind()


# from services.tts.qwen_tts_legacy import QwenTTSDeployment
# qwen_tts = QwenTTSDeployment.bind()
