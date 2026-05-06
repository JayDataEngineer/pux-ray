"""KubeRay Serve configuration — bound deployment graphs.

Each attribute is referenced by import_path in the RayService serveConfigV2.
All imports are safe at module level (heavy deps loaded lazily in _load()).
"""

# ─── CPU TTS (head node) ──────────────────────────────────────────────────────
from services.tts.kokoro import KokoroTTS
from services.tts.espeak import EspeakTTS

kokoro_tts = KokoroTTS.bind()
espeak_tts = EspeakTTS.bind()

# ─── CPU ASR (head node) ──────────────────────────────────────────────────────
from services.asr.faster_whisper import FasterWhisperASR

faster_whisper = FasterWhisperASR.bind()

# ─── GPU ASR/TTS (gpu-services image) ─────────────────────────────────────────
from services.asr.gpu_asr import VibeVoiceASRDeployment, QwenASRDeployment
from services.tts.gpu_tts import IndexTTSDeployment

vibevoice_asr = VibeVoiceASRDeployment.bind()
qwen_asr = QwenASRDeployment.bind()
index_tts = IndexTTSDeployment.bind()

# ─── GPU TTS — dedicated images ───────────────────────────────────────────────
from services.tts.qwen_tts import QwenTTSDeployment
from services.tts.vibe_voice import VibeVoiceDeployment

qwen_tts = QwenTTSDeployment.bind()
vibevoice = VibeVoiceDeployment.bind()

# ─── GPU Audio — dedicated images ─────────────────────────────────────────────
from services.audio.moss_soundeffect import MossSoundEffectDeployment
from services.audio.tangoflux import TangoFluxDeployment

moss_sfx = MossSoundEffectDeployment.bind()
tangoflux = TangoFluxDeployment.bind()

# ─── GPU Vision — dedicated images ────────────────────────────────────────────
from services.vision.florence2 import Florence2Deployment
from services.multimodal.phi4mm import Phi4MMDeployment

florence2 = Florence2Deployment.bind()
phi4mm = Phi4MMDeployment.bind()

# ─── GPU Creative — subprocess proxy (each in its own image) ──────────────────
from services.creative.trellis import TRELLISDeployment
from services.creative.hy_motion import HYMotionDeployment
from services.creative.anigen import AniGenDeployment
from services.creative.see_through import SeeThroughDeployment
from services.creative.ace_step import ACEStepDeployment
from services.image.comfyui import ComfyUIDeployment

trellis = TRELLISDeployment.bind()
hy_motion = HYMotionDeployment.bind()
anigen = AniGenDeployment.bind()
see_through = SeeThroughDeployment.bind()
ace_step = ACEStepDeployment.bind()
comfyui = ComfyUIDeployment.bind()

# ─── GPU TTS — subprocess proxy ───────────────────────────────────────────────
from services.tts.gpt_sovits import GPTSoVITSDeployment

gpt_sovits = GPTSoVITSDeployment.bind()
