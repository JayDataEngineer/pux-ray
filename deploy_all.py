"""Clean deploy ALL 15 services with Governor. Uses wrappers for problematic classes."""
import ray
ray.init(address="auto", ignore_reinit_error=True)
from ray import serve

# ── Create GPUGovernor ─────────────────────────────────────
from gateway.gpu_governor import GPUGovernor
try:
    governor = ray.get_actor("gpu_governor")
except ValueError:
    governor = GPUGovernor.options(name="gpu_governor", lifetime="detached").remote()

print(f"Governor: {ray.get(governor.status.remote())}")

# ── Import deployment classes ──────────────────────────────
# CPU services (with @serve.deployment in their modules)
from services.tts.kokoro import KokoroTTS
from services.tts.espeak import EspeakTTS
from services.asr.faster_whisper import FasterWhisperASR

# Lightweight GPU (with @serve.deployment in their modules)
from services.tts.faster_qwen3_tts import FasterQwen3TTSDeployment
from services.tts.gpu_tts import IndexTTSDeployment
from services.tts.vibevoice_cpp import VibeVoiceCppDeployment

# LLM (with @serve.deployment in its module)
from services.llm.deployment import LLMDeployment

# Heavy GPU (with @serve.deployment in their modules)
from services.creative.trellis import TRELLISDeployment
from services.creative.ace_step import ACEStepDeployment
from services.creative.see_through import SeeThroughDeployment
from services.creative.hy_motion import HYMotionDeployment
from services.image.comfyui import ComfyUIDeployment
from gateway.playground_deployment import PlaygroundDeployment

# Problematic classes (no @serve.deployment — need wrapper)
from services.audio.moss_soundeffect import MossSoundEffectDeployment as _Moss
from services.creative.anigen import AniGenDeployment as _Anigen

# ── Wrappers for problematic classes ───────────────────────
@serve.deployment(num_replicas=1, max_ongoing_requests=1, ray_actor_options={"num_gpus": 0})
class MossWrap(_Moss):
    pass

@serve.deployment(num_replicas=1, max_ongoing_requests=1, ray_actor_options={"num_gpus": 0})
class AnigenWrap(_Anigen):
    pass

# ── Deploy each ────────────────────────────────────────────
def deploy(app, name, route):
    try:
        serve.run(app, name=name, route_prefix=route)
        print(f"  ✅ {name} -> {route}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")

# Phase 1: CPU services (no GPU needed)
deploy(KokoroTTS.bind(), "kokoro_tts", "/tts/kokoro")
deploy(EspeakTTS.bind(), "espeak_tts", "/tts/espeak")
deploy(FasterWhisperASR.bind(), "faster_whisper", "/asr/whisper")

# Phase 2: Lightweight GPU (share GPU)
deploy(FasterQwen3TTSDeployment.bind(), "faster_qwen3_tts", "/tts/faster-qwen3-tts")
deploy(IndexTTSDeployment.bind(), "index_tts", "/tts/index-tts")
deploy(VibeVoiceCppDeployment.bind(), "vibevoice_cpp", "/tts/vibevoice-cpp")

# Phase 3: Heavy GPU (Governor managed)
deploy(LLMDeployment.bind(), "llm", "/llm")
deploy(TRELLISDeployment.bind(), "trellis", "/3d/trellis")
deploy(ACEStepDeployment.bind(), "ace_step", "/music/ace-step")
deploy(SeeThroughDeployment.bind(), "see_through", "/creative/see-through")
deploy(HYMotionDeployment.bind(), "hy_motion", "/3d/hy-motion")
deploy(ComfyUIDeployment.bind(), "comfyui", "/comfyui")
deploy(MossWrap.bind(), "moss_soundeffect", "/audio/moss-sfx")
deploy(AnigenWrap.bind(), "anigen", "/3d/anigen")

# Phase 4: Playground (UI)
deploy(PlaygroundDeployment.bind(), "playground", "/playground")

print("\n✅ All deployments complete")
print(f"Governor: {ray.get(governor.status.remote())}")
