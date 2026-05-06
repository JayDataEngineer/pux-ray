"""GPU service deployment bindings for KubeRay."""
from services.asr.gpu_asr import VibeVoiceASRDeployment, QwenASRDeployment
from services.tts.gpu_tts import IndexTTSDeployment

vibevoice_asr = VibeVoiceASRDeployment.bind()
qwen_asr = QwenASRDeployment.bind()
index_tts = IndexTTSDeployment.bind()
