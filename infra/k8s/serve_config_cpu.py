"""CPU service deployment bindings for KubeRay."""
from services.tts.kokoro import KokoroTTSDeployment
from services.tts.espeak import ESpeakTTSDeployment
from services.asr.faster_whisper import FasterWhisperDeployment

kokoro_tts = KokoroTTSDeployment.bind()
espeak_tts = ESpeakTTSDeployment.bind()
faster_whisper = FasterWhisperDeployment.bind()
