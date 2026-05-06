"""KubeRay Serve application builders.

Each function returns a Ray Serve deployment graph for one service.
These are referenced by import_path in the RayService serveConfigV2.
"""


def kokoro_tts_app(args):
    """Kokoro CPU TTS — lightweight 82M model."""
    from services.tts.kokoro import KokoroTTSDeployment

    return KokoroTTSDeployment.bind()


def espeak_tts_app(args):
    """eSpeak CPU TTS — phoneme synthesis."""
    from services.tts.espeak import ESpeakTTSDeployment

    return ESpeakTTSDeployment.bind()


def faster_whisper_app(args):
    """Faster-Whisper CPU ASR."""
    from services.asr.faster_whisper import FasterWhisperDeployment

    return FasterWhisperDeployment.bind()
