"""Faster-Whisper family handler — CTranslate2 ASR, no nn.Modules."""
import base64
import tempfile
from pathlib import Path

from models.base_handler import BaseFamilyHandler, _make_handler_cls


@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["faster_whisper"]
    FAMILY = "faster_whisper"
    FAMILY_INFOS = {"faster_whisper": (301, "Faster-Whisper ASR")}
    DEFAULTS = {"language": "en"}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from faster_whisper import WhisperModel
        model_path = "deepdml/faster-whisper-large-v3-turbo-ct2"
        try:
            from registry.models import ModelRegistry
            local = ModelRegistry().get_path("asr", "faster-whisper")
            if Path(local).is_dir():
                model_path = str(local)
        except (KeyError, FileNotFoundError):
            pass
        model = WhisperModel(model_path, device="cpu", compute_type="int8")
        return _Pipeline(model), {}


class _Pipeline:
    def __init__(self, model):
        self.model = model

    def generate(self, *, audio_b64=None, audio_path=None, language=None,
                 beam_size=5, seed=-1, **kw):
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
        elif audio_path:
            audio_bytes = Path(audio_path).read_bytes()
        else:
            raise ValueError("audio_b64 or audio_path required")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            segments, info = self.model.transcribe(
                tmp_path, language=language, beam_size=beam_size, vad_filter=True,
            )
            segs, text = [], []
            for s in segments:
                segs.append({"start": s.start, "end": s.end, "text": s.text})
                text.append(s.text)
            return {"status": "success", "text": " ".join(text),
                    "segments": segs, "language": info.language,
                    "language_probability": info.language_probability}
        finally:
            Path(tmp_path).unlink(missing_ok=True)
