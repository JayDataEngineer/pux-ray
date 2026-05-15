"""Faster-Whisper family handler — CTranslate2 ASR, no nn.Modules."""
import base64
import tempfile
from pathlib import Path


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["faster_whisper"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "faster_whisper"

    @staticmethod
    def query_family_infos():
        return {"faster_whisper": (301, "Faster-Whisper ASR")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {"audio_only": True, "image_outputs": False}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from faster_whisper import WhisperModel
        local = (model_def or {}).get("faster_whisper_path", "")
        model_path = str(local) if local and Path(local).is_dir() else "deepdml/faster-whisper-large-v3-turbo-ct2"
        model = WhisperModel(model_path, device="cpu", compute_type="int8")
        return _Pipeline(model), {}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"language": "en"})


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
