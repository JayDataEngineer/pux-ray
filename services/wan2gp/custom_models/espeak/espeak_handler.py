"""eSpeak-NG family handler — subprocess wrapper, no nn.Modules."""
import subprocess
import tempfile
from pathlib import Path

from models.base_handler import BaseFamilyHandler, _make_handler_cls, audio_response


@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["espeak"]
    FAMILY = "espeak"
    FAMILY_INFOS = {"espeak": (300, "eSpeak TTS")}
    DEFAULTS = {"prompt": "Hello world"}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from registry.config import Config
        bin_path = Config().get("binaries.espeak_ng", "espeak-ng")
        subprocess.run(["which", bin_path], capture_output=True, check=True)
        return _Pipeline(bin_path), {}


class _Pipeline:
    def __init__(self, bin_path):
        self.bin_path = bin_path

    def generate(self, *, input_prompt="", voice="en", speed=175, pitch=50,
                 seed=-1, **kw):
        text = input_prompt or kw.get("text", "")
        if not text:
            raise ValueError("text required")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out = tmp.name
        try:
            subprocess.run(
                [self.bin_path, "-v", str(voice), "-s", str(speed),
                 "-p", str(pitch), "-w", out, text],
                capture_output=True, check=True,
            )
            wav = Path(out).read_bytes()
        finally:
            Path(out).unlink(missing_ok=True)
        return audio_response(wav)
