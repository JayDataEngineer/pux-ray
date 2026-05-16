"""eSpeak-NG family handler — subprocess wrapper, no nn.Modules."""
import base64
import subprocess
import tempfile
from pathlib import Path

from models._shared import BaseFamilyHandler


class family_handler(BaseFamilyHandler):
    FAMILY = "espeak"
    FAMILY_ID = 300
    DISPLAY_NAME = "eSpeak TTS"
    SUPPORTED_TYPES = ["espeak"]
    AUDIO_ONLY = True
    UI_DEFAULTS = {"prompt": "Hello world"}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        bin_path = (model_def or {}).get("espeak_bin", "espeak-ng")
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
        return {"status": "success", "data": base64.b64encode(wav).decode(),
                "media_type": "audio/wav"}
