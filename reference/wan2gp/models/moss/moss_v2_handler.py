"""MOSS SoundEffect v2.0 handler — DiT + Flow Matching (1.3B).

Architecture: 1.3B WanAudioModel (DiT) + Qwen3-1.7B text encoder + DAC VAE.
48kHz mono output, up to 30 seconds. Uses SDPA (PyTorch native flash attention).

~7GB VRAM total — fits entirely on GPU, no mmgp decomposition needed.
"""
import io
import logging
import os
import wave
from pathlib import Path

import numpy as np
import torch

from models.base_handler import BaseFamilyHandler, _make_handler_cls, audio_response

logger = logging.getLogger(__name__)

HF_REPO = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"

# Stub audiotools before importing moss_soundeffect_v2 — avoids descript-audiotools
# protobuf conflict. Only AudioSignal (compress/decompress) and BaseModel (base class)
# are referenced, neither used during inference.
import sys as _sys
_stub_dir = str(Path(__file__).parent.parent.parent / "moss_soundeffect_v2" / "diffsynth" / "_stubs")
if _stub_dir not in _sys.path:
    _sys.path.insert(0, _stub_dir)


def _resolve_model_path():
    from registry.models import ModelRegistry
    p = Path(ModelRegistry().get_path("audio", "moss-soundeffect-v2"))
    if not (p / "model_index.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=HF_REPO, local_dir=str(p))
    return p


@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["moss_soundeffect_v2"]
    FAMILY = "moss_v2"
    FAMILY_INFOS = {"moss_soundeffect_v2": (350, "MOSS SoundEffect v2.0")}
    DEFAULTS = {"prompt": "gentle rain", "seconds": 10, "steps": 100, "cfg_scale": 4.0}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, **kw):
        model_path = _resolve_model_path()

        # Disable torch.compile — incompatible with mmgp/shared GPU
        os.environ["TORCHDYNAMO_DISABLE"] = "1"

        # Load on CPU first (deployment.py sets set_default_device("cpu")),
        # then move to CUDA. MOSS v2 is ~7GB — fits entirely on GPU.
        from moss_soundeffect_v2 import MossSoundEffectPipeline
        pipe = MossSoundEffectPipeline.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
        )
        pipe.to("cuda")

        pipeline = _Pipeline(pipe)
        pipe_dict = {
            "type": "moss_soundeffect_v2",
            "family": "moss_v2",
            "sample_rate": 48000,
        }
        return pipeline, pipe_dict


class _Pipeline:
    def __init__(self, pipe):
        self.pipe = pipe

    def generate(self, *, input_prompt="", prompt="", seconds=10.0,
                 steps=100, cfg_scale=4.0, seed=-1, **kw):
        text = input_prompt or prompt or kw.get("text", "")
        if not text:
            raise ValueError("prompt required")

        if seed >= 0:
            torch.manual_seed(seed)

        audio = self.pipe(
            prompt=text,
            seconds=float(seconds),
            num_inference_steps=int(steps),
            cfg_scale=float(cfg_scale),
        )

        # audio: (B, C, T) tensor — take first sample
        waveform = audio[0, 0].cpu().float().numpy()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes((waveform * 32767).clip(-32768, 32767).astype("int16").tobytes())

        return audio_response(buf.getvalue())
