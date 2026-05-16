"""MOSS-SoundEffect family handler — text-to-sound effect."""
import base64
import io
import logging
from pathlib import Path

import torch

# Shim: MOSS code imports PreTrainedConfig but transformers 4.57 uses PretrainedConfig
import transformers.configuration_utils as _tcu
if not hasattr(_tcu, 'PreTrainedConfig'):
    _tcu.PreTrainedConfig = _tcu.PretrainedConfig

# Shim: MOSS processing code uses MODALITY_TO_BASE_CLASS_MAPPING which doesn't exist in 4.57
from transformers import processing_utils as _pu
if not hasattr(_pu, 'MODALITY_TO_BASE_CLASS_MAPPING'):
    _pu.MODALITY_TO_BASE_CLASS_MAPPING = {}

from models.base_handler import BaseFamilyHandler, _make_handler_cls, audio_response

logger = logging.getLogger(__name__)


@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["moss-soundeffect"]
    FAMILY = "moss"
    FAMILY_INFOS = {"moss": (303, "MOSS SoundEffect")}
    DEFAULTS = {"prompt": "gentle rain"}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from registry.models import ModelRegistry
        model_path = Path(ModelRegistry().get_path("audio", "moss-soundeffect"))

        if not (model_path / "modeling_moss_tts.py").exists():
            raise FileNotFoundError(
                f"MOSS modeling_moss_tts.py not found at {model_path}. "
                f"Run 'task models:pull audio/moss-soundeffect' to download."
            )

        import importlib.util
        import types
        import sys

        def _load_module(name, filepath):
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Required MOSS file not found: {filepath}")
            spec = importlib.util.spec_from_file_location(name, filepath)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        # Create a fake parent package so relative imports resolve.
        # MOSS uses `from .configuration_moss_tts import ...` etc.
        moss_pkg = types.ModuleType("moss_tts_pkg")
        moss_pkg.__path__ = [str(model_path)]
        moss_pkg.__package__ = "moss_tts_pkg"
        sys.modules["moss_tts_pkg"] = moss_pkg

        # Load configuration first (no relative imports)
        config_mod = _load_module("moss_config", model_path / "configuration_moss_tts.py")
        sys.modules["moss_config"] = config_mod
        setattr(moss_pkg, "configuration_moss_tts", config_mod)

        # Load inference_utils and processing if present
        if (model_path / "inference_utils.py").exists():
            inf_mod = _load_module("moss_tts_pkg.inference_utils", model_path / "inference_utils.py")
            inf_mod.__package__ = "moss_tts_pkg"
            setattr(moss_pkg, "inference_utils", inf_mod)

        if (model_path / "processing_moss_tts.py").exists():
            proc_mod = _load_module("moss_tts_pkg.processing_moss_tts", model_path / "processing_moss_tts.py")
            proc_mod.__package__ = "moss_tts_pkg"
            setattr(moss_pkg, "processing_moss_tts", proc_mod)

        # Load modeling_moss_tts.py with the package context set
        modeling_mod = _load_module("moss_tts_pkg.modeling_moss_tts", model_path / "modeling_moss_tts.py")
        modeling_mod.__package__ = "moss_tts_pkg"

        MossTTSDelayConfig = config_mod.MossTTSDelayConfig
        MossTTSDelayModel = modeling_mod.MossTTSDelayModel

        # Register with transformers
        from transformers import AutoConfig, AutoModel
        AutoConfig.register("moss_tts_delay", MossTTSDelayConfig)
        AutoModel.register(MossTTSDelayConfig, MossTTSDelayModel)

        # Patch get_input_embeddings to be transformers-compatible
        _orig_get_emb = MossTTSDelayModel.get_input_embeddings
        if 'input_ids' in _orig_get_emb.__code__.co_varnames:
            def _patched_get_emb(self, input_ids=None):
                if input_ids is None:
                    return self.language_model.get_input_embeddings()
                return _orig_get_emb(self, input_ids)
            MossTTSDelayModel.get_input_embeddings = _patched_get_emb

        model = AutoModel.from_pretrained(
            str(model_path), torch_dtype=dtype or torch.bfloat16,
            local_files_only=True,
        )
        model.eval()

        # Patch model config — PretrainedConfig may set pad_token_id to None
        if model.config.pad_token_id is None:
            model.config.pad_token_id = 151643

        # Load processor components individually to bypass AutoProcessor type checks
        from transformers import AutoTokenizer, AutoConfig
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True,
        )
        model_config = AutoConfig.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True,
        )
        # Patch pad_token_id — PretrainedConfig may override it to None
        if model_config.pad_token_id is None:
            model_config.pad_token_id = 151643

        # Load audio tokenizer separately
        audio_tok_path = model_path / "audio_tokenizer"
        if not audio_tok_path.is_dir():
            audio_tok_path = Path(ModelRegistry().get_path("audio", "moss-audio-tokenizer"))
        audio_tokenizer = None
        if audio_tok_path.is_dir():
            audio_tokenizer = AutoModel.from_pretrained(
                str(audio_tok_path),
                torch_dtype=torch.float32, trust_remote_code=True, local_files_only=True,
            )

        # Construct processor manually — bypass AutoProcessor type check that
        # rejects MossAudioTokenizerModel as audio_tokenizer.
        # We call __init__ but with super().__init__ monkey-patched to accept anything.
        proc_mod = sys.modules.get("moss_tts_pkg.processing_moss_tts")
        if proc_mod is None:
            proc_mod = _load_module("moss_tts_pkg.processing_moss_tts", model_path / "processing_moss_tts.py")

        # Patch ProcessorMixin.__init__ to skip type checking for audio_tokenizer
        from transformers.processing_utils import ProcessorMixin
        _orig_init = ProcessorMixin.__init__
        def _patched_init(self, **kwargs):
            # Set attributes directly without type checks
            for k, v in kwargs.items():
                setattr(self, k, v)
        ProcessorMixin.__init__ = _patched_init

        try:
            processor = proc_mod.MossTTSDelayProcessor(
                tokenizer=tokenizer,
                audio_tokenizer=audio_tokenizer,
                model_config=model_config,
            )
        finally:
            ProcessorMixin.__init__ = _orig_init

        pipe = {
            "model": model,
            "audio_tokenizer": audio_tokenizer,
        }
        return _Pipeline(model, processor, audio_tokenizer), pipe


class _Pipeline:
    def __init__(self, model, processor, audio_tokenizer):
        self.model = model
        self.processor = processor
        self.audio_tokenizer = audio_tokenizer

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return next(self.model.parameters()).device

    def generate(self, *, input_prompt="", tokens=None, max_tokens=4096,
                 seed=-1, **kw):
        import soundfile as sf

        prompt = input_prompt or kw.get("prompt", "")
        if not prompt:
            raise ValueError("prompt required")

        batch_spec = {}
        if tokens is not None:
            batch_spec["tokens"] = tokens

        conversations = [
            [self.processor.build_user_message(ambient_sound=prompt, **batch_spec)]
        ]
        batch = self.processor(conversations, mode="generation")
        dev = self.device
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)

        # Model on CPU by default after AutoModel.from_pretrained.
        # Move to CUDA for GPU inference. If mmgp is active, .cuda()
        # is handled by mmgp's hooks (params stay on CPU pinned RAM).
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_tokens,
            )

        results = self.processor.decode(generated)
        if not results:
            raise RuntimeError("No audio generated")

        audio = results[0].audio_codes_list[0]
        sample_rate = self.processor.model_config.sampling_rate

        import scipy.io.wavfile as wavfile
        buf = io.BytesIO()
        wavfile.write(buf, sample_rate, audio.cpu().numpy())
        wav_data = buf.getvalue()

        return audio_response(wav_data)
