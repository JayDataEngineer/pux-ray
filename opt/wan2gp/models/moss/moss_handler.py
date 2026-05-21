"""MOSS family handler — OpenMOSS audio generation (MossTTSDelay architecture).

Supports 5 model variants, all sharing the same MossTTSDelay pipeline code:
  - moss-soundeffect: text-to-sound-effect (ambient_sound field)
  - moss-tts: text-to-speech with voice cloning (text + reference fields)
  - moss-ttsd: dialogue TTS (text field, multi-turn)
  - moss-voicegenerator: voice design from text (instruction field)
  - moss-tts-local-transformer: lighter local-transformer variant (1.7B)

Pattern follows Wan2GP's qwen3_handler.py and index_tts2_handler.py:
  - VARIANTS dict at module level
  - Per-variant model_def, defaults, validation
  - Raw family_handler with static methods (no base class needed)
"""
import importlib.util

from models.base_handler import HandlerHooks
import io
import logging
import types
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variant definitions — matches qwen3_handler.py QWEN3_TTS_VARIANTS pattern
# ---------------------------------------------------------------------------

VARIANTS = {
    "moss-soundeffect": {
        "hf_id": "OpenMOSS-Team/MOSS-SoundEffect",
        "registry": ("audio", "moss-soundeffect"),
        "description": "Text-to-sound-effect generation",
    },
    "moss-tts": {
        "hf_id": "OpenMOSS-Team/MOSS-TTS",
        "registry": ("audio", "moss-tts"),
        "description": "Text-to-speech with voice cloning",
    },
    "moss-ttsd": {
        "hf_id": "OpenMOSS-Team/MOSS-TTSD-v1.0",
        "registry": ("audio", "moss-ttsd"),
        "description": "Dialogue TTS",
    },
    "moss-voicegenerator": {
        "hf_id": "OpenMOSS-Team/MOSS-VoiceGenerator",
        "registry": ("audio", "moss-voicegenerator"),
        "description": "Voice design from text description",
    },
    "moss-tts-local-transformer": {
        "hf_id": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        "registry": ("audio", "moss-tts-local-transformer"),
        "description": "Lighter TTS with local transformer (1.7B)",
    },
}

MOSS_DURATION_SLIDER = {
    "label": "Max duration (seconds)",
    "min": 1,
    "max": 600,
    "increment": 1,
    "default": 30,
}

MOSS_SHARED_AUDIO_TOKENIZER_REGISTRY = ("audio", "moss-audio-tokenizer")

# ---------------------------------------------------------------------------
# Per-variant model_def — matches qwen3_handler.py get_qwen3_model_def()
# ---------------------------------------------------------------------------


def _get_moss_model_def(base_model_type):
    common = {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 0,
        "no_negative_prompt": True,
        "inference_steps": False,
        "temperature": True,
        "image_prompt_types_allowed": "",
        "supports_early_stop": True,
        "duration_slider": dict(MOSS_DURATION_SLIDER),
        "compile": False,
    }
    if base_model_type == "moss-soundeffect":
        return {
            **common,
            "profiles_dir": ["moss-soundeffect"],
            "top_k_slider": True,
        }
    if base_model_type == "moss-tts":
        return {
            **common,
            "profiles_dir": ["moss-tts"],
            "top_k_slider": True,
            "any_audio_prompt": True,
            "audio_prompt_choices": True,
            "audio_prompt_type_sources": {
                "selection": ["A"],
                "labels": {"A": "Voice cloning (reference audio)"},
                "letters_filter": "A",
                "default": "",
            },
            "alt_prompt": {
                "label": "Instruction (optional emotion/style)",
                "placeholder": "warm, friendly, slightly husky",
                "lines": 2,
            },
        }
    if base_model_type == "moss-ttsd":
        return {
            **common,
            "profiles_dir": ["moss-ttsd"],
            "top_k_slider": True,
            "any_audio_prompt": True,
            "audio_prompt_choices": True,
            "audio_prompt_type_sources": {
                "selection": ["A"],
                "labels": {"A": "Voice cloning (reference audio)"},
                "letters_filter": "A",
                "default": "",
            },
        }
    if base_model_type == "moss-voicegenerator":
        return {
            **common,
            "profiles_dir": ["moss-voicegenerator"],
            "top_k_slider": True,
            "alt_prompt": {
                "label": "Voice instruction",
                "placeholder": "A warm female voice with a gentle southern accent",
                "lines": 2,
            },
        }
    if base_model_type == "moss-tts-local-transformer":
        return {
            **common,
            "profiles_dir": ["moss-tts-local-transformer"],
            "top_k_slider": True,
            "any_audio_prompt": True,
            "audio_prompt_choices": True,
            "audio_prompt_type_sources": {
                "selection": ["A"],
                "labels": {"A": "Voice cloning (reference audio)"},
                "letters_filter": "A",
                "default": "",
            },
            "alt_prompt": {
                "label": "Instruction (optional emotion/style)",
                "placeholder": "warm, friendly, slightly husky",
                "lines": 2,
            },
        }
    return common


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_model_path(model_type):
    """Resolve local model path from registry, auto-download if missing."""
    from registry.models import ModelRegistry
    reg = ModelRegistry()
    variant = VARIANTS[model_type]
    cat, name = variant["registry"]

    try:
        path = Path(reg.get_path(cat, name))
        if path.is_dir() and (path / "config.json").exists():
            return path
    except (KeyError, FileNotFoundError):
        pass

    from registry.config import Config
    models_root = Path(Config().models_root)

    from huggingface_hub import snapshot_download
    hf_id = variant["hf_id"]
    logger.info("Auto-downloading %s from %s", model_type, hf_id)
    local_dir = str(models_root / cat / name)
    snapshot_download(repo_id=hf_id, local_dir=local_dir)
    return Path(local_dir)


def _resolve_audio_tokenizer_path():
    """Resolve the shared MOSS audio tokenizer path."""
    from registry.models import ModelRegistry
    reg = ModelRegistry()
    cat, name = MOSS_SHARED_AUDIO_TOKENIZER_REGISTRY
    try:
        return Path(reg.get_path(cat, name))
    except (KeyError, FileNotFoundError):
        from registry.config import Config
        models_root = Path(Config().models_root)

        from huggingface_hub import snapshot_download
        logger.info("Auto-downloading MOSS audio tokenizer")
        local_dir = str(models_root / cat / name)
        snapshot_download(
            repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer",
            local_dir=local_dir,
        )
        return Path(local_dir)


# ---------------------------------------------------------------------------
# Module loading — creates fake package for vendor relative imports
# ---------------------------------------------------------------------------


def _load_delay_modules(model_path):
    """Load the MossTTSDelay code as a synthetic package."""
    # Patch transformers for vendor code compatibility
    import torch.nn.init as _init
    sys.modules.setdefault("transformers.initialization", _init)

    import transformers.processing_utils as _pu
    if not hasattr(_pu, 'MODALITY_TO_BASE_CLASS_MAPPING'):
        _pu.MODALITY_TO_BASE_CLASS_MAPPING = {}

    import transformers.configuration_utils as _tcu
    if not hasattr(_tcu, 'PreTrainedConfig'):
        _tcu.PreTrainedConfig = _tcu.PretrainedConfig

    code_dir = model_path
    if not (code_dir / "modeling_moss_tts.py").exists():
        code_dir = Path(__file__).parent.parent.parent.parent / "vendor" / "moss-tts-delay"
    if not (code_dir / "modeling_moss_tts.py").exists():
        raise FileNotFoundError(
            f"MOSS delay code not found. Expected modeling_moss_tts.py "
            f"in model dir or vendor/moss-tts-delay/."
        )

    def _load_mod(name, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Required MOSS file not found: {filepath}")
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg_name = "moss_delay_pkg"
    if pkg_name not in sys.modules or not isinstance(sys.modules[pkg_name], types.ModuleType):
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(code_dir)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    config_mod = _load_mod(f"{pkg_name}.configuration_moss_tts", code_dir / "configuration_moss_tts.py")
    config_mod.__package__ = pkg_name
    setattr(sys.modules[pkg_name], "configuration_moss_tts", config_mod)

    if (code_dir / "inference_utils.py").exists():
        inf_mod = _load_mod(f"{pkg_name}.inference_utils", code_dir / "inference_utils.py")
        inf_mod.__package__ = pkg_name
        setattr(sys.modules[pkg_name], "inference_utils", inf_mod)

    if (code_dir / "processing_moss_tts.py").exists():
        proc_mod = _load_mod(f"{pkg_name}.processing_moss_tts", code_dir / "processing_moss_tts.py")
        proc_mod.__package__ = pkg_name
        setattr(sys.modules[pkg_name], "processing_moss_tts", proc_mod)

    modeling_mod = _load_mod(f"{pkg_name}.modeling_moss_tts", code_dir / "modeling_moss_tts.py")
    modeling_mod.__package__ = pkg_name

    return config_mod, modeling_mod


# ---------------------------------------------------------------------------
# Handler Hooks
# ---------------------------------------------------------------------------


class _MossHooks(HandlerHooks):
    needs_bf16_autocast = False
    needs_device_patch = False

    def before_generate(self, pipeline, kwargs):
        # Moss models inherit from PreTrainedModel whose device property
        # returns the parameter device (CPU — mmgp keeps weights there).
        # transformers generate() calls input_ids.to(self.device) which
        # moves inputs to CPU, but mmgp swaps modules to GPU for forward
        # → mismatch. Patch ALL sub-objects with a .device property.
        _cuda = torch.device("cuda:0")
        for attr_name in list(dir(pipeline)):
            if attr_name.startswith("_"):
                continue
            try:
                obj = getattr(pipeline, attr_name, None)
                if obj is not None and hasattr(type(obj), "device"):
                    type(obj).device = property(lambda self: _cuda)
            except Exception:
                pass
        return kwargs


HANDLER_META = {
    "input_type": "text",
    "output_type": "audio",
    "hooks": _MossHooks(),
}


# ---------------------------------------------------------------------------
# family_handler — Wan2GP discovery contract
# ---------------------------------------------------------------------------


class family_handler:
    @staticmethod
    def query_supported_types():
        return list(VARIANTS)

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "moss"

    @staticmethod
    def query_family_infos():
        return {
            "moss": (303, "MOSS Audio"),
        }

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return _get_moss_model_def(base_model_type)

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        variant = VARIANTS.get(base_model_type)
        if variant is None:
            raise ValueError(f"Unknown MOSS variant: {base_model_type}")

        model_path = _resolve_model_path(base_model_type)
        config_mod, modeling_mod = _load_delay_modules(model_path)

        MossTTSDelayConfig = config_mod.MossTTSDelayConfig
        MossTTSDelayModel = modeling_mod.MossTTSDelayModel

        from transformers import AutoConfig, AutoModel
        AutoConfig.register("moss_tts_delay", MossTTSDelayConfig)
        AutoModel.register(MossTTSDelayConfig, MossTTSDelayModel)

        # Patch get_input_embeddings if it expects positional input_ids
        _orig_get_emb = MossTTSDelayModel.get_input_embeddings
        if 'input_ids' in _orig_get_emb.__code__.co_varnames:
            def _patched_get_emb(self, input_ids=None):
                if input_ids is None:
                    return self.language_model.get_input_embeddings()
                return _orig_get_emb(self, input_ids)
            MossTTSDelayModel.get_input_embeddings = _patched_get_emb

        model = AutoModel.from_pretrained(
            str(model_path),
            torch_dtype=dtype or torch.bfloat16,
            local_files_only=True,
        )
        model.eval()

        if model.config.pad_token_id is None:
            model.config.pad_token_id = 151643

        if base_model_type == "moss-tts-local-transformer":
            language_config = getattr(model.config, "language_config", None)
            if language_config is not None:
                model.config.num_hidden_layers = language_config.num_hidden_layers
            if getattr(model.config, "num_attention_heads", None) is None:
                model.config.num_attention_heads = language_config.num_attention_heads

        from transformers import AutoTokenizer, AutoConfig as _AC
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True,
        )
        model_config = _AC.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True,
        )
        if model_config.pad_token_id is None:
            model_config.pad_token_id = 151643

        # Shared audio tokenizer
        audio_tok_path = _resolve_audio_tokenizer_path()
        audio_tokenizer = None
        if audio_tok_path.is_dir():
            audio_tokenizer = AutoModel.from_pretrained(
                str(audio_tok_path),
                torch_dtype=torch.float32,
                trust_remote_code=True,
                local_files_only=True,
            )

        # Build processor — patch ProcessorMixin to skip type checking
        proc_mod = sys.modules.get("moss_delay_pkg.processing_moss_tts")
        if proc_mod is None:
            from importlib import import_module
            proc_mod = import_module("moss_delay_pkg.processing_moss_tts")

        from transformers.processing_utils import ProcessorMixin
        _orig_init = ProcessorMixin.__init__

        def _patched_init(self, **kw):
            for k, v in kw.items():
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

        # Decompose into sub-modules so mmgp can swap them independently.
        # model.generate() calls forward() which uses all three — mmgp hooks
        # each module's forward() for just-in-time CPU↔GPU swapping.
        if base_model_type == "moss-tts-local-transformer":
            # Local-Transformer has different components (local_transformer,
            # speech_embedding_to_local_mlp, etc.) and no emb_ext. At 1.7B,
            # the whole model fits entirely in VRAM — no mmgp needed.
            pipe = {"model": model}
        else:
            pipe = {
                "transformer": model.language_model,  # Qwen3Model backbone (~16GB)
                "emb_ext": model.emb_ext,              # Audio VQ embeddings (32 small layers)
                "lm_heads": model.lm_heads,            # Text + audio prediction heads
            }
        # audio_tokenizer is NOT in the pipe dict — mmgp would intercept its
        # forward calls and cause dtype mismatches during decode.
        return _Pipeline(model, processor, audio_tokenizer, base_model_type), pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        duration_default = MOSS_DURATION_SLIDER["default"]
        common = {
            "repeat_generation": 1,
            "video_length": 0,
            "num_inference_steps": 0,
            "negative_prompt": "",
            "temperature": 0.9,
            "top_k": 50,
            "multi_prompts_gen_type": "FG",
            "duration_seconds": duration_default,
        }
        if base_model_type == "moss-soundeffect":
            ui_defaults.update({
                **common,
                "prompt": "gentle rain",
            })
        elif base_model_type == "moss-tts":
            ui_defaults.update({
                **common,
                "prompt": "Hello world",
                "alt_prompt": "",
                "audio_prompt_type": "",
            })
        elif base_model_type == "moss-ttsd":
            ui_defaults.update({
                **common,
                "prompt": "Hello, how are you today?",
                "audio_prompt_type": "",
            })
        elif base_model_type == "moss-voicegenerator":
            ui_defaults.update({
                **common,
                "prompt": "A warm female voice with a gentle southern accent",
                "alt_prompt": "",
            })
        elif base_model_type == "moss-tts-local-transformer":
            ui_defaults.update({
                **common,
                "prompt": "Hello world",
                "alt_prompt": "",
                "audio_prompt_type": "",
            })
        else:
            ui_defaults.update(common)

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            if base_model_type == "moss-voicegenerator":
                return "Voice instruction text cannot be empty."
            return "Prompt text cannot be empty."
        return None


# ---------------------------------------------------------------------------
# _Pipeline — inference wrapper
# ---------------------------------------------------------------------------


class _Pipeline:
    def __init__(self, model, processor, audio_tokenizer, model_type):
        self.model = model
        self.processor = processor
        self.audio_tokenizer = audio_tokenizer
        self.model_type = model_type

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return next(self.model.parameters()).device

    def _build_conversation(self, *, input_prompt="", reference=None,
                            instruction=None, tokens=None, language=None,
                            **kw):
        """Route parameters to the correct UserMessage fields per variant."""
        prompt = input_prompt or kw.get("text", "") or kw.get("prompt", "")
        msg_kwargs = {}

        if self.model_type == "moss-soundeffect":
            msg_kwargs["ambient_sound"] = prompt
            if tokens is not None:
                msg_kwargs["tokens"] = tokens
        elif self.model_type in ("moss-tts", "moss-tts-local-transformer"):
            msg_kwargs["text"] = prompt
            if reference:
                msg_kwargs["reference"] = [reference] if isinstance(reference, str) else reference
            if instruction:
                msg_kwargs["instruction"] = instruction
            if language:
                msg_kwargs["language"] = language
        elif self.model_type == "moss-ttsd":
            msg_kwargs["text"] = prompt
            if reference:
                msg_kwargs["reference"] = [reference] if isinstance(reference, str) else reference
            if language:
                msg_kwargs["language"] = language
        elif self.model_type == "moss-voicegenerator":
            msg_kwargs["instruction"] = prompt
            if language:
                msg_kwargs["language"] = language
        else:
            msg_kwargs["text"] = prompt

        return [self.processor.build_user_message(**msg_kwargs)]

    def generate(self, *, input_prompt="", reference=None, instruction=None,
                 tokens=None, language=None, max_tokens=4096, seed=-1, **kw):
        prompt = input_prompt or kw.get("text", "") or kw.get("prompt", "")
        if not prompt and self.model_type != "moss-voicegenerator":
            raise ValueError("input_prompt or text required")

        conversations = self._build_conversation(
            input_prompt=input_prompt, reference=reference,
            instruction=instruction, tokens=tokens, language=language, **kw,
        )
        batch = self.processor(conversations, mode="generation")
        dev = self.device
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)

        # No .cuda() — mmgp manages device placement via pipe dict hooks.
        # model.generate() → forward() → language_model/emb_ext/lm_heads
        # are swapped to GPU just-in-time by mmgp's profile() wrapper.
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_tokens,
            )

        # Move generated output to CPU for decode — audio tokenizer stays on
        # CPU (model uses ~22GB, no room for the 3.4GB tokenizer on GPU).
        gen_cpu = generated
        if isinstance(generated, torch.Tensor):
            gen_cpu = generated.cpu()
        elif isinstance(generated, (list, tuple)):
            gen_cpu = [t.cpu() if isinstance(t, torch.Tensor) else t for t in generated]

        results = self.processor.decode(gen_cpu)
        if not results:
            raise RuntimeError("No audio generated")

        audio = results[0].audio_codes_list[0]
        sample_rate = self.processor.model_config.sampling_rate

        import scipy.io.wavfile as wavfile
        buf = io.BytesIO()
        wavfile.write(buf, sample_rate, audio.cpu().numpy())
        wav_data = buf.getvalue()

        from models.base_handler import audio_response
        return audio_response(wav_data)
