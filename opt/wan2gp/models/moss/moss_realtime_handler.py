"""MOSS-TTS-Realtime handler — streaming-capable TTS (1.7B, 16 RVQ channels).

Uses MossTTSRealtime model with step-wise inference via
MossTTSRealtimeInference (from streaming_mossttsrealtime.py).
"""
import importlib.util
import io
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

VARIANTS = {
    "moss-tts-realtime": {
        "hf_id": "OpenMOSS-Team/MOSS-TTS-Realtime",
        "registry": ("audio", "moss-tts-realtime"),
        "description": "Real-time TTS with streaming support",
    },
}

SHARED_AUDIO_TOKENIZER_REGISTRY = ("audio", "moss-audio-tokenizer")


def _resolve_model_path(model_type):
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
    from registry.models import ModelRegistry
    reg = ModelRegistry()
    cat, name = SHARED_AUDIO_TOKENIZER_REGISTRY
    try:
        return Path(reg.get_path(cat, name))
    except (KeyError, FileNotFoundError):
        from registry.config import Config
        models_root = Path(Config().models_root)
        from huggingface_hub import snapshot_download
        logger.info("Auto-downloading MOSS audio tokenizer")
        local_dir = str(models_root / cat / name)
        snapshot_download(repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer", local_dir=local_dir)
        return Path(local_dir)


def _load_realtime_modules(model_path):
    """Load the MossTTSRealtime code as a synthetic package."""
    import torch.nn.init as _init
    sys.modules.setdefault("transformers.initialization", _init)

    import transformers.processing_utils as _pu
    if not hasattr(_pu, 'MODALITY_TO_BASE_CLASS_MAPPING'):
        _pu.MODALITY_TO_BASE_CLASS_MAPPING = {}

    def _load_mod(name, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Required MOSS file not found: {filepath}")
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg_name = "moss_realtime_pkg"
    if pkg_name not in sys.modules or not isinstance(sys.modules[pkg_name], types.ModuleType):
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(model_path)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    config_mod = _load_mod(
        f"{pkg_name}.configuration_mossttsrealtime",
        model_path / "configuration_mossttsrealtime.py",
    )
    config_mod.__package__ = pkg_name
    setattr(sys.modules[pkg_name], "configuration_mossttsrealtime", config_mod)

    local_mod = _load_mod(
        f"{pkg_name}.modeling_mossttsrealtime_local",
        model_path / "modeling_mossttsrealtime_local.py",
    )
    local_mod.__package__ = pkg_name
    setattr(sys.modules[pkg_name], "modeling_mossttsrealtime_local", local_mod)

    modeling_mod = _load_mod(
        f"{pkg_name}.modeling_mossttsrealtime",
        model_path / "modeling_mossttsrealtime.py",
    )
    modeling_mod.__package__ = pkg_name

    proc_mod = _load_mod(
        f"{pkg_name}.processing_mossttsrealtime",
        model_path / "processing_mossttsrealtime.py",
    )
    proc_mod.__package__ = pkg_name
    setattr(sys.modules[pkg_name], "processing_mossttsrealtime", proc_mod)

    stream_mod = None
    if (model_path / "streaming_mossttsrealtime.py").exists():
        stream_mod = _load_mod(
            f"{pkg_name}.streaming_mossttsrealtime",
            model_path / "streaming_mossttsrealtime.py",
        )
        stream_mod.__package__ = pkg_name
        setattr(sys.modules[pkg_name], "streaming_mossttsrealtime", stream_mod)

    return config_mod, modeling_mod, proc_mod, stream_mod



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
        return {"moss": (303, "MOSS Audio")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "audio_only": True,
            "image_outputs": False,
            "inference_steps": False,
            "temperature": True,
            "top_k_slider": True,
            "duration_slider": {
                "label": "Max duration (seconds)",
                "min": 1, "max": 600, "increment": 1, "default": 30,
            },
            "alt_prompt": {
                "label": "Instruction (optional)",
                "placeholder": "speak in a warm voice",
                "lines": 2,
            },
        }

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        model_path = _resolve_model_path(base_model_type)
        config_mod, modeling_mod, proc_mod, stream_mod = _load_realtime_modules(model_path)

        MossTTSRealtimeConfig = config_mod.MossTTSRealtimeConfig
        MossTTSRealtime = modeling_mod.MossTTSRealtime
        MossTTSRealtimeProcessor = proc_mod.MossTTSRealtimeProcessor
        MossTTSRealtimeInference = stream_mod.MossTTSRealtimeInference if stream_mod else None

        from transformers import AutoConfig, AutoModel, AutoTokenizer

        # Register BEFORE loading config — AutoConfig needs to know the model type
        AutoConfig.register("moss_tts_realtime", MossTTSRealtimeConfig)
        AutoModel.register(MossTTSRealtimeConfig, MossTTSRealtime)

        # Load config separately to patch local_config before model instantiation
        model_config = AutoConfig.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        local_cfg = getattr(model_config, "local_config", None)
        if local_cfg is not None:
            if not hasattr(local_cfg, "rope_scaling") or local_cfg.rope_scaling is None:
                local_cfg.rope_scaling = {"type": "linear", "factor": 1.0}

        model = AutoModel.from_pretrained(
            str(model_path),
            torch_dtype=dtype or torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            config=model_config,
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True,
        )

        processor = MossTTSRealtimeProcessor(tokenizer=tokenizer)

        # Patch generate_local_transformer to remove @torch.compile.
        # The compiled tracer sees the codebook_idx <= 0 check in the
        # inputs_embeds-is-None branch (not taken at i=0), causing a
        # non-traceable ValueError → silent CUDA assert.
        if MossTTSRealtimeInference is not None:
            def _uncompiled_glt(self, hidden_states, temperature, top_p, top_k,
                                do_sample, repetition_penalty, repetition_window,
                                generated_tokens, gen_step):
                bs = hidden_states.shape[0]
                dev = hidden_states.device
                li = hidden_states.reshape(-1, 1, self.model.config.local_config.hidden_size)
                out = torch.empty(bs, self.channels, dtype=torch.long, device=dev)
                pkv = __import__("transformers").cache_utils.StaticCache(
                    config=self.model.local_transformer.config, max_cache_len=self.channels)
                lt = None
                cpt = torch.zeros(1, dtype=torch.long, device=dev)
                for i in range(self.channels):
                    cpt.fill_(i)
                    lo = self.model.local_transformer(
                        input_ids=lt, inputs_embeds=li,
                        past_key_values=pkv, cache_position=cpt,
                        codebook_idx=i, use_cache=True, logits_to_keep=1)
                    logits = lo.logits
                    if repetition_penalty and repetition_penalty != 1.0 and generated_tokens is not None:
                        logits = self.apply_repetition_penalty(
                            scores=logits, history_tokens=generated_tokens[:, :gen_step, i],
                            penalty=float(repetition_penalty), repetition_window=repetition_window)
                    lt = self.sample_token(logits, temperature, top_p, top_k, do_sample)
                    out[:, i] = lt.squeeze(-1)
                    if i == 0:
                        li = None
                return out
            MossTTSRealtimeInference.generate_local_transformer = _uncompiled_glt
            inference = MossTTSRealtimeInference(model, tokenizer)
        else:
            inference = None

        audio_tok_path = _resolve_audio_tokenizer_path()
        audio_tokenizer = None
        if audio_tok_path.is_dir():
            audio_tokenizer = AutoModel.from_pretrained(
                str(audio_tok_path),
                torch_dtype=torch.float32,
                trust_remote_code=True,
                local_files_only=True,
            )

        pipe = {"model": model}
        return _Pipeline(model, processor, inference, audio_tokenizer), pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "repeat_generation": 1,
            "temperature": 0.9,
            "top_k": 50,
            "prompt": "Hello, how are you today?",
        })

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            return "Prompt text cannot be empty."
        return None


class _Pipeline:
    def __init__(self, model, processor, inference, audio_tokenizer):
        self.model = model
        self.processor = processor
        self.inference = inference
        self.audio_tokenizer = audio_tokenizer

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return next(self.model.parameters()).device

    def generate(self, *, input_prompt="", max_tokens=4096, **kw):
        prompt = input_prompt or kw.get("text", "") or kw.get("prompt", "")
        if not prompt:
            raise ValueError("input_prompt or text required")

        if self.inference is None:
            raise RuntimeError("MossTTSRealtimeInference not available")

        dev = self.device
        self.model.to(dev)
        if self.audio_tokenizer is not None:
            self.audio_tokenizer.to(dev)

        with torch.no_grad():
            system_prompt = self.processor.make_ensemble(prompt_audio_tokens=None)
            system_prompt = torch.from_numpy(system_prompt).to(dev)

            text_ids = self.processor.tokenizer.encode(prompt, add_special_tokens=False)

            self.inference.reset_generation_state(keep_cache=False)
            audio_first = self.inference.prefill(
                input_ids=[system_prompt.cpu().numpy()],
                text_prefix_ids=[text_ids],
                temperature=0.8,
                top_p=0.6,
                top_k=30,
                do_sample=True,
                repetition_penalty=1.1,
                repetition_window=50,
            )

            audio_frames = self.inference.finish(
                max_steps=min(max_tokens, 2000),
                temperature=0.8,
                top_p=0.6,
                top_k=30,
                do_sample=True,
                repetition_penalty=1.1,
                repetition_window=50,
            )

        all_frames = [audio_first] + audio_frames
        audio_tokens = torch.cat(all_frames, dim=0).cpu()

        if self.audio_tokenizer is not None:
            self.audio_tokenizer.to(dev)
            nq, t = audio_tokens.shape[1], audio_tokens.shape[0]
            valid = audio_tokens.clone()
            eos_mask = (valid == 1026).any(dim=1)
            if eos_mask.any():
                first_eos = eos_mask.nonzero(as_tuple=True)[0][0].item()
                valid = valid[:first_eos]
            valid = valid.clamp(0, 1023)
            if valid.shape[0] == 0:
                raise RuntimeError("All audio frames filtered out (all EOS)")
            audio_codes = valid.permute(1, 0).unsqueeze(1).long().to(dev)
            decoded = self.audio_tokenizer.decode(
                audio_codes,
                num_quantizers=nq,
                chunk_duration=None,
            )
            wav = decoded.audio[0, 0]
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            audio_data = wav.cpu().float().numpy()
            sample_rate = getattr(self.audio_tokenizer.config, "sampling_rate", 24000)
        else:
            audio_data = np.zeros(24000, dtype=np.float32)
            sample_rate = 24000

        import scipy.io.wavfile as wavfile
        buf = io.BytesIO()
        wavfile.write(buf, sample_rate, audio_data)

        from models.base_handler import audio_response
        return audio_response(buf.getvalue())
