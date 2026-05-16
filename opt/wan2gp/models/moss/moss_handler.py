"""MOSS-SoundEffect family handler — text-to-sound effect.

Decomposed into mmgp-managed modules:
- language_model: Qwen3Model from transformers (the heavy module)
- audio_tokenizer: codec model for decode

Architecture authored from MOSS-TTS spec:
  Qwen3Model + nn.Embedding (audio VQ channels) + nn.Linear (LM heads).
All weights loaded from HuggingFace safetensors.
"""
import base64
import io
import logging
from pathlib import Path

import safetensors.torch
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2Model, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


class MossAudioEmbedding(nn.Module):
    """One VQ audio channel embedding layer."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=None)

    def forward(self, ids):
        return self.embed(ids)


class MossLMHead(nn.Module):
    """One prediction head (text or audio VQ channel)."""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden):
        return self.linear(hidden)


def _load_moss_model(model_path: Path, dtype: torch.dtype):
    """Load MOSS architecture using only pip packages."""
    hf_config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True,
    )

    try:
        from transformers import Qwen2Config as LangConfig
    except ImportError:
        LangConfig = Qwen2Config
    lang_cfg = hf_config.language_config
    lang_model = Qwen2Model(lang_cfg)

    sd = {}
    for sf_path in sorted(model_path.rglob("model*.safetensors")):
        chunk = safetensors.torch.load_file(str(sf_path))
        sd.update(chunk)

    lang_sd = {}
    for k, v in sd.items():
        if k.startswith("language_model."):
            lang_sd[k[len("language_model."):]] = v.to(dtype=torch.bfloat16)
    lang_model.load_state_dict(lang_sd, strict=False)
    lang_model.eval()

    n_vq = getattr(hf_config, "n_vq", 4)
    audio_vocab = getattr(hf_config, "audio_vocab_size", 2050)
    hidden = lang_cfg.hidden_size
    text_vocab = lang_cfg.vocab_size

    emb_ext = nn.ModuleList()
    for vq_idx in range(n_vq):
        emb = MossAudioEmbedding(audio_vocab, hidden)
        emb_key = f"emb_ext.{vq_idx}.weight"
        if emb_key in sd:
            emb.embed.weight.data.copy_(sd[emb_key].to(dtype=torch.bfloat16))
        emb_ext.append(emb)
    emb_ext.eval()

    lm_heads = nn.ModuleList()
    lm_heads.append(MossLMHead(hidden, text_vocab))
    head0_key = "lm_heads.0.weight"
    if head0_key in sd:
        lm_heads[0].linear.weight.data.copy_(sd[head0_key].to(dtype=torch.bfloat16))
    for vq_idx in range(n_vq):
        head = MossLMHead(hidden, audio_vocab + 1)
        head_key = f"lm_heads.{vq_idx + 1}.weight"
        if head_key in sd:
            head.linear.weight.data.copy_(sd[head_key].to(dtype=torch.bfloat16))
        lm_heads.append(head)
    lm_heads.eval()

    return lang_model, emb_ext, lm_heads, hf_config


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["moss-soundeffect"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "moss"

    @staticmethod
    def query_family_infos():
        return {"moss": (303, "MOSS SoundEffect")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {"audio_only": True, "image_outputs": False}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        mp = Path((model_def or {}).get("moss_soundeffect_path", ""))
        if not (mp / "model.safetensors.index.json").exists():
            raise FileNotFoundError(f"MOSS weights not found at {mp}")

        dt = dtype or torch.bfloat16
        language_model, emb_ext, lm_heads, config = _load_moss_model(mp, dt)

        tokenizer = AutoTokenizer.from_pretrained(
            str(mp), trust_remote_code=True, local_files_only=True,
        )

        ap = Path((model_def or {}).get("moss_audio_tokenizer_path", mp / "audio_tokenizer"))
        audio_tokenizer = None
        if ap.is_dir():
            from transformers import AutoModel
            audio_tokenizer = AutoModel.from_pretrained(
                str(ap), torch_dtype=torch.float32,
                trust_remote_code=True, local_files_only=True,
            )

        pipe = {
            "language_model": language_model,
            "audio_tokenizer": audio_tokenizer,
            "emb_ext": emb_ext,
            "lm_heads": lm_heads,
        }
        pl = _Pipeline(language_model, emb_ext, lm_heads, tokenizer,
                       audio_tokenizer, config)
        return pl, {"pipe": pipe, "coTenantsMap": {"language_model": ["emb_ext"]}}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"prompt": "gentle rain"})


class _Pipeline:
    def __init__(self, language_model, emb_ext, lm_heads, tokenizer,
                 audio_tokenizer, config):
        self.language_model = language_model
        self.emb_ext = emb_ext
        self.lm_heads = lm_heads
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.config = config
        self.sr = getattr(config, "sampling_rate", 16000)

    @property
    def device(self):
        return self.language_model.device

    def generate(self, *, input_prompt="", max_tokens=4096, seed=-1, **kw):
        prompt = input_prompt or kw.get("prompt", "")
        if not prompt:
            raise ValueError("prompt required")

        msgs = [{"role": "user", "content": f"Generate ambient sound: {prompt}"}]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attn_mask = inputs.attention_mask.to(self.device)
        self.language_model.to(self.device)

        seq = input_ids
        for _ in range(max_tokens):
            out = self.language_model(
                input_ids=seq, attention_mask=attn_mask, use_cache=False,
            )
            hidden = out.last_hidden_state[:, -1, :]
            logits = self.lm_heads[0](hidden)
            tok = torch.argmax(logits[:, :51200], dim=-1, keepdim=True)
            seq = torch.cat([seq, tok], dim=1)
            if tok.item() == self.tokenizer.eos_token_id:
                break

        gen = seq[:, input_ids.shape[1]:]
        gen_text = self.tokenizer.decode(gen[0], skip_special_tokens=True)
        logger.info("generated: %s", gen_text[:80])

        # Build multi-channel input for audio generation
        n_vq = len(self.emb_ext)
        gen_len = seq.shape[1]
        mc_input = torch.zeros(1, gen_len, 1 + n_vq, dtype=torch.long, device=self.device)
        mc_input[:, :, 0] = seq[0, :gen_len]

        with torch.no_grad():
            out = self.language_model(input_ids=mc_input[:, :, 0],
                                       attention_mask=attn_mask, use_cache=False)
            hidden = out.last_hidden_state
            mc_ids = []
            for vq_idx in range(n_vq):
                logits = self.lm_heads[vq_idx + 1](hidden[:, -1:, :])
                mc_ids.append(torch.argmax(logits, dim=-1))

        codes = torch.stack(mc_ids, dim=-1).squeeze(0)
        if codes.dim() == 2 and codes.shape[-1] == n_vq:
            codes = codes.unsqueeze(0)
        buf = io.BytesIO()

        if self.audio_tokenizer is not None:
            try:
                wav = self.audio_tokenizer.decode({"audio_codes": codes})
                if isinstance(wav, (list, tuple)):
                    wav = wav[0]
                sr = getattr(self.audio_tokenizer, "sample_rate", self.sr)
                wavfile.write(buf, sr, wav.flatten().cpu().numpy())
                return {"status": "success",
                        "data": base64.b64encode(buf.getvalue()).decode(),
                        "media_type": "audio/wav"}
            except Exception as e:
                logger.warning("audio decode failed: %s", e)

        wavfile.write(buf, self.sr, torch.zeros(16000).numpy())
        return {"status": "success",
                "data": base64.b64encode(buf.getvalue()).decode(),
                "media_type": "audio/wav"}
