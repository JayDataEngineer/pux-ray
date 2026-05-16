"""VibeVoice ASR family handler — speech-to-text.

Decomposed into mmgp-managed nn.Modules:
- language_model: Qwen2-based LM backbone
- acoustic_tokenizer: conv codec encoder for speech
- acoustic_connector: speech→LM projection
- lm_head: vocabulary projection
"""
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from models._shared import BaseFamilyHandler, load_safetensors, load_prefix_into_module, resolve_model_path

from .vibevoice_asr.blocks import VibeVoiceAcousticTokenizer, SpeechConnector

logger = logging.getLogger(__name__)


class family_handler(BaseFamilyHandler):
    FAMILY = "vibevoice_asr"
    FAMILY_ID = 304
    DISPLAY_NAME = "VibeVoice ASR"
    SUPPORTED_TYPES = ["vibevoice-asr"]
    AUDIO_ONLY = True
    UI_DEFAULTS = {"language": "english"}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from models._shared import resolve_model_path
        mp = resolve_model_path(
            "vibevoice_asr", "vibevoice_asr_path", model_def,
            check_file="config.json", category="asr",
            registry_name="vibevoice-asr", quant=kwargs.get("quant"),
        )
        if not (mp / "config.json").exists():
            raise FileNotFoundError(f"VibeVoice ASR not found at {mp}")

        with open(mp / "config.json") as f:
            cfg = json.load(f)
        dt = dtype or torch.bfloat16

        lang_cfg = AutoConfig.for_model("qwen2", **cfg["decoder_config"])
        lang_cfg.torch_dtype = dt
        language_model = AutoModel.from_config(lang_cfg)
        sd = load_safetensors(mp)
        sd = load_prefix_into_module(sd, "model.language_model", language_model, dt)

        acoustic_tokenizer = VibeVoiceAcousticTokenizer(cfg["acoustic_tokenizer_config"])
        sd = load_prefix_into_module(sd, "model.acoustic_tokenizer", acoustic_tokenizer)

        h = cfg["decoder_config"]["hidden_size"]
        acoustic_connector = SpeechConnector(cfg.get("acoustic_vae_dim", 64), h)
        sd = load_prefix_into_module(sd, "model.acoustic_connector", acoustic_connector)

        lm_head = nn.Linear(h, cfg["decoder_config"]["vocab_size"], bias=False)
        lm_head_key = "lm_head.weight"
        if lm_head_key in sd:
            lm_head.weight.data.copy_(sd.pop(lm_head_key).to(dt))

        tokenizer = AutoTokenizer.from_pretrained(
            str(mp), trust_remote_code=True, local_files_only=True)

        pipe = {
            "language_model": language_model,
            "acoustic_tokenizer": acoustic_tokenizer,
            "acoustic_connector": acoustic_connector,
            "lm_head": lm_head,
        }
        co_tenants = {"language_model": ["lm_head"]}
        pl = _Pipeline(language_model, acoustic_tokenizer, acoustic_connector,
                       lm_head, tokenizer)
        return pl, {"pipe": pipe, "coTenantsMap": co_tenants}


class _Pipeline:
    def __init__(self, language_model, acoustic_tokenizer, acoustic_connector,
                 lm_head, tokenizer):
        self.language_model = language_model
        self.acoustic_tokenizer = acoustic_tokenizer
        self.acoustic_connector = acoustic_connector
        self.lm_head = lm_head
        self.tokenizer = tokenizer

    @property
    def device(self):
        return next(self.language_model.parameters()).device

    def generate(self, *, audio_b64=None, audio_path=None, language="english",
                 max_tokens=512, seed=-1, **kw):
        import base64, io, soundfile as sf

        if audio_b64:
            audio_np, sr = sf.read(io.BytesIO(base64.b64decode(audio_b64)), dtype="float32")
        elif audio_path:
            audio_np, sr = sf.read(audio_path, dtype="float32")
        else:
            raise ValueError("audio_b64 or audio_path required")

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        dev = self.device
        speech = torch.tensor(audio_np, dtype=torch.float32, device=dev).unsqueeze(0)

        with torch.no_grad():
            frames = self.acoustic_tokenizer.encode(speech)
            audio_embeds = self.acoustic_connector(frames.transpose(1, 2))

        prompt = f"Transcribe the following audio into {language}."
        conversations = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        text_ids = inputs.input_ids.to(dev)
        text_attn = inputs.attention_mask.to(dev)

        text_embeds = self.language_model.get_input_embeddings()(text_ids)
        audio_embeds = audio_embeds.to(dev)
        inputs_embeds = torch.cat([text_embeds, audio_embeds], dim=1)
        num_audio = audio_embeds.shape[1]
        attn_mask = torch.cat([
            text_attn,
            torch.ones(1, num_audio, dtype=text_attn.dtype, device=dev),
        ], dim=1)

        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]

        past = None
        seq_len = inputs_embeds.shape[1]
        generated = []

        out = self.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask,
            use_cache=True, return_dict=True,
        )
        past = out.past_key_values
        logits = self.lm_head(out.last_hidden_state[:, -1, :])
        tok = torch.argmax(logits, dim=-1)
        generated.append(tok.item())

        for _ in range(max_tokens - 1):
            if tok.item() == eos_id:
                break
            tok_embed = self.language_model.get_input_embeddings()(tok.unsqueeze(0).unsqueeze(0))
            out = self.language_model(
                inputs_embeds=tok_embed, past_key_values=past,
                use_cache=True, return_dict=True,
            )
            past = out.past_key_values
            logits = self.lm_head(out.last_hidden_state[:, -1, :])
            tok = torch.argmax(logits, dim=-1)
            generated.append(tok.item())

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {"status": "success", "text": text}
