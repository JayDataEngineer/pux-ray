"""VibeVoice TTS family handler — multi-speaker TTS with voice cloning.

Decomposed into 7 mmgp-managed nn.Modules:
- language_model: Qwen2-based LM backbone
- acoustic_tokenizer: conv VAE codec encode/decode speech
- semantic_tokenizer: encoder-only semantic extractor
- acoustic_connector, semantic_connector: speech→LM projections
- prediction_head: DDPM diffusion denoiser
- lm_head: vocab projection

All model code authored in vibevoice_tts/ using only torch/transformers/diffusers.
"""
import json
import logging
from pathlib import Path

import safetensors.torch
import torch
from diffusers import DPMSolverMultistepScheduler
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .vibevoice_tts.blocks import (
    VibeVoiceAcousticTokenizer, VibeVoiceSemanticTokenizer, SpeechConnector,
)
from .vibevoice_tts.diffusion import VibeVoiceDiffusionHead

logger = logging.getLogger(__name__)


def _load_state_dict(model_path: Path):
    """Load all safetensors files and return flat state dict."""
    sd = {}
    for sf_path in sorted(model_path.rglob("model*.safetensors")):
        sd.update(safetensors.torch.load_file(str(sf_path)))
    return sd


def _load_and_strip(sd: dict, prefix: str, module, dtype=torch.bfloat16):
    """Load weights matching *prefix* into *module*, return leftover keys."""
    module_sd = {}
    rest = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            module_sd[k[len(prefix):].lstrip(".")] = v.to(dtype)
        else:
            rest[k] = v
    module.load_state_dict(module_sd, strict=False)
    return rest


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["vibevoice-tts"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "vibevoice_tts"

    @staticmethod
    def query_family_infos():
        return {"vibevoice_tts": (305, "VibeVoice TTS")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {"audio_only": True, "image_outputs": False}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        mp = Path((model_def or {}).get("vibevoice_tts_path", ""))
        if not (mp / "config.json").exists():
            raise FileNotFoundError(f"VibeVoice TTS not found at {mp}")

        with open(mp / "config.json") as f:
            cfg = json.load(f)
        dt = dtype or torch.bfloat16

        # 1. Language model (Qwen2 from transformers)
        lang_cfg = AutoConfig.for_model("qwen2", **cfg["decoder_config"])
        lang_cfg.torch_dtype = dt
        language_model = AutoModel.from_config(lang_cfg)
        sd = _load_state_dict(mp)
        sd = _load_and_strip(sd, "model.language_model", language_model, dt)

        # 2. Acoustic tokenizer (authored conv codec)
        acoustic_tokenizer = VibeVoiceAcousticTokenizer(cfg["acoustic_tokenizer_config"])
        sd = _load_and_strip(sd, "model.acoustic_tokenizer", acoustic_tokenizer)

        # 3. Semantic tokenizer (authored encoder-only)
        semantic_tokenizer = VibeVoiceSemanticTokenizer(cfg["semantic_tokenizer_config"])
        sd = _load_and_strip(sd, "model.semantic_tokenizer", semantic_tokenizer)

        # 4. Prediction head (authored diffusion)
        prediction_head = VibeVoiceDiffusionHead(cfg["diffusion_head_config"])
        sd = _load_and_strip(sd, "model.prediction_head", prediction_head)

        # 5. Connectors (authored Linear+RMSNorm+Linear)
        h = cfg["decoder_config"]["hidden_size"]
        acoustic_connector = SpeechConnector(cfg["acoustic_vae_dim"], h)
        sd = _load_and_strip(sd, "model.acoustic_connector", acoustic_connector)
        semantic_connector = SpeechConnector(cfg["semantic_vae_dim"], h)
        sd = _load_and_strip(sd, "model.semantic_connector", semantic_connector)

        # 6. LM head
        lm_head = torch.nn.Linear(h, cfg["decoder_config"]["vocab_size"], bias=False)
        lm_head_key = "lm_head.weight"
        if lm_head_key in sd:
            lm_head.weight.data.copy_(sd.pop(lm_head_key).to(dt))

        # 7. Noise scheduler (from diffusers pip)
        dh_cfg = cfg["diffusion_head_config"]
        noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=dh_cfg["ddpm_num_steps"],
            beta_schedule=dh_cfg.get("ddpm_beta_schedule", "cosine"),
            prediction_type=dh_cfg.get("prediction_type", "v_prediction"),
        )

        scaling_factor = sd.pop("model.speech_scaling_factor", torch.tensor(1.0)).float().item()
        tokenizer = AutoTokenizer.from_pretrained(
            str(mp), trust_remote_code=True, local_files_only=True)

        pipe = {
            "language_model": language_model,
            "acoustic_tokenizer": acoustic_tokenizer,
            "semantic_tokenizer": semantic_tokenizer,
            "prediction_head": prediction_head,
            "acoustic_connector": acoustic_connector,
            "semantic_connector": semantic_connector,
            "lm_head": lm_head,
        }
        co_tenants = {"language_model": ["lm_head"]}
        pl = _Pipeline(language_model, acoustic_tokenizer, semantic_tokenizer,
                       prediction_head, acoustic_connector, semantic_connector,
                       lm_head, noise_scheduler, tokenizer, scaling_factor)
        return pl, {"pipe": pipe, "coTenantsMap": co_tenants}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"prompt": ""})


class _Pipeline:
    def __init__(self, language_model, acoustic_tokenizer, semantic_tokenizer,
                 prediction_head, acoustic_connector, semantic_connector,
                 lm_head, noise_scheduler, tokenizer, scaling_factor):
        self.language_model = language_model
        self.acoustic_tokenizer = acoustic_tokenizer
        self.semantic_tokenizer = semantic_tokenizer
        self.prediction_head = prediction_head
        self.acoustic_connector = acoustic_connector
        self.semantic_connector = semantic_connector
        self.lm_head = lm_head
        self.noise_scheduler = noise_scheduler
        self.tokenizer = tokenizer
        self.scaling_factor = scaling_factor

    @property
    def device(self):
        return next(self.language_model.parameters()).device

    def generate(self, *, text="", language="English", ref_audio_b64=None,
                 max_tokens=4096, seed=-1, **kw):
        import soundfile as sf
        import base64, io

        if not text:
            raise ValueError("text required")

        dev = self.device
        conversations = [
            {"role": "system", "content": f"Speak the following text in {language}."},
            {"role": "user", "content": text},
        ]
        input_text = self.tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(dev)
        attn_mask = inputs.attention_mask.to(dev)
        text_len = input_ids.shape[1]

        # Autoregressive LM generation
        seq = input_ids
        past = None
        eos_id = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            out = self.language_model(
                input_ids=seq if past is None else seq[:, -1:],
                attention_mask=attn_mask if past is None else None,
                past_key_values=past, use_cache=True,
            )
            past = out.past_key_values
            logits = self.lm_head(out.last_hidden_state[:, -1, :])
            tok = torch.argmax(logits, dim=-1, keepdim=True)
            seq = torch.cat([seq, tok], dim=1)
            attn_mask = torch.cat([
                attn_mask,
                torch.ones(1, 1, dtype=attn_mask.dtype, device=dev),
            ], dim=1)
            if tok.item() == eos_id:
                break

        # Extract audio hidden states (re-run without cache for full sequence)
        with torch.no_grad():
            full_out = self.language_model(
                input_ids=seq, attention_mask=attn_mask,
                use_cache=False,
            )
        audio_hiddens = full_out.last_hidden_state[:, text_len:]

        # Project through acoustic connector: [B, T, acoustic_vae_dim]
        audio_feats = self.acoustic_connector(audio_hiddens)
        audio_feats = audio_feats * self.scaling_factor

        # Diffusion refinement via prediction head
        n_steps = 50
        self.noise_scheduler.set_timesteps(n_steps, device=dev)
        latents = torch.randn_like(audio_feats) * self.noise_scheduler.init_noise_sigma

        for t in self.noise_scheduler.timesteps:
            t_batch = t.expand(latents.shape[0])
            noise_pred = self.prediction_head(latents, t_batch, audio_feats)
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode to waveform via acoustic tokenizer
        latents = latents.transpose(1, 2)  # [B, vae_dim, T] for conv decoder
        audio = self.acoustic_tokenizer.decode(latents)
        audio_np = audio.squeeze(0).squeeze(0).cpu().float().numpy()

        buf = io.BytesIO()
        sf.write(buf, audio_np, 24000, format="WAV")
        return {"status": "success", "data": base64.b64encode(buf.getvalue()).decode(),
                "media_type": "audio/wav"}
