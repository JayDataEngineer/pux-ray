"""VibeVoice TTS family handler — multi-speaker TTS with voice cloning.

Pipeline decomposes VibeVoiceForConditionalGeneration into 7 mmgp-managed modules:
- language_model: Qwen2-based LM (autoregressive text + speech token generation)
- acoustic_tokenizer: encode/decode speech <-> acoustic latents
- semantic_tokenizer: speech -> semantic features
- acoustic_connector: acoustic -> LM space projection
- semantic_connector: semantic -> LM space projection
- prediction_head: DDPM diffusion head for speech generation
- lm_head: vocabulary projection

Inference: tokenize text → LM autoregressive loop → detect <|speech_diffusion|>
→ prediction_head sampling → acoustic_tokenizer.decode → audio waveform.
"""
import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


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
        paths = (model_def or {}).get("model_paths", {})
        model_path = Path(paths.get("vibevoice_tts", ""))

        # Register vibevoice architecture with transformers
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
        from transformers import AutoConfig, AutoModelForCausalLM
        AutoConfig.register("vibevoice", VibeVoiceConfig)
        AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGeneration)

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=dtype or torch.bfloat16,
            device_map="cpu", local_files_only=True,
        )
        model.eval()

        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        processor = VibeVoiceProcessor.from_pretrained(str(model_path))

        # Decompose into mmgp-managed modules
        inner = model.model
        pipe = {
            "language_model": inner.language_model,
            "acoustic_tokenizer": inner.acoustic_tokenizer,
            "semantic_tokenizer": inner.semantic_tokenizer,
            "acoustic_connector": inner.acoustic_connector,
            "semantic_connector": inner.semantic_connector,
            "prediction_head": inner.prediction_head,
            "lm_head": model.lm_head,
        }
        co_tenants = {"language_model": ["acoustic_tokenizer", "prediction_head"]}

        pipeline = _Pipeline(inner, processor, model.lm_head, pipe)
        return pipeline, {"pipe": pipe, "coTenantsMap": co_tenants}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"prompt": ""})


class _Pipeline:
    def __init__(self, inner_model, processor, lm_head, modules):
        self.inner = inner_model
        self.processor = processor
        self.lm_head = lm_head
        self.modules = modules

    @property
    def device(self):
        return next(self.inner.language_model.parameters()).device

    def generate(self, *, text="", language="English", ref_audio_b64=None,
                 max_tokens=4096, seed=-1, **kw):
        import soundfile as sf

        if not text:
            raise ValueError("text required")

        conversations = [
            {"role": "system", "content": f"Speak the following text in {language}."},
            {"role": "user", "content": text},
        ]
        text_inputs = self.processor.tokenizer.apply_chat_template(
            conversations, return_tensors="pt", return_dict=True,
        )
        input_ids = text_inputs["input_ids"].to(self.device)
        attn = text_inputs.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.device)

        speech_embeds = None
        if ref_audio_b64:
            ref_bytes = base64.b64decode(ref_audio_b64)
            audio_np, sr = sf.read(io.BytesIO(ref_bytes), dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            ref_tensor = torch.tensor(
                audio_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            acoustic_out = self.inner.acoustic_tokenizer.encode(ref_tensor)
            if isinstance(acoustic_out, torch.Tensor):
                acoustic_features = acoustic_out
            else:
                acoustic_features = getattr(acoustic_out, "last_hidden_state", acoustic_out)
            speech_embeds = self.inner.acoustic_connector(acoustic_features)

        with torch.no_grad():
            generated_tokens, speech_hidden = self._generate_loop(
                input_ids=input_ids, attention_mask=attn,
                speech_embeds=speech_embeds, max_new_tokens=max_tokens,
            )

        if speech_hidden is None:
            raise RuntimeError("No speech tokens generated")

        audio = self._generate_speech(speech_hidden)
        if isinstance(audio, torch.Tensor):
            audio = audio.float().cpu().numpy()

        sample_rate = 24000
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio.flatten(), sample_rate)
            wav_data = open(f.name, "rb").read()

        return {"status": "success", "data": base64.b64encode(wav_data).decode(),
                "media_type": "audio/wav"}

    def _generate_loop(self, input_ids, attention_mask, speech_embeds, max_new_tokens):
        from transformers import DynamicCache

        past_kv = DynamicCache()
        device = self.device
        embed_layer = self.inner.language_model.get_input_embeddings()

        inputs_embeds = embed_layer(input_ids)
        if speech_embeds is not None:
            inputs_embeds = torch.cat([inputs_embeds, speech_embeds], dim=1)
            if attention_mask is not None:
                speech_attn = torch.ones(
                    1, speech_embeds.shape[1],
                    dtype=attention_mask.dtype, device=device,
                )
                attention_mask = torch.cat([attention_mask, speech_attn], dim=1)

        out = self.inner.language_model.forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_kv, use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        logits = self.lm_head(hidden)
        token = torch.argmax(logits[:, -1, :], dim=-1)

        eos_token_id = self.processor.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        generated = [token.item()]
        speech_hidden_list = []
        collecting_speech = False
        speech_tokens = self._get_speech_token_ids()

        for _ in range(max_new_tokens - 1):
            if token.item() == eos_token_id:
                break
            tok_embed = embed_layer(token.unsqueeze(0))
            out = self.inner.language_model.forward(
                inputs_embeds=tok_embed, past_key_values=past_kv, use_cache=True,
            )
            hidden = out.last_hidden_state[:, -1:, :]
            logits = self.lm_head(hidden)
            token = torch.argmax(logits[:, -1, :], dim=-1)
            generated.append(token.item())

            tok_id = token.item()
            if tok_id == speech_tokens.get("diffusion"):
                collecting_speech = True
                speech_hidden_list.append(hidden.squeeze(0))
            elif collecting_speech:
                if tok_id == speech_tokens.get("end"):
                    collecting_speech = False
                else:
                    speech_hidden_list.append(hidden.squeeze(0))

        speech_hidden = torch.cat(speech_hidden_list, dim=0) if speech_hidden_list else None
        return generated, speech_hidden

    def _generate_speech(self, speech_hidden, num_steps=10):
        condition = speech_hidden.unsqueeze(0)
        scale = 1.0
        sf_attr = getattr(self.inner, "speech_scaling_factor", None)
        if sf_attr is not None and not (isinstance(sf_attr, torch.Tensor) and torch.isnan(sf_attr).any()):
            scale = sf_attr.item() if isinstance(sf_attr, torch.Tensor) else sf_attr

        vae_dim = getattr(self.inner.config, "acoustic_tokenizer_config", None)
        if vae_dim is not None:
            vae_dim = getattr(vae_dim, "vae_dim", 64)
        else:
            vae_dim = 64

        scheduler = getattr(self.inner, "noise_scheduler", None)
        if scheduler is not None:
            scheduler.set_timesteps(num_steps)
            latent_shape = (condition.shape[0], condition.shape[1], vae_dim)
            latents = torch.randn(latent_shape, device=self.device, dtype=torch.bfloat16)
            latents = latents * scheduler.init_noise_sigma
            for t in scheduler.timesteps:
                t_batch = t.unsqueeze(0).expand(latents.shape[0]).to(torch.bfloat16)
                noise_pred = self.inner.prediction_head.forward(latents, t_batch, condition)
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            latents = latents / scale
        else:
            latents = condition

        audio = self.inner.acoustic_tokenizer.decode(latents)
        if isinstance(audio, (list, tuple)):
            audio = audio[0]
        return audio

    def _get_speech_token_ids(self):
        tokenizer = self.processor.tokenizer
        return {
            "start": getattr(tokenizer, "speech_start_id", None) or
                     (tokenizer.convert_tokens_to_ids("<|speech_start|>") if hasattr(tokenizer, "convert_tokens_to_ids") else -1),
            "diffusion": getattr(tokenizer, "speech_diffusion_id", None) or
                         (tokenizer.convert_tokens_to_ids("<|speech_diffusion|>") if hasattr(tokenizer, "convert_tokens_to_ids") else -1),
            "end": getattr(tokenizer, "speech_end_id", None) or
                   (tokenizer.convert_tokens_to_ids("<|speech_end|>") if hasattr(tokenizer, "convert_tokens_to_ids") else -1),
        }
