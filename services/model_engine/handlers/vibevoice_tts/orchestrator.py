"""VibeVoice TTS orchestrator — direct forward() calls on decomposed modules.
 
Inference flow:
1. Tokenize text via processor -> input_ids
2. language_model.forward(input_ids) -> hidden states (autoregressive)
3. lm_head(hidden_states) -> logits -> detect <|speech_diffusion|> token
4. prediction_head (diffusion sampling) -> speech latents
5. acoustic_tokenizer.decode(latents) -> audio waveform
"""
from __future__ import annotations

import base64
import io
import logging
import tempfile
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VibeVoiceTTSOrchestrator:
    """VibeVoice TTS inference via direct forward() on decomposed modules."""

    def __init__(self, modules):
        self.m = modules

    def generate(
        self,
        *,
        text: str = "",
        language: str = "English",
        attention_mask: Optional[torch.Tensor] = None,
        ref_audio_b64: Optional[str] = None,
        max_tokens: int = 4096,
        seed: int = -1,
    ) -> dict:
        import soundfile as sf

        if not text:
            raise ValueError("text required")

        conversations = [
            {"role": "system", "content": f"Speak the following text in {language}."},
            {"role": "user", "content": text},
        ]
        text_inputs = self.m.processor.tokenizer.apply_chat_template(
            conversations, return_tensors="pt", return_dict=True,
        )
        input_ids = text_inputs["input_ids"].to(self.m.device)
        attn = text_inputs.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.m.device)

        speech_embeds = None
        if ref_audio_b64:
            ref_bytes = base64.b64decode(ref_audio_b64)
            audio_np, sr = sf.read(io.BytesIO(ref_bytes), dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

            ref_tensor = torch.tensor(
                audio_np, dtype=torch.float32, device=self.m.device
            ).unsqueeze(0)
            ref_mask = torch.ones(
                ref_tensor.shape[1], dtype=torch.bool, device=self.m.device
            ).unsqueeze(0)

            acoustic_out = self.m.acoustic_tokenizer.encode(ref_tensor)
            if isinstance(acoustic_out, torch.Tensor):
                acoustic_features = acoustic_out
            else:
                acoustic_features = getattr(acoustic_out, "last_hidden_state", acoustic_out)

            speech_embeds = self.m.acoustic_connector(acoustic_features)

        with torch.no_grad():
            generated_tokens, speech_hidden = self._generate_loop(
                input_ids=input_ids,
                attention_mask=attn,
                speech_embeds=speech_embeds,
                max_new_tokens=max_tokens,
            )

        if speech_hidden is not None:
            audio = self._generate_speech(speech_hidden)
        else:
            raise RuntimeError("No speech tokens generated")

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        sample_rate = 24000
        if hasattr(self.m.processor, "audio_processor"):
            sr = getattr(self.m.processor.audio_processor, "sampling_rate", None)
            if sr is not None:
                sample_rate = sr if isinstance(sr, int) else 24000

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio.flatten(), sample_rate)
            wav_data = open(f.name, "rb").read()

        return {
            "status": "success",
            "data": base64.b64encode(wav_data).decode(),
            "media_type": "audio/wav",
        }

    def _generate_loop(self, input_ids, attention_mask, speech_embeds, max_new_tokens):
        from transformers import DynamicCache

        past_kv = DynamicCache()
        device = self.m.device
        embed_layer = self.m.language_model.get_input_embeddings()

        inputs_embeds = embed_layer(input_ids)
        if speech_embeds is not None:
            inputs_embeds = torch.cat([inputs_embeds, speech_embeds], dim=1)
            if attention_mask is not None:
                speech_attn = torch.ones(
                    1, speech_embeds.shape[1],
                    dtype=attention_mask.dtype, device=device,
                )
                attention_mask = torch.cat([attention_mask, speech_attn], dim=1)

        out = self.m.language_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        logits = self.m.lm_head(hidden)
        token = torch.argmax(logits[:, -1, :], dim=-1)

        eos_token_id = self.m.processor.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        generated = [token.item()]
        speech_hidden_list = []
        collecting_speech = False

        for _ in range(max_new_tokens - 1):
            if token.item() == eos_token_id:
                break

            tok_embed = embed_layer(token.unsqueeze(0))
            out = self.m.language_model.forward(
                inputs_embeds=tok_embed,
                past_key_values=past_kv,
                use_cache=True,
            )
            hidden = out.last_hidden_state[:, -1:, :]
            logits = self.m.lm_head(hidden)
            token = torch.argmax(logits[:, -1, :], dim=-1)
            generated.append(token.item())

            tok_id = token.item()
            speech_tokens = self._get_speech_token_ids()
            if tok_id == speech_tokens.get("diffusion"):
                collecting_speech = True
                speech_hidden_list.append(hidden.squeeze(0))
            elif collecting_speech:
                if tok_id == speech_tokens.get("end"):
                    collecting_speech = False
                else:
                    speech_hidden_list.append(hidden.squeeze(0))

        speech_hidden = None
        if speech_hidden_list:
            speech_hidden = torch.cat(speech_hidden_list, dim=0)

        return generated, speech_hidden

    def _generate_speech(self, speech_hidden):
        hidden = speech_hidden.unsqueeze(0)

        if self.m.speech_scaling_factor is not None:
            scale = self.m.speech_scaling_factor
            if isinstance(scale, torch.Tensor):
                scale = scale.item()
        else:
            scale = 1.0

        scheduler = self.m.noise_scheduler
        if scheduler is not None:
            num_steps = scheduler.config.num_inference_steps if hasattr(scheduler, "config") else 10
            scheduler.set_timesteps(num_steps)

            latent_shape = hidden.shape
            latents = torch.randn(latent_shape, device=self.m.device, dtype=self.m.dtype)
            latents = latents * scheduler.init_noise_sigma

            for t in scheduler.timesteps:
                noise_pred = self.m.prediction_head.forward(
                    hidden,
                    t.unsqueeze(0).expand(hidden.shape[0]),
                )
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            latents = latents / scale
        else:
            latents = self.m.prediction_head.forward(hidden)

        audio = self.m.acoustic_tokenizer.decode(latents)

        if isinstance(audio, (list, tuple)):
            audio = audio[0]

        return audio

    def _get_speech_token_ids(self):
        tokenizer = self.m.processor.tokenizer
        return {
            "start": getattr(tokenizer, "speech_start_id", None) or
                     (tokenizer.convert_tokens_to_ids("<|speech_start|>") if hasattr(tokenizer, "convert_tokens_to_ids") else -1),
            "diffusion": getattr(tokenizer, "speech_diffusion_id", None) or
                         (tokenizer.convert_tokens_to_ids("<|speech_diffusion|>") if hasattr(tokenizer, "convert_tokens_to_ids") else -1),
            "end": getattr(tokenizer, "speech_end_id", None) or
                   (tokenizer.convert_tokens_to_ids("<|speech_end|>") if hasattr(tokenizer, "convert_tokens_to_ids") else -1),
        }
