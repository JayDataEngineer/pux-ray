"""VibeVoice ASR family handler — speech-to-text.

Uses VibeVoice's acoustic_tokenizer + acoustic_connector to embed audio,
then runs the Qwen2 language_model autoregressively for transcription.
"""
import base64
import io
import logging
from pathlib import Path

import torch

from models.base_handler import BaseFamilyHandler, _make_handler_cls

logger = logging.getLogger(__name__)


@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["vibevoice-asr"]
    FAMILY = "vibevoice_asr"
    FAMILY_INFOS = {"vibevoice_asr": (304, "VibeVoice ASR")}
    DEFAULTS = {"language": "english"}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from registry.models import ModelRegistry
        model_path = Path(ModelRegistry().get_path("asr", "vibevoice-asr"))

        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
        from transformers import AutoConfig, AutoModel
        AutoConfig.register("vibevoice", VibeVoiceConfig)
        AutoModel.register(VibeVoiceConfig, VibeVoiceForConditionalGeneration)

        model = AutoModel.from_pretrained(
            str(model_path), torch_dtype=dtype or torch.bfloat16,
            local_files_only=True,
        )
        model.eval()

        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        processor = VibeVoiceProcessor.from_pretrained(str(model_path))

        return _Pipeline(model, processor), {}


class _Pipeline:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @property
    def device(self):
        return next(self.model.parameters()).device

    def generate(self, *, audio_b64=None, audio_path=None, language="english",
                 max_tokens=512, seed=-1, **kw):
        import soundfile as sf

        if audio_b64:
            audio_np, sr = sf.read(io.BytesIO(base64.b64decode(audio_b64)), dtype="float32")
        elif audio_path:
            audio_np, sr = sf.read(audio_path, dtype="float32")
        else:
            raise ValueError("audio_b64 or audio_path required")

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        speech = torch.tensor(audio_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        tokenizer = self.processor.tokenizer
        inner = self.model.model

        # Encode audio through acoustic_tokenizer + connector
        with torch.no_grad():
            enc_out = inner.acoustic_tokenizer.encode(speech.unsqueeze(1).to(torch.bfloat16))
            # encode returns VibeVoiceTokenizerEncoderOutput
            if hasattr(enc_out, 'mean'):
                frames = enc_out
            else:
                frames = enc_out[0][0]
            audio_tokens = frames.sample(inner.acoustic_tokenizer.std_dist_type)
            if isinstance(audio_tokens, (list, tuple)):
                audio_tokens = audio_tokens[0]
            audio_embeds = inner.acoustic_connector(audio_tokens)

        num_audio_tokens = audio_embeds.shape[1]

        # Build text prompt
        prompt = f"Transcribe the following audio into {language}."
        conversations = [{"role": "user", "content": prompt}]
        text_inputs = tokenizer.apply_chat_template(
            conversations, return_tensors="pt", return_dict=True,
            add_generation_prompt=True,
        )
        text_ids = text_inputs["input_ids"].to(self.device)
        text_attn = text_inputs.get("attention_mask", None)
        if text_attn is not None:
            text_attn = text_attn.to(self.device)

        # Build combined inputs_embeds: [text] + [audio] + [text_gen_prompt]
        text_embeds = self.model.get_input_embeddings()(text_ids)
        inputs_embeds = torch.cat([text_embeds, audio_embeds], dim=1)
        attention_mask = torch.cat([
            text_attn,
            torch.ones(1, num_audio_tokens, dtype=text_attn.dtype, device=self.device),
        ], dim=1)

        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        generated_tokens = []

        with torch.no_grad():
            out = inner.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            logits = self.model.lm_head(out.last_hidden_state[:, -1:, :])
            token = torch.argmax(logits[:, -1, :], dim=-1)
            generated_tokens.append(token.item())
            past_kv = out.past_key_values
            cur_embed = self.model.get_input_embeddings()(token.unsqueeze(0))

            for _ in range(max_tokens - 1):
                if token.item() == eos_token_id:
                    break
                out = inner.language_model(
                    inputs_embeds=cur_embed,
                    attention_mask=None,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                logits = self.model.lm_head(out.last_hidden_state[:, -1:, :])
                token = torch.argmax(logits[:, -1, :], dim=-1)
                generated_tokens.append(token.item())
                past_kv = out.past_key_values
                cur_embed = self.model.get_input_embeddings()(token.unsqueeze(0))

        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return {"status": "success", "text": text}
