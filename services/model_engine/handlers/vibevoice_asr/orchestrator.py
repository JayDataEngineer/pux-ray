"""VibeVoice ASR orchestrator — direct forward() calls on decomposed modules.

Inference flow:
1. acoustic_tokenizer.encode(speech) -> acoustic features
2. acoustic_connector(features) -> projected to LM space
3. semantic_tokenizer.encode(speech) -> semantic features
4. semantic_connector(features) -> projected to LM space
5. Build input embeddings: text tokens + speech features at marked positions
6. language_model.forward(input_ids, inputs_embeds, ...) -> hidden states (autoregressive)
7. lm_head(hidden_states) -> logits -> decode text tokens
"""
from __future__ import annotations

import base64
import io
import logging

import torch

logger = logging.getLogger(__name__)


class VibeVoiceASROrchestrator:
    """VibeVoice ASR inference via direct forward() on decomposed modules."""

    def __init__(self, modules):
        self.m = modules

    def __call__(self, payload: dict) -> dict:
        return self.transcribe(payload)

    def transcribe(self, payload: dict) -> dict:
        import soundfile as sf

        # Get audio input
        audio_b64 = payload.get("audio_b64") or payload.get("audio")
        audio_path = payload.get("audio_path")

        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            audio_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        elif audio_path:
            audio_np, sr = sf.read(audio_path, dtype="float32")
        else:
            raise ValueError("audio_b64 or audio_path required")

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        speech_tensor = torch.tensor(
            audio_np, dtype=torch.float32, device=self.m.device
        ).unsqueeze(0)
        speech_mask = torch.ones(
            speech_tensor.shape[1], dtype=torch.bool, device=self.m.device
        ).unsqueeze(0)

        language = payload.get("language", "english")

        # 1. Encode speech through acoustic tokenizer
        acoustic_out = self.m.acoustic_tokenizer.encode(speech_tensor)
        if isinstance(acoustic_out, torch.Tensor):
            acoustic_features = acoustic_out
        else:
            acoustic_features = acoustic_out.last_hidden_state if hasattr(acoustic_out, "last_hidden_state") else acoustic_out[0]

        # 2. Project acoustic features to LM space
        acoustic_embeds = self.m.acoustic_connector(acoustic_features)

        # 3. Encode speech through semantic tokenizer
        semantic_out = self.m.semantic_tokenizer.encode(speech_tensor)
        if isinstance(semantic_out, torch.Tensor):
            semantic_features = semantic_out
        else:
            semantic_features = semantic_out.last_hidden_state if hasattr(semantic_out, "last_hidden_state") else semantic_out[0]

        # 4. Project semantic features to LM space
        semantic_embeds = self.m.semantic_connector(semantic_features)

        # 5. Build input embeddings with speech features
        # Use the processor to tokenize the text prompt
        prompt = f"Transcribe the following audio into {language}."
        conversations = [{"role": "user", "content": prompt}]
        text_inputs = self.m.processor.tokenizer.apply_chat_template(
            conversations, return_tensors="pt", return_dict=True,
        )
        input_ids = text_inputs["input_ids"].to(self.m.device)
        attention_mask = text_inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.m.device)

        # Build acoustic_input_mask: which positions get speech features
        text_len = input_ids.shape[1]
        acoustic_len = acoustic_embeds.shape[1]
        semantic_len = semantic_embeds.shape[1]

        # Insert speech features after text tokens
        # Build combined inputs_embeds: [text_embeds, acoustic_embeds, semantic_embeds]
        text_embeds = self.m.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([text_embeds, acoustic_embeds, semantic_embeds], dim=1)

        # Extend attention mask
        if attention_mask is not None:
            speech_attn = torch.ones(
                1, acoustic_len + semantic_len,
                dtype=attention_mask.dtype, device=self.m.device,
            )
            attention_mask = torch.cat([attention_mask, speech_attn], dim=1)

        # 6-7. Autoregressive generation via language_model.forward() + lm_head
        max_new_tokens = int(payload.get("max_tokens", 512))

        with torch.no_grad():
            generated_tokens = self._generate_loop(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )

        # Decode output tokens to text
        text = self.m.processor.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return {
            "status": "success",
            "text": text,
        }

    def _generate_loop(self, inputs_embeds, attention_mask, max_new_tokens):
        """Autoregressive decode: language_model.forward() -> lm_head -> sample."""
        from transformers import DynamicCache

        past_kv = DynamicCache()
        device = self.m.device

        # Prefill
        out = self.m.language_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        logits = self.m.lm_head(hidden)

        # Find EOS token from tokenizer
        eos_token_id = self.m.processor.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        token = torch.argmax(logits[:, -1, :], dim=-1)
        generated = [token.item()]

        # Get embeddings for decode
        embed_layer = self.m.language_model.get_input_embeddings()

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

        return generated
