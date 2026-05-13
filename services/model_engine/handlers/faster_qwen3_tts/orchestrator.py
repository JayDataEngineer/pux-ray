"""FasterQwen3-TTS orchestrator — direct forward() calls on decomposed modules.
 
Inference flow (no CUDA graphs, no model.generate()):
1. Preprocess: build talker input embeddings via base_model methods
2. Prefill: talker.forward() → past_kv, past_hidden, first token logits
3. Decode loop (per token):
   a. talker_codec_embed(token) → embedding
   b. Predictor 15-step loop:
      - pred_projection(cat(past_hidden, embed)) → projected
      - pred_model.forward() → hidden state
      - pred_lm_heads[0](hidden) → codebook 0 token
      - For i in 1..14:
        - pred_codec_embeds[i-1](prev_token) → embed
        - pred_projection(embed) → projected
        - pred_model.forward() → hidden
        - pred_lm_heads[i](hidden) → codebook i token
   c. Build codec embedding from all 16 tokens (first + 15 codebooks)
   d. talker_model.forward() → hidden states (single decode step)
   e. talker_codec_head(hidden) → logits → sample next token
4. speech_tokenizer.decode() → audio waveform
"""
from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SPEAKER_LANG = {
    "Vivian": "Chinese", "Serena": "Chinese", "Uncle_Fu": "Chinese",
    "Dylan": "Chinese", "Eric": "Chinese",
    "Ono_Anna": "Japanese",
    "Sohee": "Korean",
}


def sample_logits(
    logits,
    *,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
    do_sample=True,
    suppress_mask=None,
    suppress_tokens=None,
):
    logits = logits.clone()
    if suppress_mask is not None:
        logits[..., suppress_mask] = float("-inf")
    if suppress_tokens:
        logits[..., list(suppress_tokens)] = float("-inf")
    if not do_sample:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < topk_vals[..., -1:], float("-inf"), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_indices, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)


class FasterQwen3TTSOrchestrator:
    """Qwen3-TTS inference via direct forward() on decomposed modules."""

    def __init__(self, modules):
        self.m = modules

    def generate(
        self,
        *,
        text: str = "",
        mode: str = "custom_voice",
        ref_audio_b64: Optional[str] = None,
        voice: str = "Aiden",
        language: str = "English",
        instruct: str = "",
        ref_text: str = "",
        xvec_only: bool = False,
        max_tokens: int = 2048,
        min_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        seed: int = -1,
    ) -> dict:
        import soundfile as sf

        if not text:
            raise ValueError("text required")

        if ref_audio_b64:
            audio_list, sr = self._voice_clone(
                text=text, ref_audio_b64=ref_audio_b64,
                language=language, ref_text=ref_text, xvec_only=xvec_only,
                max_tokens=max_tokens, min_tokens=min_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, repetition_penalty=repetition_penalty,
            )
        elif mode == "voice_design":
            audio_list, sr = self._voice_design(
                text=text, instruct=instruct, language=language,
                max_tokens=max_tokens, min_tokens=min_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, repetition_penalty=repetition_penalty,
            )
        else:
            audio_list, sr = self._custom_voice(
                text=text, voice=voice, language=language, instruct=instruct,
                max_tokens=max_tokens, min_tokens=min_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, repetition_penalty=repetition_penalty,
            )

        buf = io.BytesIO()
        sf.write(buf, audio_list[0], sr, format="WAV")
        buf.seek(0)
        wav_data = buf.read()

        return {
            "status": "success",
            "data": base64.b64encode(wav_data).decode(),
            "media_type": "audio/wav",
        }

    def _custom_voice(self, *, text, voice, language, instruct,
                      max_tokens, min_tokens, temperature, top_k, top_p,
                      do_sample, repetition_penalty):
        language = SPEAKER_LANG.get(voice, language)

        m = self.m.base_model.model
        input_texts = [m._build_assistant_text(text)]
        input_ids = m._tokenize_texts(input_texts)

        instruct_ids = [None]
        if instruct:
            instruct_ids = [m._tokenize_texts([m._build_instruct_text(instruct)])[0]]

        tie, tam, tth, tpe = self._build_talker_inputs(
            m=m,
            input_ids=input_ids,
            ref_ids=[None],
            voice_clone_prompt=None,
            languages=[language],
            speakers=[voice],
            instruct_ids=instruct_ids,
            non_streaming_mode=True,
        )

        codec_ids = self._generate_tokens(
            tie, tam, tth, tpe,
            max_tokens=max_tokens, min_tokens=min_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=do_sample, repetition_penalty=repetition_penalty,
        )

        if codec_ids is None:
            return [np.zeros(1, dtype=np.float32)], self.m.sample_rate

        audio_list, sr = self.m.speech_tokenizer.decode(
            {"audio_codes": codec_ids.unsqueeze(0)}
        )
        return [a.flatten().cpu().numpy() if hasattr(a, "cpu") else a.flatten() for a in audio_list], sr

    def _voice_clone(self, *, text, ref_audio_b64, language, ref_text, xvec_only,
                     max_tokens, min_tokens, temperature, top_k, top_p,
                     do_sample, repetition_penalty):
        import soundfile as sf

        ref_bytes = base64.b64decode(ref_audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            ref_path = tmp.name

        try:
            m = self.m.base_model.model
            input_texts = [m._build_assistant_text(text)]
            input_ids = m._tokenize_texts(input_texts)

            if xvec_only:
                prompt_items = m.create_voice_clone_prompt(
                    ref_audio=ref_path, ref_text="", x_vector_only_mode=True,
                )
                vcp = dict(
                    ref_code=[None],
                    ref_spk_embedding=[prompt_items[0].ref_spk_embedding],
                    x_vector_only_mode=[True],
                    icl_mode=[False],
                )
                ref_ids = [None]
            else:
                ref_audio_input = self._load_ref_audio(ref_path)
                prompt_items = m.create_voice_clone_prompt(
                    ref_audio=ref_audio_input, ref_text=ref_text,
                )
                vcp = m._prompt_items_to_voice_clone_prompt(prompt_items)
                rt = prompt_items[0].ref_text
                ref_ids = []
                if rt:
                    ref_ids.append(m._tokenize_texts([m._build_ref_text(rt)])[0])
                else:
                    ref_ids.append(None)

            tie, tam, tth, tpe = self._build_talker_inputs(
                m=m,
                input_ids=input_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=vcp,
                languages=[language],
                speakers=None,
                instruct_ids=[None],
                non_streaming_mode=False,
            )

            codec_ids = self._generate_tokens(
                tie, tam, tth, tpe,
                max_tokens=max_tokens, min_tokens=min_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, repetition_penalty=repetition_penalty,
            )

            if codec_ids is None:
                return [np.zeros(1, dtype=np.float32)], self.m.sample_rate

            ref_codes = None
            if not xvec_only and vcp.get("ref_code") and vcp["ref_code"][0] is not None:
                ref_codes = vcp["ref_code"][0]

            if ref_codes is not None:
                codes_for_decode = torch.cat([ref_codes.to(codec_ids.device), codec_ids], dim=0)
            else:
                codes_for_decode = codec_ids

            audio_list, sr = self.m.speech_tokenizer.decode(
                {"audio_codes": codes_for_decode.unsqueeze(0)}
            )

            audio_arrays = []
            for a in audio_list:
                a = a.flatten().cpu().numpy() if hasattr(a, "cpu") else a.flatten()
                if ref_codes is not None:
                    ref_len = ref_codes.shape[0]
                    total_len = codes_for_decode.shape[0]
                    cut = int(ref_len / max(total_len, 1) * len(a))
                    a = a[cut:]
                audio_arrays.append(a)

            return audio_arrays, sr
        finally:
            os.unlink(ref_path)

    def _voice_design(self, *, text, instruct, language,
                      max_tokens, min_tokens, temperature, top_k, top_p,
                      do_sample, repetition_penalty):
        m = self.m.base_model.model
        input_texts = [m._build_assistant_text(text)]
        input_ids = m._tokenize_texts(input_texts)
        instruct_ids = [m._tokenize_texts([m._build_instruct_text(instruct)])[0] if instruct else None]

        tie, tam, tth, tpe = self._build_talker_inputs(
            m=m,
            input_ids=input_ids,
            ref_ids=[None],
            voice_clone_prompt=None,
            languages=[language],
            speakers=[None],
            instruct_ids=instruct_ids,
            non_streaming_mode=True,
        )

        codec_ids = self._generate_tokens(
            tie, tam, tth, tpe,
            max_tokens=max_tokens, min_tokens=min_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=do_sample, repetition_penalty=repetition_penalty,
        )
        if codec_ids is None:
            return [np.zeros(1, dtype=np.float32)], self.m.sample_rate

        audio_list, sr = self.m.speech_tokenizer.decode(
            {"audio_codes": codec_ids.unsqueeze(0)}
        )
        return [a.flatten().cpu().numpy() if hasattr(a, "cpu") else a.flatten() for a in audio_list], sr

    @torch.inference_mode()
    def _generate_tokens(self, talker_input_embeds, attention_mask,
                         trailing_text_hiddens, tts_pad_embed,
                         *, max_tokens, min_tokens, temperature, top_k, top_p,
                         do_sample, repetition_penalty):
        m = self.m
        config = m.talker_config
        device = m.device

        eos_id = config.codec_eos_token_id
        num_code_groups = config.num_code_groups
        vocab_size = config.vocab_size

        suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        suppress_start = max(0, vocab_size - 1024)
        for i in range(suppress_start, vocab_size):
            if i != eos_id:
                suppress_mask[i] = True

        out = m.talker.forward(
            inputs_embeds=talker_input_embeds.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=trailing_text_hiddens.to(device),
            tts_pad_embed=tts_pad_embed.to(device),
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )

        past_kv = out.past_key_values
        past_hidden = out.past_hidden
        gen_step = out.generation_step

        logits = out.logits[:, -1, :]
        suppress_eos = min_tokens > 0
        token = sample_logits(
            logits,
            temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=[eos_id] if suppress_eos else None,
        )

        prefill_len = talker_input_embeds.shape[1]

        all_codec_ids = []

        for step_idx in range(max_tokens):
            if token.item() == eos_id:
                break

            last_id_hidden = m.talker_codec_embed(token.unsqueeze(1).to(device))

            pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
            codebook_token_ids = self._predictor_loop(pred_input)

            all_cb = torch.cat([token.view(1), codebook_token_ids])
            all_codec_ids.append(all_cb.detach())

            codec_hiddens = [last_id_hidden]
            for i in range(num_code_groups - 1):
                cb_emb = m.pred_codec_embeds[i](
                    codebook_token_ids[i].unsqueeze(0).unsqueeze(0)
                )
                codec_hiddens.append(cb_emb)
            inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)

            if gen_step < trailing_text_hiddens.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hiddens[:, gen_step].unsqueeze(1).to(device)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed.to(device)

            current_pos = prefill_len + step_idx
            cache_position = torch.tensor([current_pos], device=device)

            out = m.talker_model.forward(
                inputs_embeds=inputs_embeds,
                past_key_values=past_kv,
                cache_position=cache_position,
                use_cache=True,
            )

            hidden_states = out.last_hidden_state
            logits = m.talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)

            if repetition_penalty != 1.0 and len(all_codec_ids) > 0:
                history = torch.stack([c[0] for c in all_codec_ids])
                unique_toks = history.unique()
                tok_logits = logits[..., unique_toks]
                logits[..., unique_toks] = torch.where(
                    tok_logits > 0,
                    tok_logits / repetition_penalty,
                    tok_logits * repetition_penalty,
                )

            suppress_eos = len(all_codec_ids) < min_tokens
            token = sample_logits(
                logits.squeeze(0),
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample,
                suppress_mask=suppress_mask,
                suppress_tokens=[eos_id] if suppress_eos else None,
            )

            past_hidden = hidden_states[:, -1:, :].clone()
            gen_step += 1

        if all_codec_ids:
            return torch.stack(all_codec_ids)
        return None

    def _predictor_loop(self, pred_input):
        m = self.m
        device = m.device

        h = m.pred_projection(pred_input)

        from transformers import DynamicCache
        pred_cache = DynamicCache()

        cache_pos = torch.arange(2, device=device)
        out = m.pred_model.forward(
            inputs_embeds=h,
            past_key_values=pred_cache,
            cache_position=cache_pos,
            use_cache=True,
        )
        hidden = out.last_hidden_state

        logits = m.pred_lm_heads[0](hidden[:, -1:, :])
        tok = sample_logits(
            logits[:, 0, :],
            temperature=0.9, top_k=50, top_p=1.0, do_sample=True,
        )
        tokens = [tok]

        for cb_idx in range(1, len(m.pred_lm_heads)):
            emb = m.pred_codec_embeds[cb_idx - 1](tok.unsqueeze(0))
            emb = m.pred_projection(emb)

            cache_pos = torch.tensor([2 + cb_idx - 1], device=device)
            out = m.pred_model.forward(
                inputs_embeds=emb,
                past_key_values=pred_cache,
                cache_position=cache_pos,
                use_cache=True,
            )
            hidden = out.last_hidden_state

            logits = m.pred_lm_heads[cb_idx](hidden[:, -1:, :])
            tok = sample_logits(
                logits[:, 0, :],
                temperature=0.9, top_k=50, top_p=1.0, do_sample=True,
            )
            tokens.append(tok)

        return torch.stack(tokens)

    def _build_talker_inputs(self, m, input_ids, ref_ids, voice_clone_prompt,
                             languages, speakers, instruct_ids,
                             non_streaming_mode):
        from faster_qwen3_tts.model import FasterQwen3TTS
        tie, tam, tth, tpe = FasterQwen3TTS._build_talker_inputs_local(
            self=None,
            m=m,
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            instruct_ids=instruct_ids,
        )
        return tie, tam, tth, tpe

    def _load_ref_audio(self, ref_path, silence_secs=0.5):
        import soundfile as sf
        audio, sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if silence_secs > 0:
            silence = np.zeros(int(silence_secs * sr), dtype=np.float32)
            audio = np.concatenate([audio, silence])
        return audio, sr
