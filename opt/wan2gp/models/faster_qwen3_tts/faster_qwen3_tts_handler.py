"""Faster Qwen3-TTS family handler — custom optimized pipeline.

NOT the vendor qwen3_handler which uses model.generate(). This decomposes
Qwen3TTSModel into 3 mmgp-managed modules for granular VRAM control:
talker, code_predictor, speech_tokenizer.

Inference via direct forward() calls: prefill → predictor loop → decode.
Modes: custom_voice (9 speakers), voice_clone, voice_design.
"""
import base64
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from models.base_handler import BaseFamilyHandler, _make_handler_cls, audio_response

logger = logging.getLogger(__name__)

SPEAKER_LANG = {
    "Vivian": "Chinese", "Serena": "Chinese", "Uncle_Fu": "Chinese",
    "Dylan": "Chinese", "Eric": "Chinese",
    "Ono_Anna": "Japanese",
    "Sohee": "Korean",
}



@_make_handler_cls
class family_handler(BaseFamilyHandler):
    SUPPORTED_TYPES = ["faster-qwen3-tts"]
    FAMILY = "faster_qwen3_tts"
    FAMILY_INFOS = {"faster_qwen3_tts": (306, "Faster Qwen3-TTS")}
    DEFAULTS = {"prompt": ""}

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   quantizeTransformer=False, text_encoder_quantization=None,
                   dtype=None, VAE_dtype=None, profile=0, **kwargs):
        from registry.models import ModelRegistry
        model_path = Path(ModelRegistry().get_path("tts", "qwen3-tts"))

        from faster_qwen3_tts.utils import suppress_flash_attn_warning
        with suppress_flash_attn_warning():
            from qwen_tts import Qwen3TTSModel

        base_model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            torch_dtype=dtype or torch.bfloat16,
            attn_implementation="sdpa",
        )
        base_model.model.cuda()
        base_model.model.eval()

        inner = base_model.model
        talker = inner.talker
        code_predictor = talker.code_predictor
        speech_tokenizer = inner.speech_tokenizer

        pipe = {
            "talker": talker,
            "speech_tokenizer": speech_tokenizer,
        }
        co_tenants = {}

        pipeline = _Pipeline(
            talker=talker,
            code_predictor=code_predictor,
            speech_tokenizer=speech_tokenizer,
            talker_model=talker.model,
            pred_model=code_predictor.model,
            pred_projection=code_predictor.small_to_mtp_projection,
            pred_lm_heads=code_predictor.lm_head,
            pred_codec_embeds=code_predictor.model.codec_embedding,
            talker_codec_embed=talker.get_input_embeddings(),
            talker_codec_head=talker.codec_head,
            base_model=base_model,
            talker_config=inner.config.talker_config,
            pred_config=code_predictor.model.config,
            sample_rate=int(getattr(speech_tokenizer, "sample_rate", 24000) or 24000),
        )
        return pipeline, {"pipe": pipe, "coTenantsMap": co_tenants}


def _sample_logits(logits, *, temperature=0.9, top_k=50, top_p=1.0,
                    do_sample=True, suppress_mask=None, suppress_tokens=None):
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


class _Pipeline:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, *, text="", mode="custom_voice", ref_audio_b64=None,
                 voice="Aiden", language="English", instruct="", ref_text="",
                 xvec_only=False, max_tokens=2048, min_tokens=2,
                 temperature=0.9, top_k=50, top_p=1.0, do_sample=True,
                 repetition_penalty=1.05, seed=-1, **kwargs):
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
        return audio_response(buf.read())

    def _custom_voice(self, *, text, voice, language, instruct, **kw):
        language = SPEAKER_LANG.get(voice, language)
        bm = self.base_model
        m = bm.model
        device = self.device
        input_texts = [bm._build_assistant_text(text)]
        input_ids = [t.to(device) for t in bm._tokenize_texts(input_texts)]
        instruct_ids = [bm._tokenize_texts([bm._build_instruct_text(instruct)])[0].to(device)] if instruct else [None]
        tie, tam, tth, tpe = self._build_talker_inputs(
            m=m, input_ids=input_ids, ref_ids=[None],
            voice_clone_prompt=None, languages=[language],
            speakers=[voice], instruct_ids=instruct_ids,
            non_streaming_mode=True,
        )
        codec_ids = self._generate_tokens(tie, tam, tth, tpe, **kw)
        if codec_ids is None:
            return [np.zeros(1, dtype=np.float32)], self.sample_rate
        audio_list, sr = self.speech_tokenizer.decode({"audio_codes": codec_ids.unsqueeze(0)})
        return [a.flatten().cpu().numpy() if hasattr(a, "cpu") else a.flatten() for a in audio_list], sr

    def _voice_clone(self, *, text, ref_audio_b64, language, ref_text, xvec_only, **kw):
        import soundfile as sf
        ref_bytes = base64.b64decode(ref_audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            ref_path = tmp.name
        try:
            m = self.base_model.model
            bm = self.base_model
            device = self.device
            input_texts = [bm._build_assistant_text(text)]
            input_ids = [t.to(device) for t in bm._tokenize_texts(input_texts)]
            if xvec_only:
                prompt_items = bm.create_voice_clone_prompt(
                    ref_audio=ref_path, ref_text="", x_vector_only_mode=True,
                )
                vcp = dict(ref_code=[None], ref_spk_embedding=[prompt_items[0].ref_spk_embedding],
                           x_vector_only_mode=[True], icl_mode=[False])
                ref_ids = [None]
            else:
                ref_audio_input, _ = self._load_ref_audio(ref_path)
                prompt_items = bm.create_voice_clone_prompt(ref_audio=ref_audio_input, ref_text=ref_text)
                vcp = bm._prompt_items_to_voice_clone_prompt(prompt_items)
                rt = prompt_items[0].ref_text
                ref_ids = [bm._tokenize_texts([bm._build_ref_text(rt)])[0].to(device) if rt else None]

            tie, tam, tth, tpe = self._build_talker_inputs(
                m=m, input_ids=input_ids, ref_ids=ref_ids,
                voice_clone_prompt=vcp, languages=[language],
                speakers=None, instruct_ids=[None], non_streaming_mode=False,
            )
            codec_ids = self._generate_tokens(tie, tam, tth, tpe, **kw)
            if codec_ids is None:
                return [np.zeros(1, dtype=np.float32)], self.sample_rate

            ref_codes = None
            if not xvec_only and vcp.get("ref_code") and vcp["ref_code"][0] is not None:
                ref_codes = vcp["ref_code"][0]
            if ref_codes is not None:
                codes_for_decode = torch.cat([ref_codes.to(codec_ids.device), codec_ids], dim=0)
            else:
                codes_for_decode = codec_ids

            audio_list, sr = self.speech_tokenizer.decode({"audio_codes": codes_for_decode.unsqueeze(0)})
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

    def _voice_design(self, *, text, instruct, language, **kw):
        bm = self.base_model
        m = bm.model
        device = self.device
        input_texts = [bm._build_assistant_text(text)]
        input_ids = [t.to(device) for t in bm._tokenize_texts(input_texts)]
        instruct_ids = [bm._tokenize_texts([bm._build_instruct_text(instruct)])[0].to(device) if instruct else None]
        tie, tam, tth, tpe = self._build_talker_inputs(
            m=m, input_ids=input_ids, ref_ids=[None],
            voice_clone_prompt=None, languages=[language],
            speakers=[None], instruct_ids=instruct_ids,
            non_streaming_mode=True,
        )
        codec_ids = self._generate_tokens(tie, tam, tth, tpe, **kw)
        if codec_ids is None:
            return [np.zeros(1, dtype=np.float32)], self.sample_rate
        audio_list, sr = self.speech_tokenizer.decode({"audio_codes": codec_ids.unsqueeze(0)})
        return [a.flatten().cpu().numpy() if hasattr(a, "cpu") else a.flatten() for a in audio_list], sr

    @torch.inference_mode()
    def _generate_tokens(self, talker_input_embeds, attention_mask,
                         trailing_text_hiddens, tts_pad_embed,
                         *, max_tokens, min_tokens, temperature, top_k, top_p,
                         do_sample, repetition_penalty):
        config = self.talker_config
        device = self.device
        eos_id = config.codec_eos_token_id
        num_code_groups = config.num_code_groups
        vocab_size = config.vocab_size

        suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        suppress_start = max(0, vocab_size - 1024)
        for i in range(suppress_start, vocab_size):
            if i != eos_id:
                suppress_mask[i] = True

        out = self.talker.forward(
            inputs_embeds=talker_input_embeds.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=True, output_hidden_states=True, return_dict=True,
            trailing_text_hidden=trailing_text_hiddens.to(device),
            tts_pad_embed=tts_pad_embed.to(device),
            generation_step=None, past_hidden=None, past_key_values=None,
        )
        past_kv = out.past_key_values
        past_hidden = out.past_hidden
        gen_step = out.generation_step
        logits = out.logits[:, -1, :]
        suppress_eos = min_tokens > 0
        token = _sample_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p,
                                do_sample=do_sample, suppress_mask=suppress_mask,
                                suppress_tokens=[eos_id] if suppress_eos else None)

        prefill_len = talker_input_embeds.shape[1]
        all_codec_ids = []

        for step_idx in range(max_tokens):
            if token.item() == eos_id:
                break
            last_id_hidden = self.talker_codec_embed(token.unsqueeze(1).to(device))
            pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
            codebook_token_ids = self._predictor_loop(pred_input)
            all_cb = torch.cat([token, codebook_token_ids])
            all_codec_ids.append(all_cb.detach())

            codec_hiddens = [last_id_hidden]
            for i in range(num_code_groups - 1):
                cb_emb = self.pred_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0))
                codec_hiddens.append(cb_emb)
            inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)

            if gen_step < trailing_text_hiddens.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hiddens[:, gen_step].unsqueeze(1).to(device)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed.to(device)

            cache_position = torch.tensor([prefill_len + step_idx], device=device)
            out = self.talker_model.forward(
                inputs_embeds=inputs_embeds, past_key_values=past_kv,
                cache_position=cache_position, use_cache=True,
            )
            hidden_states = out.last_hidden_state
            logits = self.talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)

            if repetition_penalty != 1.0 and all_codec_ids:
                history = torch.stack([c[0] for c in all_codec_ids])
                unique_toks = history.unique()
                tok_logits = logits[..., unique_toks]
                logits[..., unique_toks] = torch.where(
                    tok_logits > 0, tok_logits / repetition_penalty,
                    tok_logits * repetition_penalty,
                )

            suppress_eos = len(all_codec_ids) < min_tokens
            token = _sample_logits(logits.squeeze(0), temperature=temperature, top_k=top_k, top_p=top_p,
                                    do_sample=do_sample, suppress_mask=suppress_mask,
                                    suppress_tokens=[eos_id] if suppress_eos else None)
            past_hidden = hidden_states[:, -1:, :].clone()
            gen_step += 1

        return torch.stack(all_codec_ids) if all_codec_ids else None

    def _predictor_loop(self, pred_input):
        from transformers import DynamicCache
        device = self.device
        h = self.pred_projection(pred_input)
        pred_cache = DynamicCache()
        cache_pos = torch.arange(2, device=device)
        out = self.pred_model.forward(inputs_embeds=h, past_key_values=pred_cache,
                                       cache_position=cache_pos, use_cache=True)
        hidden = out.last_hidden_state
        logits = self.pred_lm_heads[0](hidden[:, -1:, :])
        tok = _sample_logits(logits[0, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
        tokens = [tok]
        for cb_idx in range(1, len(self.pred_lm_heads)):
            emb = self.pred_codec_embeds[cb_idx - 1](tok.unsqueeze(0).unsqueeze(0))
            emb = self.pred_projection(emb)
            cache_pos = torch.tensor([2 + cb_idx - 1], device=device)
            out = self.pred_model.forward(inputs_embeds=emb, past_key_values=pred_cache,
                                           cache_position=cache_pos, use_cache=True)
            hidden = out.last_hidden_state
            logits = self.pred_lm_heads[cb_idx](hidden[:, -1:, :])
            tok = _sample_logits(logits[0, 0, :], temperature=0.9, top_k=50, top_p=1.0, do_sample=True)
            tokens.append(tok)
        return torch.stack(tokens)

    def _build_talker_inputs(self, m, input_ids, ref_ids, voice_clone_prompt,
                             languages, speakers, instruct_ids, non_streaming_mode):
        from faster_qwen3_tts.model import FasterQwen3TTS
        return FasterQwen3TTS._build_talker_inputs_local(
            self=None, m=m, input_ids=input_ids, ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt, languages=languages,
            speakers=speakers, non_streaming_mode=non_streaming_mode,
            instruct_ids=instruct_ids,
        )

    def _load_ref_audio(self, ref_path, silence_secs=0.5):
        import soundfile as sf
        audio, sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if silence_secs > 0:
            silence = np.zeros(int(silence_secs * sr), dtype=np.float32)
            audio = np.concatenate([audio, silence])
        return audio, sr
