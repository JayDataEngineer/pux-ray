"""MOSS-SoundEffect family handler — text-to-sound effect.

Decomposed into mmgp-managed modules:
- language_model: Qwen3Model from transformers (the heavy module)
- audio_tokenizer: codec model for decode (CPU-resident)
- emb_ext: audio VQ channel embeddings
- lm_heads: multi-head prediction (text + 16 audio VQ channels)

Architecture: Qwen3Model + nn.Embedding (audio VQ) + nn.Linear (LM heads).
All weights loaded from HuggingFace safetensors. Generation uses the delay
pattern from the upstream MossTTSDelayModel.
"""
import base64
import io
import logging
from pathlib import Path

import safetensors.torch
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig

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

    from transformers import AutoModel
    lang_cfg = hf_config.language_config
    lang_model = AutoModel.from_config(lang_cfg)

    sd = {}
    for sf_path in sorted(model_path.rglob("model*.safetensors")):
        chunk = safetensors.torch.load_file(str(sf_path))
        sd.update(chunk)

    # Load language model weights
    lang_prefix = "language_model."
    lang_sd = {}
    for k, v in sd.items():
        if k.startswith(lang_prefix):
            lang_sd[k[len(lang_prefix):]] = v.to(dtype=dtype)
    lang_model.load_state_dict(lang_sd, strict=False)
    lang_model.eval()

    # Audio VQ embeddings
    n_vq = getattr(hf_config, "n_vq", 16)
    audio_vocab = getattr(hf_config, "audio_vocab_size", 1024)
    hidden = lang_cfg.hidden_size
    text_vocab = lang_cfg.vocab_size

    emb_ext = nn.ModuleList()
    for vq_idx in range(n_vq):
        emb = MossAudioEmbedding(audio_vocab, hidden)
        emb_key = f"emb_ext.{vq_idx}.weight"
        if emb_key in sd:
            emb.embed.weight.data.copy_(sd[emb_key].to(dtype=dtype))
        emb_ext.append(emb)
    emb_ext.eval()

    lm_heads = nn.ModuleList()
    # Head 0: text
    text_head = MossLMHead(hidden, text_vocab)
    head0_key = "lm_heads.0.weight"
    if head0_key in sd:
        text_head.linear.weight.data.copy_(sd[head0_key].to(dtype=dtype))
    lm_heads.append(text_head)
    # Heads 1..n_vq: audio VQ channels
    for vq_idx in range(n_vq):
        head = MossLMHead(hidden, audio_vocab + 1)
        head_key = f"lm_heads.{vq_idx + 1}.weight"
        if head_key in sd:
            head.linear.weight.data.copy_(sd[head_key].to(dtype=dtype))
        lm_heads.append(head)
    lm_heads.eval()

    return lang_model, emb_ext, lm_heads, hf_config


def _apply_top_k(logits, top_k):
    top_k = min(top_k, logits.shape[-1])
    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    batch_idx = torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1)
    filtered[batch_idx, top_k_indices] = top_k_values
    return filtered


def _apply_top_p(logits, top_p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = logits.clone()
    for i in range(logits.shape[0]):
        filtered[i, sorted_indices[i][remove[i]]] = float("-inf")
    return filtered


def _sample_token(logits, temperature=1.5, top_p=0.6, top_k=50, do_sample=True):
    if not do_sample or temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    if top_k and top_k > 0:
        logits = _apply_top_k(logits, top_k)
    if top_p and top_p < 1.0:
        logits = _apply_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _find_last_equal(tensor, value):
    """Find last occurrence of value per row. Returns -1 if not found."""
    mask = (tensor == value).int()
    flipped = mask.flip(dims=[1])
    flipped_idx = flipped.argmax(dim=1)
    seq_len = tensor.shape[1]
    last_idx = (seq_len - 1) - flipped_idx
    actual = tensor[torch.arange(tensor.shape[0], device=tensor.device), last_idx]
    last_idx[actual != value] = -1
    return last_idx


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
            try:
                from transformers import AutoModel
                audio_tokenizer = AutoModel.from_pretrained(
                    str(ap), torch_dtype=torch.float32,
                    trust_remote_code=True, local_files_only=True,
                )
            except Exception as e:
                logger.warning("Audio tokenizer load failed: %s", e)
                logger.warning("Continuing without audio tokenizer")

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
        self.n_vq = getattr(config, "n_vq", 16)
        self.audio_vocab = getattr(config, "audio_vocab_size", 1024)
        self.audio_pad_code = getattr(config, "audio_pad_code", self.audio_vocab)
        self.sr = getattr(config, "sampling_rate", 24000)

    @property
    def device(self):
        return self.language_model.device

    def _get_input_embeddings(self, input_ids):
        """Compute combined embeddings from text + audio VQ channels.
        input_ids: (batch, seq, 1 + n_vq)
        """
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids[..., 0])
        for i, embed_layer in enumerate(self.emb_ext):
            inputs_embeds = inputs_embeds + embed_layer(input_ids[..., i + 1])
        return inputs_embeds

    @torch.inference_mode()
    def generate(self, *, input_prompt="", max_tokens=1000, seed=-1,
                 text_temperature=1.5, text_top_p=0.6, text_top_k=50,
                 audio_temperature=1.5, audio_top_p=0.6, audio_top_k=50,
                 audio_repetition_penalty=1.2, **kw):
        prompt = input_prompt or kw.get("prompt", "")
        if not prompt:
            raise ValueError("prompt required")

        if seed >= 0:
            torch.manual_seed(seed)

        dev = self.device
        cfg = self.config
        n_vq = self.n_vq
        batch_size = 1

        # Build chat input
        msgs = [{"role": "user", "content": f"Generate ambient sound: {prompt}"}]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        text_ids = inputs.input_ids.to(dev)  # (1, seq)

        # Build initial 3D input_ids: (1, seq, 1 + n_vq)
        seq_len = text_ids.shape[1]
        current_ids = torch.zeros(1, seq_len, 1 + n_vq, dtype=torch.long, device=dev)
        current_ids[:, :, 0] = text_ids
        # Fill audio channels with pad code for text tokens
        current_ids[:, :, 1:] = self.audio_pad_code

        attn_mask = inputs.attention_mask.to(dev)
        past = None
        generation_ids = current_ids.clone()

        # Delay pattern state
        is_stopping = torch.zeros(batch_size, dtype=torch.bool, device=dev)
        audio_lengths = torch.zeros(batch_size, dtype=torch.int64, device=dev)
        int64_max = torch.iinfo(torch.int64).max
        delayed_lengths = torch.full((batch_size,), int64_max, dtype=torch.int64, device=dev)

        # Token IDs from config
        audio_start_id = getattr(cfg, "audio_start_token_id", 151652)
        audio_end_id = getattr(cfg, "audio_end_token_id", 151653)
        audio_gen_slot_id = getattr(cfg, "audio_assistant_gen_slot_token_id", 151656)
        audio_delay_slot_id = getattr(cfg, "audio_assistant_delay_slot_token_id", 151662)
        pad_id = getattr(cfg, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(cfg.language_config, "pad_token_id", 0)
        im_end_id = getattr(cfg, "im_end_token_id",
                            getattr(cfg.language_config, "eos_token_id", 151645))

        # Masks for token filtering
        pre_exclude_ids = torch.tensor(
            [pad_id, audio_gen_slot_id, audio_delay_slot_id, audio_end_id], device=dev)
        vocab_size = (getattr(cfg, 'language_config', None) and
                      getattr(cfg.language_config, 'vocab_size', None)) or 155648
        pre_exclude_text = torch.ones(vocab_size, device=dev).bool()
        pre_exclude_text[[audio_gen_slot_id, audio_delay_slot_id]] = False

        is_continuation = (text_ids[0, -1] == audio_start_id) | (text_ids[0, -1] == audio_gen_slot_id)
        audio_start_indices = _find_last_equal(text_ids, audio_start_id)
        audio_start_mask = is_continuation.unsqueeze(0) & (audio_start_indices != -1)
        if audio_start_mask[0]:
            audio_lengths[0] = seq_len - audio_start_indices[0].item()
        is_audio = audio_start_mask.clone()

        for step in range(max_tokens):
            # Forward pass
            inputs_embeds = self._get_input_embeddings(current_ids)
            out = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask if past is None else None,
                past_key_values=past, use_cache=True, return_dict=True,
            )
            past = out.past_key_values
            hidden = out.last_hidden_state

            # Compute logits from all heads
            all_logits = [head(hidden[:, -1:, :]) for head in self.lm_heads]

            # Sample text token
            text_logits = all_logits[0][:, 0, :]  # (batch, vocab)
            if text_temperature > 0:
                text_logits = text_logits / text_temperature
            text_logits = text_logits.clone()
            next_text = torch.full((batch_size,), pad_id, dtype=torch.long, device=dev)

            # Delay pattern logic
            not_stopping = ~is_stopping
            next_text[not_stopping & (delayed_lengths < n_vq)] = audio_delay_slot_id
            is_audio_eos = not_stopping & (delayed_lengths == n_vq)
            next_text[is_audio_eos] = audio_end_id
            is_audio[is_audio_eos] = False

            sampling_mask = not_stopping & (delayed_lengths > n_vq)
            # Mask excluded tokens for non-audio positions
            text_logits[~is_audio] = text_logits[~is_audio].index_fill(
                -1, pre_exclude_ids, float('-inf'))
            text_logits[is_audio] = text_logits[is_audio].masked_fill(
                pre_exclude_text, float('-inf'))
            if step == 0:
                text_logits[..., audio_delay_slot_id] = float('-inf')
            if step <= n_vq:
                text_logits[..., im_end_id] = float('-inf')

            next_text[sampling_mask] = _sample_token(
                text_logits[sampling_mask],
                temperature=text_temperature, top_p=text_top_p, top_k=text_top_k)

            is_audio[next_text == audio_start_id] = True
            is_stopping[next_text == im_end_id] = True

            # Sample audio tokens
            next_audio = torch.full((batch_size, n_vq), self.audio_pad_code,
                                    dtype=torch.long, device=dev)
            pre_audio_mask = audio_lengths.unsqueeze(1) > torch.arange(n_vq, device=dev).unsqueeze(0)
            post_audio_mask = torch.arange(n_vq, device=dev).unsqueeze(0) > (delayed_lengths.unsqueeze(1) - 1)
            post_audio_mask[delayed_lengths == int64_max] = True
            sampling_audio_mask = pre_audio_mask & post_audio_mask

            if sampling_audio_mask.sum() > 0:
                audio_logits = torch.stack(
                    [l[:, 0, :] for l in all_logits[1:]], dim=1)[sampling_audio_mask]
                audio_logits[..., self.audio_pad_code] = float('-inf')
                next_audio[sampling_audio_mask] = _sample_token(
                    audio_logits, temperature=audio_temperature,
                    top_p=audio_top_p, top_k=audio_top_k)

            # Update delay pattern state
            audio_lengths[(next_text == audio_start_id) | (next_text == audio_gen_slot_id) | (next_text == audio_delay_slot_id)] += 1
            audio_lengths[next_text == audio_end_id] = 0
            delayed_lengths[(delayed_lengths == int64_max) & (next_text == audio_delay_slot_id)] = 0
            delayed_lengths[delayed_lengths != int64_max] += 1
            delayed_lengths[delayed_lengths > n_vq] = int64_max

            # Build next input: (1, 1, 1 + n_vq)
            current_ids = torch.cat(
                [next_text[:, None, None], next_audio[:, None, :]], dim=2)
            attn_mask = torch.cat(
                [attn_mask, (~is_stopping).unsqueeze(-1)], dim=-1)
            generation_ids = torch.cat([generation_ids, current_ids], dim=1)

            if is_stopping.sum() == batch_size:
                break

        # Extract audio codes from generated tokens
        gen = generation_ids[0]  # (seq, 1 + n_vq)
        text_tokens = gen[:, 0].cpu()

        # Find the audio section: between audio_start and audio_end
        start_positions = (text_tokens == audio_start_id).nonzero()
        end_positions = (text_tokens == audio_end_id).nonzero()

        if len(start_positions) == 0:
            logger.warning("No audio start token found in output")
            return self._silence_response()

        audio_start = start_positions[-1].item() + 1
        audio_end = end_positions[-1].item() if len(end_positions) > 0 else gen.shape[0]

        if audio_start >= audio_end:
            logger.warning("Empty audio section")
            return self._silence_response()

        # Extract audio codes: (audio_len, n_vq) → (n_vq, audio_len)
        audio_section = gen[audio_start:audio_end, 1:]  # (len, n_vq)
        # Clamp to valid codebook range [0, 1023], replace pad code 1024 with 0
        audio_codes = audio_section.clamp(max=self.audio_vocab - 1).t()  # (n_vq, len)

        # Decode with audio tokenizer
        if self.audio_tokenizer is not None:
            try:
                decoded = self.audio_tokenizer.decode(
                    audio_codes.unsqueeze(1),  # (n_vq, 1, len)
                    num_quantizers=self.n_vq,
                )
                if hasattr(decoded, 'audio') and decoded.audio is not None:
                    wav = decoded.audio
                    if isinstance(wav, (list, tuple)):
                        wav = wav[0]
                    wav_np = wav.flatten().cpu().float().numpy()
                    buf = io.BytesIO()
                    wavfile.write(buf, self.sr, wav_np)
                    return {"status": "success",
                            "data": base64.b64encode(buf.getvalue()).decode(),
                            "media_type": "audio/wav"}
            except Exception as e:
                logger.error("Audio decode failed: %s", e, exc_info=True)

        logger.warning("No audio tokenizer available, returning silence")
        return self._silence_response()

    def _silence_response(self):
        buf = io.BytesIO()
        wavfile.write(buf, self.sr, torch.zeros(self.sr).numpy())
        return {"status": "success",
                "data": base64.b64encode(buf.getvalue()).decode(),
                "media_type": "audio/wav"}
