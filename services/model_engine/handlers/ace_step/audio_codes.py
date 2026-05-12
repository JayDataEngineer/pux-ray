"""Audio code generation via Qwen3 LM for ACE-Step v1.5.

The LM generates <|audio_code_N|> tokens at 5Hz (one every 200ms).
These codes represent discrete audio features that the transformer
uses as hints during denoising.

Uses CFG (classifier-free guidance) with a negative prompt to improve
code quality, and a logits mask to force only audio-code tokens.

Reference: Wan2GP's models/TTS/ace_step15/qwen3_audio_codes.py
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import torch
from transformers import LogitsProcessor

logger = logging.getLogger(__name__)


class AudioCodeMaskProcessor(LogitsProcessor):
    """Forces generation of only audio-code tokens.

    Sets all non-audio-code logits to -inf, ensuring the LM
    can only sample from the audio code vocabulary.
    """

    def __init__(self, audio_code_mask: torch.Tensor):
        """
        Args:
            audio_code_mask: Boolean mask where True = allowed token.
                             Shape [vocab_size].
        """
        self.mask = audio_code_mask
        self._allowed_cache: dict[torch.device, torch.Tensor] = {}

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Set all non-audio-code positions to -inf
        scores.masked_fill_(~self.mask.to(scores.device), float("-inf"))
        return scores


def build_audio_code_vocab(lm_tokenizer) -> tuple:
    """Build audio code vocabulary from the LM tokenizer.

    Scans the tokenizer vocabulary for <|audio_code_N|> tokens and
    builds the lookup tables needed for generation and decoding.

    Returns:
        (audio_code_token_ids, audio_code_token_map, audio_code_mask)
        - token_ids: tensor of token IDs that are audio codes
        - token_map: dict mapping token_id → audio code integer
        - mask: float tensor [vocab_size] with -inf for non-audio-code tokens
    """
    vocab = lm_tokenizer.get_vocab()
    audio_code_pattern = re.compile(r"<\|audio_code_(\d+)\|>")

    token_ids = []
    token_map = {}

    for token, token_id in vocab.items():
        m = audio_code_pattern.match(token)
        if m:
            code = int(m.group(1))
            token_ids.append(token_id)
            token_map[token_id] = code

    token_ids_tensor = torch.tensor(sorted(token_ids), dtype=torch.long)

    # Build logits mask: -inf everywhere except audio code positions
    vocab_size = len(vocab)
    mask = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
    for tid in token_ids:
        mask[tid] = 0.0

    logger.info(
        "Built audio code vocab: %d codes, vocab_size=%d",
        len(token_ids), vocab_size,
    )

    return token_ids_tensor, token_map, mask


def generate_audio_codes(
    *,
    lm_model,
    lm_tokenizer,
    caption: str,
    lyrics: str,
    bpm: int,
    keyscale: str,
    timesignature: int,
    language: str,
    duration_seconds: float,
    min_tokens: int,
    max_tokens: int,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    cfg_scale: float = 2.5,
    audio_code_token_ids=None,
    audio_code_token_map=None,
    audio_code_mask=None,
    device: Optional[torch.device] = None,
) -> list[int]:
    """Generate audio codes from text via the LM.

    Uses classifier-free guidance (CFG) with a negative prompt.
    The LM generates <|audio_code_N|> tokens autoregressively.

    Args:
        lm_model: Qwen3ForCausalLM instance
        lm_tokenizer: tokenizer for the LM
        caption: music description (genre, mood, style)
        lyrics: song lyrics
        bpm, keyscale, timesignature, language: music metadata
        duration_seconds: target duration
        min_tokens: minimum audio code tokens to generate
        max_tokens: maximum tokens (truncation limit)
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        top_k: top-k sampling (None = disabled)
        cfg_scale: CFG guidance scale (1.0 = disabled)
        audio_code_token_ids: tensor of audio code token IDs
        audio_code_token_map: dict token_id → code integer
        audio_code_mask: logits mask tensor
        device: torch device

    Returns:
        List of integer audio codes
    """
    if device is None:
        device = next(lm_model.parameters()).device

    # Build prompts
    positive_prompt = _build_lm_prompt(
        caption, lyrics, bpm, keyscale, timesignature, language, duration_seconds,
    )
    negative_prompt = _build_negative_prompt()

    # Tokenize both together for matching lengths (CFG needs same seq dim)
    both = lm_tokenizer(
        [positive_prompt, negative_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(device)

    # Logits processor
    mask_processor = AudioCodeMaskProcessor(audio_code_mask.to(device))

    # Generate
    with torch.no_grad():
        outputs = lm_model.generate(
            input_ids=both["input_ids"],
            attention_mask=both["attention_mask"],
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            top_k=top_k if top_k else -1,
            do_sample=temperature > 0,
            logits_processor=[mask_processor],
            return_dict_in_generate=True,
            output_scores=False,
        )

    # Extract audio codes from the positive (first) sequence
    generated_ids = outputs.sequences[0]
    # Remove the input portion
    input_len = both["input_ids"].shape[1]
    new_ids = generated_ids[input_len:]

    # Map token IDs to audio codes
    audio_codes = []
    for token_id in new_ids.tolist():
        if token_id in audio_code_token_map:
            audio_codes.append(audio_code_token_map[token_id])

    if not audio_codes:
        logger.warning("LM generated 0 audio codes — returning silence codes")
        audio_codes = [0] * min_tokens

    # Post-process: pad or truncate
    if len(audio_codes) < min_tokens:
        audio_codes.extend([audio_codes[-1]] * (min_tokens - len(audio_codes)))
    audio_codes = audio_codes[:max_tokens]

    logger.info("Generated %d audio codes (target: %d-%d)", len(audio_codes), min_tokens, max_tokens)

    return audio_codes


def _build_lm_prompt(
    caption: str,
    lyrics: str,
    bpm: int,
    keyscale: str,
    timesignature: int,
    language: str,
    duration: float,
) -> str:
    """Build the positive LM prompt with metadata and lyrics."""
    return (
        f"<think/>\n"
        f"bpm: {bpm}\n"
        f"timesignature: {timesignature}\n"
        f"keyscale: {keyscale}\n"
        f"language: {language}\n"
        f"duration: {duration} seconds\n"
        f"<caption>{caption}</caption>\n"
        f"<lyric>{lyrics}</lyric>\n"
        f"<audio>"
    )


def _build_negative_prompt() -> str:
    """Build the negative prompt for CFG."""
    return "<think/>\nNO USER INPUT\n<caption></caption>\n<lyric></lyric>\n<audio>"
