"""Prompt Relay — temporal attention masking for segment-based prompting.

Ports the WhatDreamsCost LTX Director's core innovation: per-segment temporal
attention masks that allow different prompts to govern different portions of
the generated video timeline.

Architecture:
  - Global prompt conditions the entire video (characters, scene)
  - Local prompts (pipe-separated) each govern a temporal segment
  - Segment lengths define how many latent frames each prompt covers
  - A Gaussian penalty matrix biases cross-attention so each query frame
    attends most to its assigned segment's tokens

Adapted for Wan2GP's LTX2 pipeline: patches cross-attention blocks by
pre-computing penalty matrices for known latent dimensions and injecting
them into the mask parameter of each attention forward call.

Reference: WhatDreamsCost/ComfyUI prompt_relay.py (Apache-2.0 compatible)
"""
from __future__ import annotations

import math
import logging
import types
from typing import Callable

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token mapping
# ---------------------------------------------------------------------------

def map_token_indices(
    tokenizer,
    global_prompt: str,
    local_prompts: list[str],
) -> tuple[str, list[tuple[int, int]]]:
    """Tokenize global + space-prefixed locals; return (full_prompt, token_ranges).

    Uses incremental tokenization to avoid context-dependency issues with
    SentencePiece tokenizers.
    """
    prefixed_locals = [" " + lp for lp in local_prompts]
    full_prompt = global_prompt + "".join(prefixed_locals)

    has_eos = getattr(tokenizer, "add_eos", False)
    eos_adj = 1 if has_eos else 0

    prev_len = len(tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges = []

    built = global_prompt
    for plp in prefixed_locals:
        built += plp
        cur_len = len(tokenizer(built)["input_ids"]) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"Local prompt produced no tokens: '{plp.strip()}'")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len

    return full_prompt, token_ranges


# ---------------------------------------------------------------------------
# Segment distribution
# ---------------------------------------------------------------------------

def distribute_segment_lengths(
    num_segments: int,
    latent_frames: int,
    specified_lengths: list[int] | None = None,
) -> list[int]:
    """Validate or auto-distribute segment frame counts."""
    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"Number of segment_lengths ({len(specified_lengths)}) "
                f"must match number of local prompts ({num_segments})"
            )
        lengths = specified_lengths
    else:
        step = -(-latent_frames // num_segments)  # ceil division
        lengths = [step] * num_segments

    effective = []
    cursor = 0
    for length in lengths:
        end = min(cursor + length, latent_frames)
        effective.append(max(end - cursor, 0))
        cursor = end
    return effective


# ---------------------------------------------------------------------------
# Temporal penalty matrix
# ---------------------------------------------------------------------------

def build_temporal_cost(
    q_token_idx: list[dict],
    Lq: int,
    Lk: int,
    device: torch.device,
    dtype: torch.dtype,
    tokens_per_frame: int,
) -> torch.Tensor:
    """Gaussian penalty matrix [Lq, Lk] for video cross-attention."""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.long) // tokens_per_frame

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames.float()[:, None] - seg["midpoint"]).abs()
        strength = seg.get("strength", 1.0)
        cost = strength * (torch.relu(d - seg["window"]) ** 2) / (2 * seg["sigma"] ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


def build_temporal_cost_scaled(
    q_token_idx: list[dict],
    Lq: int,
    Lk: int,
    device: torch.device,
    dtype: torch.dtype,
    latent_frames: int,
) -> torch.Tensor:
    """Penalty matrix for queries that don't map to integer frames (audio tokens)."""
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.float32) * latent_frames / Lq

    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames[:, None] - seg["midpoint"]).abs()
        sigma_a = seg.get("sigma_audio", seg["sigma"])
        window_a = seg.get("window_audio", seg["window"])
        strength_a = seg.get("strength_audio", 1.0)
        cost = strength_a * (torch.relu(d - window_a) ** 2) / (2 * sigma_a ** 2)
        offset[:, local] = cost.to(offset.dtype)

    return offset


# ---------------------------------------------------------------------------
# Segment builder
# ---------------------------------------------------------------------------

def build_segments(
    token_ranges: list[tuple[int, int]],
    segment_lengths: list[int],
    epsilon: float = 1e-3,
) -> list[dict]:
    """Build per-segment metadata for the temporal penalty."""
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448

    q_token_idx = []
    frame_cursor = 0
    for (tok_start, tok_end), length in zip(token_ranges, segment_lengths):
        if length <= 0:
            frame_cursor += length
            continue

        midpoint = (2 * frame_cursor + length) // 2
        base_window = max(length // 2 - 2, 0)

        q_token_idx.append({
            "local_token_idx": torch.arange(tok_start, tok_end),
            "midpoint": midpoint,
            "window": max(base_window, 0.0),
            "sigma": sigma,
            "strength": 1.0,
            "window_audio": max(base_window, 0.0),
            "sigma_audio": sigma,
            "strength_audio": 1.0,
        })
        frame_cursor += length

    return q_token_idx


# ---------------------------------------------------------------------------
# Pre-computed mask builder
# ---------------------------------------------------------------------------

def precompute_relay_masks(
    q_token_idx: list[dict],
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    total_text_tokens: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Pre-compute temporal penalty matrices for video and audio cross-attention.

    Returns a dict with keys:
      "video": penalty matrix for video cross-attention [Lq_video, Lk]
      "audio": penalty matrix for audio cross-attention [Lq_audio, Lk]
      "text_tokens": total_text_tokens (for validation)
    """
    # Video tokens per frame: (latent_height / patch_h) * (latent_width / patch_w)
    # For LTX2: patch_size=4 in the VAE, downscale = 8 temporal, 32 spatial
    # After patching: latent_height // 1 * latent_width // 1 (already in latent space)
    # tokens_per_frame = (H_latent) * (W_latent) for the patchified latent
    # LTX2 uses patch_size 1x1 in latent space, so tokens_per_frame = latent_h * latent_w
    # Actually: the transformer uses a spatial downscale of 32 from pixels.
    # Latent dims = pixel // 32. Each latent position is one token per frame.
    tokens_per_frame = latent_height * latent_width
    Lq_video = latent_frames * tokens_per_frame
    Lk = total_text_tokens

    masks = {}

    if Lq_video > 0 and Lk > 0:
        cost = build_temporal_cost(q_token_idx, Lq_video, Lk, device, dtype, tokens_per_frame)
        masks["video"] = -cost  # Negative because we want to PENALIZE attending to wrong segments
        logger.info(
            "[PromptRelay] Pre-computed video mask: Lq=%d, Lk=%d, tpf=%d",
            Lq_video, Lk, tokens_per_frame,
        )

    return masks


# ---------------------------------------------------------------------------
# Transformer patching (Wan2GP LTX2 specific)
# ---------------------------------------------------------------------------

def apply_relay_patches(transformer, q_token_idx: list[dict], latent_frames: int,
                        latent_height: int, latent_width: int,
                        total_text_tokens: int) -> None:
    """Patch transformer blocks with pre-computed temporal penalty masks.

    For Wan2GP's LTX2 pipeline, we patch each block's `_apply_text_cross_attention`
    to add the temporal mask to the context_mask parameter before it reaches
    the attention module.
    """
    # Pre-compute masks
    masks = precompute_relay_masks(
        q_token_idx, latent_frames, latent_height, latent_width,
        total_text_tokens,
    )
    video_mask = masks.get("video")

    if video_mask is None:
        logger.warning("[PromptRelay] No video mask computed, skipping patches")
        return

    for idx, block in enumerate(transformer.transformer_blocks):
        _patch_block(block, video_mask, idx)


def _patch_block(block, video_mask: torch.Tensor, block_idx: int) -> None:
    """Patch a single transformer block to inject temporal mask into cross-attention.

    Monkey-patches `_apply_text_cross_attention` to add the temporal penalty
    to the context_mask parameter. This is called for both video (attn2) and
    audio (audio_attn2) cross-attention.
    """
    original_fn = block._apply_text_cross_attention
    mask = video_mask

    def patched_fn(self, x, context, attn, scale_shift_table,
                   prompt_scale_shift_table, timestep, prompt_timestep,
                   context_mask, nag=None, cross_attention_adaln=False):
        # Add temporal penalty to context_mask
        if context_mask is None:
            # Move mask to same device/dtype as x
            context_mask = mask.to(device=x.device, dtype=x.dtype)
        else:
            context_mask = context_mask + mask.to(device=x.device, dtype=x.dtype)

        return original_fn(
            x, context, attn, scale_shift_table,
            prompt_scale_shift_table, timestep, prompt_timestep,
            context_mask, nag=nag, cross_attention_adaln=cross_attention_adaln,
        )

    block._apply_text_cross_attention = types.MethodType(patched_fn, block)
    block._relay_original_fn = original_fn


def remove_relay_patches(transformer) -> None:
    """Remove prompt relay patches by restoring original methods."""
    for block in transformer.transformer_blocks:
        original_fn = getattr(block, "_relay_original_fn", None)
        if original_fn is not None:
            block._apply_text_cross_attention = original_fn
            del block._relay_original_fn
