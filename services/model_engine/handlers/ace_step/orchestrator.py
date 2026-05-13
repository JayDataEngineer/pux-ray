"""ACE-Step v1.5 orchestrator — raw nn.Module composition.

Calls transformer.forward(), codec.decode(), etc. directly.
No pipeline wrapper — every tensor op is explicit.

Supports all Wan2GP ACE-Step features:
  - CoT metadata inference (5 model modes)
  - Audio code generation with CFG
  - Reference audio encoding (cover/timbre)
  - Cover-strength blending
  - ODE + SDE denoising
  - VAE temporal tiling for long audio
  - User-supplied audio codes
  - All shift schedules
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

import torch

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────────

SAMPLE_RATE = 48000
HOP_LENGTH = 1920

# Predefined timestep schedules (8-step)
SCHEDULES: dict[float, list[float]] = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [1.0, 0.933, 0.857, 0.769, 0.667, 0.545, 0.4, 0.222],
    3.0: [1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3],
}

DEFAULT_BPM = 120
DEFAULT_KEYSCALE = "C major"
DEFAULT_TIMESIGNATURE = 4
DEFAULT_LANGUAGE = "unknown"
DEFAULT_DURATION = 30.0


# ═══════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class AceStepOrchestrator:
    """Orchestrates ACE-Step v1.5 generation via raw nn.Module.forward() calls."""

    def __init__(self, modules):
        self.m = modules

    # ── Pipeline API ────────────────────────────────────────────────────

    def generate(
        self,
        *,
        input_prompt: str = "",
        alt_prompt: str = "",
        duration_seconds: float = DEFAULT_DURATION,
        num_inference_steps: int = 8,
        seed: Optional[int] = None,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        alt_guidance_scale: float = 2.5,
        shift: float = 1.0,
        infer_method: str = "ode",
        model_mode: int = 0,
        audio_cover_strength: float = 1.0,
        reference_audio: Optional[str] = None,
        audio_codes: Optional[list[int]] = None,
        lm_negative_prompt: str = "",
        custom_settings: Optional[dict] = None,
    ) -> dict:
        """Generate music. Returns dict with keys: audio, sample_rate, duration_seconds."""
        device = self.m.device
        dtype = self.m.dtype

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        cs = custom_settings or {}
        caption = alt_prompt or ""
        lyrics = input_prompt or ""

        bpm = int(cs.get("bpm", DEFAULT_BPM))
        keyscale = cs.get("keyscale", DEFAULT_KEYSCALE)
        timesignature = int(cs.get("timesignature", DEFAULT_TIMESIGNATURE))
        language = cs.get("language", DEFAULT_LANGUAGE)

        # ── Phase 0: CoT metadata inference (if enabled) ──────────────
        if model_mode > 0 and self.m.lm_engine is not None:
            try:
                meta = self.m.lm_engine.infer_metadata(caption, lyrics, seed)
                if meta.get("bpm") is not None:
                    bpm = meta["bpm"]
                if meta.get("keyscale") is not None:
                    keyscale = meta["keyscale"]
                if meta.get("timesignature") is not None:
                    timesignature = meta["timesignature"]
                if meta.get("language") is not None:
                    language = meta["language"]
                if meta.get("duration") is not None and model_mode in (3, 4):
                    duration_seconds = float(meta["duration"])
                if meta.get("caption") is not None and model_mode in (2, 3):
                    caption = meta["caption"]
            except Exception as e:
                logger.warning("CoT inference failed: %s", e)

        logger.info(
            "ACE-Step: caption=%r lyrics=%dch dur=%.0fs steps=%d shift=%.1f "
            "method=%s mode=%d",
            caption[:60], len(lyrics), duration_seconds,
            num_inference_steps, shift, infer_method, model_mode,
        )

        # ── Phase 1: Audio codes ─────────────────────────────────────
        if audio_codes is not None:
            audio_codes_list = audio_codes
        elif self.m.lm_engine is not None:
            min_tokens = max(1, int(duration_seconds)) * 5
            max_tokens = min_tokens + 50
            audio_codes_list = self.m.lm_engine.generate_codes(
                caption=caption, lyrics=lyrics,
                bpm=bpm, keyscale=keyscale, timesignature=timesignature,
                language=language, duration_seconds=duration_seconds,
                min_tokens=min_tokens, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                cfg_scale=alt_guidance_scale, seed=seed,
            )
        else:
            audio_codes_list = None

        # ── Phase 2: Text encoding ───────────────────────────────────
        text_hidden, text_mask = self._encode_caption(
            caption, bpm, keyscale, timesignature, duration_seconds, device,
        )
        lyric_hidden, lyric_mask = self._encode_lyrics(lyrics, language, device)

        # ── Phase 3: Latents ─────────────────────────────────────────
        latent_length = round(duration_seconds * SAMPLE_RATE / HOP_LENGTH)
        silence_latent = self._get_silence_latent(latent_length, device, dtype)

        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        noise = torch.randn(silence_latent.shape, device=device, dtype=dtype, generator=gen)

        # Pre-compute LM hints from audio codes
        precomputed_lm_hints = torch.zeros(1, latent_length, 64, device=device, dtype=dtype)
        if audio_codes_list is not None:
            hints = self._codes_to_hints(audio_codes_list, latent_length, dtype)
            if hints is not None:
                precomputed_lm_hints = hints.to(device=device, dtype=dtype)

        # ── Phase 4: Reference audio ─────────────────────────────────
        ref_latent = None
        if reference_audio is not None:
            ref_latent = self._encode_reference_audio(reference_audio, latent_length, device)

        has_cover = (
            audio_codes_list is not None
            and self.m.codebook is not None
        )

        # ── Phase 5: Conditioning ────────────────────────────────────
        refer_audio = ref_latent if ref_latent is not None else torch.zeros(
            1, 1, 64 if ref_latent is None else ref_latent.shape[1], 64,
            device=device, dtype=dtype,
        )
        if ref_latent is None:
            refer_audio = torch.zeros(1, latent_length, 64, device=device, dtype=dtype)
        refer_audio_order_mask = torch.tensor([0], device=device, dtype=torch.long)

        latent_mask = torch.ones(1, latent_length, device=device)
        src_latents = silence_latent.clone()
        chunk_masks = torch.ones_like(src_latents)
        is_covers = torch.tensor([1 if has_cover else 0], device=device, dtype=torch.long)

        with torch.no_grad():
            cond = self.m.transformer.prepare_condition(
                text_hidden_states=text_hidden,
                text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hidden,
                lyric_attention_mask=lyric_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio,
                refer_audio_order_mask=refer_audio_order_mask,
                hidden_states=src_latents,
                attention_mask=latent_mask,
                silence_latent=silence_latent,
                src_latents=src_latents,
                chunk_masks=chunk_masks,
                is_covers=is_covers,
                precomputed_lm_hints_25Hz=precomputed_lm_hints,
            )

        # Cover-strength: pre-compute non-cover condition if blending
        cond_nc = None
        if has_cover and audio_cover_strength < 1.0:
            with torch.no_grad():
                cond_nc = self.m.transformer.prepare_condition(
                    text_hidden_states=text_hidden,
                    text_attention_mask=text_mask,
                    lyric_hidden_states=lyric_hidden,
                    lyric_attention_mask=lyric_mask,
                    refer_audio_acoustic_hidden_states_packed=refer_audio,
                    refer_audio_order_mask=refer_audio_order_mask,
                    hidden_states=src_latents,
                    attention_mask=latent_mask,
                    silence_latent=silence_latent,
                    src_latents=silence_latent.clone(),
                    chunk_masks=chunk_masks,
                    is_covers=torch.tensor([0], device=device, dtype=torch.long),
                    precomputed_lm_hints_25Hz=torch.zeros(
                        1, latent_length, 64, device=device, dtype=dtype,
                    ),
                )

        # ── Phase 6: Denoise ─────────────────────────────────────────
        t_schedule = self._build_schedule(num_inference_steps, shift)
        xt = noise
        cover_steps = int(num_inference_steps * audio_cover_strength)

        with torch.no_grad():
            for i, t in enumerate(t_schedule):
                t_tensor = torch.tensor([t], device=device)

                # Switch to non-cover after cover steps
                eh, em, cl = cond
                if cond_nc is not None and i >= cover_steps:
                    eh, em, cl = cond_nc

                vt = self.m.transformer.decoder(
                    hidden_states=xt,
                    timestep=t_tensor,
                    timestep_r=t_tensor,
                    attention_mask=latent_mask,
                    encoder_hidden_states=eh,
                    encoder_attention_mask=em,
                    context_latents=cl,
                )[0].to(device)

                if i == len(t_schedule) - 1:
                    xt = xt - vt * t_tensor.view(-1, 1, 1)
                elif infer_method == "sde":
                    pred_clean = xt - vt * t_tensor.view(-1, 1, 1)
                    t_next = t_schedule[i + 1]
                    xt = t_next * torch.randn_like(pred_clean) + (1 - t_next) * pred_clean
                else:
                    t_next = t_schedule[i + 1]
                    xt = xt - vt * (t - t_next)

        # ── Phase 7: VAE decode ──────────────────────────────────────
        tile_seconds = self._get_tile_seconds(duration_seconds)
        if tile_seconds is not None and duration_seconds > tile_seconds:
            audio_out = self._decode_tiled(xt, tile_seconds)
        else:
            audio_out = self.m.codec.decode(xt.permute(0, 2, 1))
            if hasattr(audio_out, "sample"):
                audio_out = audio_out.sample

        target_samples = int(duration_seconds * SAMPLE_RATE)
        audio_out = audio_out[..., :target_samples]

        logger.info("ACE-Step done: shape=%s method=%s", tuple(audio_out.shape), infer_method)

        return {"audio": audio_out, "sample_rate": SAMPLE_RATE, "duration_seconds": duration_seconds}

    # ── Audio codes to latent hints ─────────────────────────────────────────

    def _codes_to_hints(
        self, codes: list[int], target_length: int, dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Convert audio code ints → 25Hz latent hints.

        Manual codebook lookup + projection + detokenize on CPU,
        bypassing the quantizer object to avoid mmgp/einx issues.
        """
        if self.m.codebook is None or self.m.detokenizer is None:
            return None
        try:
            c = torch.tensor(codes, dtype=torch.long, device="cpu").unsqueeze(0)
            c = c.clamp(0, self.m.codebook.shape[1] - 1)
            with torch.no_grad():
                looked_up = self.m.codebook[0][c]
                projected = torch.nn.functional.linear(
                    looked_up, self.m.proj_weight, self.m.proj_bias,
                )
                detok_dtype = next(self.m.detokenizer.parameters()).dtype
                hints = self.m.detokenizer(projected.to(detok_dtype))
            return hints[:, :target_length, :]
        except Exception as e:
            logger.warning("codes_to_hints failed: %s", e)
            return None

    # ── Text encoding ────────────────────────────────────────────────────

    def _encode_caption(self, caption, bpm, keyscale, timesignature, duration, device):
        prompt = (
            f"# Instruction\nMusic generation\n\n"
            f"# Caption\n{caption}\n\n"
            f"# Metas\n"
            f"- bpm: {bpm}\n"
            f"- timesignature: {timesignature}\n"
            f"- keyscale: {keyscale}\n"
            f"- duration: {duration} seconds"
        )
        inputs = self.m.pre_text_tokenizer(
            prompt, return_tensors="pt", padding=True, max_length=256, truncation=True,
        ).to(device)
        with torch.no_grad():
            out = self.m.text_encoder_2(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                output_hidden_states=False, use_cache=False,
            )
        return out.last_hidden_state, inputs["attention_mask"]

    def _encode_lyrics(self, lyrics, language, device):
        prompt = f"# Languages\n{language}\n\n# Lyric\n{lyrics}"
        inputs = self.m.pre_text_tokenizer(
            prompt, return_tensors="pt", padding=True, max_length=2048, truncation=True,
        ).to(device)
        with torch.no_grad():
            hs = self.m.text_encoder_2.embed_tokens(inputs["input_ids"])
        return hs, inputs["attention_mask"]

    # ── Reference audio ──────────────────────────────────────────────────

    def _encode_reference_audio(
        self, audio_path: str, target_length: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Encode reference audio → [1, T, 64] latent for timbre conditioning.

        Takes 3×10s segments (front/middle/back), VAE encodes, returns latent.
        """
        try:
            import soundfile as sf
            import random

            audio, sr = sf.read(audio_path)
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            elif audio.ndim == 2 and audio.shape[0] > 2:
                audio = audio.T

            # Normalize to stereo 48kHz
            if sr != SAMPLE_RATE:
                import torchaudio.functional as F
                audio = F.resample(
                    torch.from_numpy(audio).float(), sr, SAMPLE_RATE,
                ).numpy()
                sr = SAMPLE_RATE

            if audio.shape[0] == 1:
                audio = audio.repeat(2, axis=0)
            elif audio.shape[0] > 2:
                audio = audio[:2]

            total = audio.shape[1]
            seg_frames = int(10 * sr)
            target_frames = int(30 * sr)

            if total < target_frames:
                repeats = (target_frames // total) + 1
                audio = audio.repeat(repeats, axis=1)
                total = audio.shape[1]

            seg_size = total // 3

            def rand_start(base, avail):
                return base + random.randint(0, max(0, avail - seg_frames)) if avail > seg_frames else base

            front = audio[:, rand_start(0, seg_size):rand_start(0, seg_size) + seg_frames]
            middle = audio[:, rand_start(seg_size, seg_size):rand_start(seg_size, seg_size) + seg_frames]
            back = audio[:, rand_start(2 * seg_size, total - 2 * seg_size):rand_start(2 * seg_size, total - 2 * seg_size) + seg_frames]

            concat = torch.from_numpy(
                np.concatenate([front, middle, back], axis=1)
            ).float().unsqueeze(0).to(device)

            with torch.no_grad():
                encoded = self.m.codec.encode(concat)
                latent = encoded.latent_dist.mode()

            # latent shape: [1, 64, T] → [1, T, 64]
            return latent.permute(0, 2, 1)[:, :target_length, :]
        except Exception as e:
            logger.warning("Reference audio encoding failed: %s", e)
            return None

    # ── Silence latent ───────────────────────────────────────────────────

    def _get_silence_latent(self, length, device, dtype):
        if self.m.silence_latent is not None:
            sl = self.m.silence_latent.to(device=device, dtype=dtype)
            if sl.dim() == 2:
                sl = sl.unsqueeze(0)
            if sl.dim() == 3 and sl.shape[1] == 64 and sl.shape[2] != 64:
                sl = sl.permute(0, 2, 1)
            t = sl.shape[1]
            if t < length:
                sl = torch.nn.functional.pad(sl, (0, 0, 0, length - t))
            elif t > length:
                sl = sl[:, :length, :]
            return sl
        return torch.zeros(1, length, 64, device=device, dtype=dtype)

    # ── Schedule ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_schedule(num_steps: int, shift: float = 1.0) -> list[float]:
        if shift in SCHEDULES:
            sched = SCHEDULES[shift]
        else:
            sched = [1.0 - i / num_steps for i in range(num_steps)]
            sched[-1] = max(sched[-1], 0.001)
        if len(sched) > num_steps:
            idxs = [int(i * (len(sched) - 1) / max(num_steps - 1, 1)) for i in range(num_steps)]
            sched = [sched[i] for i in idxs]
        return sched[:num_steps]

    # ── VAE tiling ───────────────────────────────────────────────────────

    @staticmethod
    def _get_tile_seconds(duration: float) -> Optional[float]:
        """Determine if tiling is needed based on duration + GPU memory."""
        if not torch.cuda.is_available():
            return None
        try:
            gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            return None
        if gb >= 24:
            tile = 80.0
        elif gb >= 12:
            tile = 40.0
        else:
            tile = 20.0
        return tile if duration > tile else None

    def _decode_tiled(self, latents: torch.Tensor, tile_seconds: float) -> torch.Tensor:
        """Tiled VAE decode with cross-fade overlap."""
        frames_per_sec = SAMPLE_RATE / HOP_LENGTH
        tile_frames = int(round(tile_seconds * frames_per_sec))
        overlap = int(round(tile_frames * 0.25))
        if overlap >= tile_frames:
            overlap = max(0, tile_frames // 4)

        hop = HOP_LENGTH
        total_frames = latents.shape[1]
        total_samples = total_frames * hop
        step = max(1, tile_frames - overlap)

        x = latents.permute(0, 2, 1)
        output = None

        for start in range(0, total_frames, step):
            end = min(start + tile_frames, total_frames)
            chunk = x[:, :, start:end]
            with torch.no_grad():
                decoded = self.m.codec.decode(chunk)
            chunk_audio = decoded.sample if hasattr(decoded, "sample") else decoded

            expected = (end - start) * hop
            if chunk_audio.shape[-1] > expected:
                chunk_audio = chunk_audio[..., :expected]
            elif chunk_audio.shape[-1] < expected:
                chunk_audio = torch.nn.functional.pad(chunk_audio, (0, expected - chunk_audio.shape[-1]))

            if output is None:
                output = chunk_audio.new_zeros((1, chunk_audio.shape[1], total_samples))

            start_s = start * hop
            end_s = start_s + expected

            if start == 0 or overlap == 0:
                output[..., start_s:end_s] = chunk_audio
            else:
                ov = min(overlap * hop, start_s, expected)
                fade = torch.linspace(0.0, 1.0, ov, device=chunk_audio.device, dtype=chunk_audio.dtype).view(1, 1, -1)
                output[..., start_s:start_s + ov] = (
                    output[..., start_s:start_s + ov] * (1.0 - fade) + chunk_audio[..., :ov] * fade
                )
                output[..., start_s + ov:end_s] = chunk_audio[..., ov:]

        return output



