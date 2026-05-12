"""ACE-Step v1.5 inference pipeline.

Orchestrates the full text-to-music generation flow:
  1. Audio code generation via LM (Qwen3ForCausalLM 1.7B)
  2. Text encoding (Qwen3Model 0.6B embedding)
  3. Conditioning preparation (AceStepConditionEncoder)
  4. Denoising (AceStepDiTModel)
  5. Audio decoding (AutoencoderOobleck VAE)

Reference: Wan2GP's models/TTS/ace_step15/pipeline_ace_step15.py
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class AceStepV15Pipeline:
    """ACE-Step v1.5 text-to-music pipeline.

    Models are managed by mmgp — this class orchestrates inference.
    The pipeline object is stored in LoadResult.pipeline and called
    by the executor: result.pipeline(payload_dict).
    """

    SAMPLE_RATE = 48000
    HOP_LENGTH = 1920  # VAE downsampling product: 2*4*4*6*10

    def __init__(
        self,
        transformer,              # AceStepConditionGenerationModel
        audio_vae,                # AutoencoderOobleck
        text_encoder_2,           # Qwen3Model (0.6B embedding)
        lm_model=None,            # Qwen3ForCausalLM (1.7B, optional)
        pre_text_tokenizer=None,  # tokenizer for text_encoder_2
        lm_tokenizer=None,        # tokenizer for lm_model
        silence_latent=None,      # pre-computed silence latent tensor
        audio_code_token_ids=None,
        audio_code_token_map=None,
        audio_code_mask=None,
    ):
        self.transformer = transformer
        self.audio_vae = audio_vae
        self.text_encoder_2 = text_encoder_2
        self.lm_model = lm_model
        self.pre_text_tokenizer = pre_text_tokenizer
        self.lm_tokenizer = lm_tokenizer
        self.silence_latent = silence_latent
        self.audio_code_token_ids = audio_code_token_ids
        self.audio_code_token_map = audio_code_token_map
        self.audio_code_mask = audio_code_mask

        # Deep-copy quantizer + detokenizer to CPU for audio code → latent
        # conversion without touching GPU-loaded transformer
        if (
            hasattr(transformer, "tokenizer")
            and hasattr(transformer.tokenizer, "quantizer")
        ):
            self._lm_hint_quantizer = (
                copy.deepcopy(transformer.tokenizer.quantizer).cpu().float()
            )
            self._lm_hint_detokenizer = (
                copy.deepcopy(transformer.detokenizer).cpu().float()
            )
        else:
            self._lm_hint_quantizer = None
            self._lm_hint_detokenizer = None

    def __call__(self, payload: dict) -> dict:
        """Run inference from a payload dict."""
        return self.generate(
            input_prompt=payload.get("prompt", payload.get("input_prompt", "")),
            alt_prompt=payload.get("caption", payload.get("alt_prompt", "")),
            duration_seconds=float(
                payload.get("duration", payload.get("duration_seconds", 30))
            ),
            num_inference_steps=int(
                payload.get("steps", payload.get("num_inference_steps", 8))
            ),
            seed=payload.get("seed"),
            temperature=float(payload.get("temperature", 0.85)),
            top_p=float(payload.get("top_p", 0.9)),
            top_k=payload.get("top_k"),
            alt_guidance_scale=float(payload.get("alt_guidance_scale", 2.5)),
            shift=float(payload.get("shift", 1.0)),
            custom_settings=payload.get("custom_settings"),
        )

    def generate(
        self,
        *,
        input_prompt: str = "",
        alt_prompt: str = "",
        duration_seconds: float = 30,
        num_inference_steps: int = 8,
        seed: Optional[int] = None,
        temperature: float = 0.85,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        alt_guidance_scale: float = 2.5,
        shift: float = 1.0,
        custom_settings: Optional[dict] = None,
    ) -> dict:
        """Generate music from text prompts.

        Returns dict with keys: audio (tensor), sample_rate, duration_seconds.
        """
        device = next(self.transformer.parameters()).device
        dtype = next(self.transformer.parameters()).dtype

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        cs = custom_settings or {}
        bpm = int(cs.get("bpm", 120))
        keyscale = cs.get("keyscale", "C")
        timesignature = int(cs.get("timesignature", 4))
        language = cs.get("language", "unknown")

        caption = alt_prompt or ""
        lyrics = input_prompt or ""

        logger.info(
            "ACE-Step v1.5: caption=%r lyrics=%dch dur=%.0fs steps=%d shift=%.1f",
            caption[:60],
            len(lyrics),
            duration_seconds,
            num_inference_steps,
            shift,
        )

        # ── Phase 1: Audio codes via LM ──────────────────────────────
        audio_codes = None
        if self.lm_model is not None and self.audio_code_mask is not None:
            audio_codes = self._generate_audio_codes(
                caption=caption,
                lyrics=lyrics,
                duration_seconds=duration_seconds,
                bpm=bpm,
                keyscale=keyscale,
                timesignature=timesignature,
                language=language,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                alt_guidance_scale=alt_guidance_scale,
                device=device,
            )

        # ── Phase 2: Text encoding ───────────────────────────────────
        text_hidden_states, text_attention_mask = self._encode_caption(
            caption, bpm, keyscale, timesignature, duration_seconds, device,
        )
        lyric_hidden_states, lyric_attention_mask = self._encode_lyrics(
            lyrics, language, device,
        )

        # ── Phase 3: Prepare latents ─────────────────────────────────
        latent_length = round(
            duration_seconds * self.SAMPLE_RATE / self.HOP_LENGTH
        )

        silence_latent = self._get_silence_latent(latent_length, device, dtype)

        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        noise = torch.randn(
            1,
            silence_latent.shape[1],
            latent_length,
            device=device,
            dtype=dtype,
            generator=gen,
        )

        audio_codes_tensor = None
        if audio_codes is not None:
            audio_codes_tensor = (
                torch.tensor(audio_codes, dtype=torch.long, device=device)
                .unsqueeze(0)
                .unsqueeze(-1)
            )

        # ── Phase 4: Conditioning ────────────────────────────────────
        latent_attention_mask = torch.ones(1, latent_length, device=device)
        refer_audio = torch.zeros(
            1, 64, latent_length, device=device, dtype=dtype,
        )
        refer_audio_order_mask = torch.ones(1, device=device, dtype=torch.long)
        src_latents = silence_latent.clone()
        chunk_masks = torch.zeros_like(src_latents)
        is_covers = torch.tensor(
            [1 if audio_codes_tensor is not None else 0],
            device=device,
            dtype=torch.long,
        )

        with torch.no_grad():
            encoder_hidden_states, encoder_attention_mask, context_latents = (
                self.transformer.prepare_condition(
                    text_hidden_states=text_hidden_states,
                    text_attention_mask=text_attention_mask,
                    lyric_hidden_states=lyric_hidden_states,
                    lyric_attention_mask=lyric_attention_mask,
                    refer_audio_acoustic_hidden_states_packed=refer_audio,
                    refer_audio_order_mask=refer_audio_order_mask,
                    hidden_states=src_latents,
                    attention_mask=latent_attention_mask,
                    silence_latent=silence_latent,
                    src_latents=src_latents,
                    chunk_masks=chunk_masks,
                    is_covers=is_covers,
                    audio_codes=audio_codes_tensor,
                )
            )

        # ── Phase 5: Denoise ─────────────────────────────────────────
        t_schedule = self._build_t_schedule(num_inference_steps, shift)
        xt = noise

        with torch.no_grad():
            for i, t in enumerate(t_schedule):
                t_next = t_schedule[i + 1] if i + 1 < len(t_schedule) else 0.0
                t_tensor = torch.tensor([t], device=device)

                vt = self.transformer.decoder(
                    hidden_states=xt,
                    timestep=t_tensor,
                    timestep_r=t_tensor,
                    attention_mask=latent_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                )[0]

                # ODE update
                if i == len(t_schedule) - 1:
                    xt = xt - vt * t_tensor.view(-1, 1, 1)
                else:
                    dt = t - t_next
                    xt = xt - vt * dt

        # ── Phase 6: Decode audio ─────────────────────────────────────
        with torch.no_grad():
            audio_out = self.audio_vae.decode(xt)
        if hasattr(audio_out, "sample"):
            audio_out = audio_out.sample

        target_samples = int(duration_seconds * self.SAMPLE_RATE)
        audio_out = audio_out[..., :target_samples]

        logger.info("ACE-Step v1.5 done: shape=%s", tuple(audio_out.shape))

        return {
            "audio": audio_out,
            "sample_rate": self.SAMPLE_RATE,
            "duration_seconds": duration_seconds,
        }

    # ── Internal helpers ──────────────────────────────────────────────

    def _get_silence_latent(
        self, latent_length: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get or create silence latent for the target length."""
        if self.silence_latent is not None:
            sl = self.silence_latent.to(device=device, dtype=dtype)
            if sl.shape[-1] >= latent_length:
                return sl[:, :, :latent_length]
            # Pad by repeating
            reps = (latent_length // sl.shape[-1]) + 1
            sl = sl.repeat(1, 1, reps)[:, :, :latent_length]
            return sl
        return torch.zeros(1, 64, latent_length, device=device, dtype=dtype)

    def _encode_caption(
        self,
        caption: str,
        bpm: int,
        keyscale: str,
        timesignature: int,
        duration: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode caption + metadata via text_encoder_2 (full forward)."""
        prompt = (
            f"# Instruction\nMusic generation\n\n"
            f"# Caption\n{caption}\n\n"
            f"# Metas\n"
            f"- bpm: {bpm}\n"
            f"- timesignature: {timesignature}\n"
            f"- keyscale: {keyscale}\n"
            f"- duration: {duration} seconds"
        )

        inputs = self.pre_text_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            max_length=256,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = self.text_encoder_2(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=False,
                use_cache=False,
            )

        return outputs.last_hidden_state, inputs["attention_mask"]

    def _encode_lyrics(
        self, lyrics: str, language: str, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode lyrics via text_encoder_2 (embed_tokens only)."""
        prompt = f"# Languages\n{language}\n\n# Lyric\n{lyrics}"

        inputs = self.pre_text_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            max_length=2048,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            hidden_states = self.text_encoder_2.embed_tokens(inputs["input_ids"])

        return hidden_states, inputs["attention_mask"]

    def _generate_audio_codes(
        self,
        caption: str,
        lyrics: str,
        duration_seconds: float,
        bpm: int,
        keyscale: str,
        timesignature: int,
        language: str,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        alt_guidance_scale: float,
        device: torch.device,
    ) -> list[int]:
        """Generate audio codes via the LM."""
        from .audio_codes import generate_audio_codes

        min_tokens = max(1, int(duration_seconds)) * 5
        max_tokens = min_tokens + 50

        return generate_audio_codes(
            lm_model=self.lm_model,
            lm_tokenizer=self.lm_tokenizer,
            caption=caption,
            lyrics=lyrics,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
            language=language,
            duration_seconds=duration_seconds,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cfg_scale=alt_guidance_scale,
            audio_code_token_ids=self.audio_code_token_ids,
            audio_code_token_map=self.audio_code_token_map,
            audio_code_mask=self.audio_code_mask,
            device=device,
        )

    @staticmethod
    def _build_t_schedule(num_steps: int, shift: float = 1.0) -> list[float]:
        """Build timestep schedule for denoising.

        Predefined 8-step schedules from ACE-Step:
          shift=1.0 — standard, good balance
          shift=2.0 — more time at high noise (slower structure formation)
          shift=3.0 — even more conservative (better for complex pieces)
        """
        predefined = {
            1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
            2.0: [1.0, 0.933, 0.857, 0.769, 0.667, 0.545, 0.4, 0.222],
            3.0: [1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3],
        }

        if shift in predefined:
            schedule = predefined[shift]
        else:
            # Linear schedule for non-standard shifts
            schedule = [1.0 - i / num_steps for i in range(num_steps)]
            schedule[-1] = max(schedule[-1], 0.001)

        # Truncate or subsample to requested step count
        if len(schedule) > num_steps:
            indices = [
                int(i * (len(schedule) - 1) / max(num_steps - 1, 1))
                for i in range(num_steps)
            ]
            schedule = [schedule[i] for i in indices]

        return schedule
