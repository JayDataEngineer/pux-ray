"""MOSS-SoundEffect orchestrator — explicit generate loop with delay patterns.
 
Decomposed inference:
1. processor.build_user_message() — build conversation
2. processor() — tokenize + apply delay pattern
3. Manual generate loop: model.generate() (delay pattern coupling)
4. processor.decode() — extract audio codes
5. audio_tokenizer.decode() — codes → waveform
"""
from __future__ import annotations

import logging
import tempfile
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class MossOrchestrator:
    """MOSS-SoundEffect inference via direct forward() calls."""

    def __init__(self, modules):
        self.m = modules

    def generate(
        self,
        *,
        prompt: str = "",
        tokens: Optional[int] = None,
        max_tokens: int = 4096,
        seed: int = -1,
    ) -> dict:
        import base64
        import torchaudio

        if not prompt:
            raise ValueError("prompt required")

        batch_spec = {}
        if tokens is not None:
            batch_spec["tokens"] = tokens

        conversations = [
            [self.m.processor.build_user_message(ambient_sound=prompt, **batch_spec)]
        ]

        batch = self.m.processor(conversations, mode="generation")
        input_ids = batch["input_ids"].to(self.m.device)
        attention_mask = batch["attention_mask"].to(self.m.device)

        generated = self._generate_loop(input_ids, attention_mask, max_tokens)

        results = self.m.processor.decode(generated)
        if not results:
            raise RuntimeError("No audio generated")

        audio = results[0].audio_codes_list[0]

        sample_rate = self.m.processor.model_config.sampling_rate
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            torchaudio.save(f.name, audio.unsqueeze(0).cpu(), sample_rate)
            wav_data = open(f.name, "rb").read()

        return {
            "status": "success",
            "data": base64.b64encode(wav_data).decode(),
            "media_type": "audio/wav",
        }

    def _generate_loop(self, input_ids, attention_mask, max_new_tokens):
        with torch.no_grad():
            return self.m.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
