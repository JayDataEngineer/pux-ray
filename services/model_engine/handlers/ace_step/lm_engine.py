"""CoT metadata inference + audio code generation LM engine."""
from __future__ import annotations

import logging
import re
from typing import Optional

import torch
from transformers import LogitsProcessor

logger = logging.getLogger(__name__)


# ── Audio-code-only processor ───────────────────────────────────────────────

class AudioCodeOnlyProcessor(LogitsProcessor):
    """Only allow audio-code + EOS tokens."""

    def __init__(self, audio_code_mask, max_codes, eos_id=None):
        self.audio_code_mask = audio_code_mask.to(torch.float32)
        self.max_codes = max_codes
        self.eos_id = eos_id
        self.count = 0

    def update_state(self, token_id):
        self.count += 1

    def __call__(self, input_ids, scores):
        scores = scores + self.audio_code_mask.to(scores.device)
        if self.count >= self.max_codes and self.eos_id is not None:
            scores.fill_(float("-inf"))
            if self.eos_id < scores.shape[-1]:
                scores[..., self.eos_id] = 0.0
        return scores


# ── LM Engine ───────────────────────────────────────────────────────────────

class LmEngine:
    """LM with CFG + CoT metadata inference + audio code generation."""

    def __init__(self, model, tokenizer, audio_code_mask, audio_code_token_map):
        self.model = model
        self.tokenizer = tokenizer
        self.audio_code_mask = audio_code_mask.to(model.device)
        self.audio_code_token_map = audio_code_token_map
        self.device = model.device

    def generate_text(
        self,
        prompt: str,
        prompt_negative: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        cfg_scale: float = 1.0,
        seed: Optional[int] = None,
        logits_processor: Optional[LogitsProcessor] = None,
        logits_processor_update_state=None,
    ) -> dict:
        """Generate text with CFG. Returns dict with 'token_ids' and 'text'."""
        if seed is not None:
            torch.manual_seed(seed)

        both = self.tokenizer(
            [prompt, prompt_negative], return_tensors="pt", padding=True, padding_side="left",
        ).to(self.device)

        input_ids = both["input_ids"]
        attention_mask = both["attention_mask"]
        past_key_values = None
        generated_ids = []
        cond_ids = input_ids[0:1]

        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)

        with torch.no_grad():
            for step in range(max_tokens):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

                if cfg_scale != 1.0:
                    cond = logits[0:1]
                    uncond = logits[1:2]
                    logits = uncond + cfg_scale * (cond - uncond)
                else:
                    logits = logits[0:1]

                if logits_processor is not None:
                    logits = logits_processor(cond_ids, logits)

                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
                        cumsum = sorted_probs.cumsum(dim=-1)
                        mask = cumsum - sorted_probs > (1 - top_p)
                        sorted_probs[mask] = 0.0
                        probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                        tid = torch.multinomial(probs.squeeze(0), num_samples=1, generator=gen).item()
                        tid = sorted_idx[0, tid].item()
                    else:
                        tid = torch.multinomial(probs.squeeze(0), num_samples=1, generator=gen).item()
                else:
                    tid = logits.argmax(dim=-1).item()

                if tid == self.tokenizer.eos_token_id:
                    break

                generated_ids.append(tid)
                cond_ids = torch.cat([cond_ids, torch.tensor([[tid]], device=self.device)], dim=1)

                if logits_processor_update_state is not None:
                    logits_processor_update_state(tid)

                if tid in self.audio_code_token_map:
                    break

                input_ids = torch.tensor([[tid], [tid]], device=self.device)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((2, 1), device=self.device, dtype=torch.long),
                ], dim=1)

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {"token_ids": generated_ids, "text": text}

    def infer_metadata(self, caption: str, lyrics: str, seed=None) -> dict:
        """Run CoT metadata inference. Model outputs <think> block with metadata."""
        meta_prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Extract music metadata (bpm, keyscale, timesignature, language, duration) from this description: {caption}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        neg_prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "Extract music metadata (bpm, keyscale, timesignature, language, duration) from this description: NO USER INPUT\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        result = self.generate_text(
            prompt=meta_prompt, prompt_negative=neg_prompt,
            max_tokens=256, temperature=0.7, cfg_scale=1.0,
            seed=seed,
        )

        meta = _parse_think_meta(result["text"])
        logger.info("CoT metadata: %s", meta)
        return meta

    def generate_codes(
        self,
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
        seed: Optional[int] = None,
    ) -> list[int]:
        """Generate audio codes with CFG and audio-code-only constraint."""
        from .audio_codes import _build_lm_prompt, _build_negative_prompt

        prompt = _build_lm_prompt(caption, lyrics, bpm, keyscale, timesignature, language, duration_seconds)
        prompt_neg = _build_negative_prompt()

        code_processor = AudioCodeOnlyProcessor(self.audio_code_mask, max_tokens, eos_id=self.tokenizer.eos_token_id)
        result = self.generate_text(
            prompt=prompt, prompt_negative=prompt_neg,
            max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, cfg_scale=cfg_scale,
            seed=seed,
            logits_processor=code_processor,
            logits_processor_update_state=code_processor.update_state,
        )

        codes = []
        for tid in result["token_ids"]:
            if tid in self.audio_code_token_map:
                codes.append(self.audio_code_token_map[tid])
        if not codes:
            code_pattern = re.compile(r"\|audio_code_(\d+)\|")
            matches = code_pattern.findall(result["text"])
            if matches:
                codes = [int(x) for x in matches]

        if not codes:
            logger.warning("LM generated 0 audio codes — returning silence")
            codes = [0] * min_tokens
        if len(codes) < min_tokens:
            codes.extend([codes[-1]] * (min_tokens - len(codes)))
        return codes[:max_tokens]


# ── <think> block parser ────────────────────────────────────────────────────

THINK_META_RE = re.compile(
    r"<think>(.*?)</think>", re.DOTALL,
)
META_LINE_RE = re.compile(r"\b(bpm|keyscale|timesignature|language|duration|key)\s*[:=]\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _parse_think_meta(text: str) -> dict:
    """Extract metadata from <think> block."""
    meta = {}
    m = THINK_META_RE.search(text)
    if m:
        body = m.group(1)
    else:
        body = text

    for key, val in META_LINE_RE.findall(body):
        key = key.lower().strip()
        val = val.strip()
        if key == "bpm":
            try:
                meta["bpm"] = max(30, min(300, int(float(val.split()[0]))))
            except (ValueError, TypeError):
                pass
        elif key in ("keyscale", "key"):
            meta["keyscale"] = val.split()[0] if val else "C major"
        elif key == "timesignature":
            try:
                meta["timesignature"] = max(2, min(12, int(val.split("/")[0] if "/" in val else val)))
            except (ValueError, TypeError):
                pass
        elif key == "language":
            meta["language"] = val.lower().strip(".,!?\"'")
        elif key == "duration":
            try:
                meta["duration"] = max(1, min(600, float(val.split()[0])))
            except (ValueError, TypeError):
                pass

    return meta
