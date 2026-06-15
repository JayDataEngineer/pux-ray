"""Native model serving — transformers 4 + custom VRAM hooks.

NO diffusers. NO SGLang serve. NO mmGP. NO Wan2GP.

Architecture:
  registry.py      — Model configs (repo, defaults)
  loader.py        — Custom VRAM package (hooks + CUDA streams + pinned memory)
  lora.py          — LoRA via PEFT
  service.py       — transformers 4 loading + manual denoise loop
  ltx_sequencer.py — LTX via ltx-pipelines (Lightricks native)
  forge_adapter.py — Forge integration

Models loaded via transformers 4 AutoModel.from_pretrained().
VRAM managed by our BlockStreamHook (register_forward_pre_hook + CUDA streams).
Denoising loop is manual — we control every step.
NO diffusers pipelines. NO SGLang serving engine.
"""
