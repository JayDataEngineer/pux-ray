"""Native model serving — direct diffusers pipelines, no mmGP/Wan2GP.

Architecture:
  registry.py  — Model configurations (pipeline class, repo, defaults)
  loader.py    — Adaptive VRAM loader (replaces mmGP)
  lora.py      — LoRA management via PEFT
  service.py   — ForgeService implementation
  forge_adapter.py — Forge integration

All models (Z-Image, FLUX, Anima, Wan, LTX) load via diffusers
from_pretrained() with adaptive VRAM optimization.
"""
