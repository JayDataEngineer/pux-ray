"""Native model serving via SGLang — NO diffusers, NO mmGP.

Architecture:
  registry.py      — Model configs (repo, defaults) for SGLang
  loader.py        — SGLang config manager (performance mode, sleep/wake)
  lora.py          — LoRA management (SGLang /v1/loras/load API)
  service.py       — ForgeService via SGLang HTTP API
  ltx_sequencer.py — LTX advanced features via ltx-pipelines
  forge_adapter.py — Forge integration

SGLang serves models (sglang serve --model-type diffusion).
Service code calls SGLang's OpenAI-compatible HTTP API.
NO diffusers. NO mmGP. NO Wan2GP.
"""
