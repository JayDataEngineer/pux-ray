#!/usr/bin/env python3
"""
Bootstrap launcher for Qwen-Image-Edit-2511 FP8 weight-only.

Patches OmniDiffusionConfig to inject env-var overrides, then delegates
to the original omni_run_server.

Env-var overrides (set via docker -e):
  DIFFUSION_CACHE_BACKEND   → cache_backend (e.g. "cache_dit")
  DIFFUSION_CACHE_CONFIG    → cache_config as JSON
  DIFFUSION_VAE_USE_SLICING → enable VAE slicing ("1").
  DIFFUSION_VAE_USE_TILING  → enable VAE tiling ("1").
"""
import os
import sys
import json as _json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("launcher")

# ─── Patch: Inject env-var overrides into OmniDiffusionConfig ─────────────
# We need this because the api_server CLI doesn't expose cache_backend,
# vae_use_slicing, vae_use_tiling, or cache_config as flags. The env vars
# bypass the CLI parser and apply directly to the config object.
from vllm_omni.diffusion import data as diffusion_data

_orig_post_init = diffusion_data.OmniDiffusionConfig.__post_init__

def _patched_post_init(self):
    env_overrides = {
        "cache_backend": os.environ.get("DIFFUSION_CACHE_BACKEND"),
        "cache_config": os.environ.get("DIFFUSION_CACHE_CONFIG"),
        "vae_use_slicing": os.environ.get("DIFFUSION_VAE_USE_SLICING"),
        "vae_use_tiling": os.environ.get("DIFFUSION_VAE_USE_TILING"),
    }
    for key, val in env_overrides.items():
        if val is not None:
            if key in ("vae_use_slicing", "vae_use_tiling"):
                val = val.lower() in ("1", "true", "yes")
                logger.warning("Env override: %s = %s", key, val)
            elif key == "cache_config":
                try:
                    val = _json.loads(val)
                    logger.warning("Env override: %s = %s", key, val)
                except _json.JSONDecodeError:
                    logger.warning("Invalid JSON for %s, skipping", key)
                    continue
            setattr(self, key, val)
    return _orig_post_init(self)

diffusion_data.OmniDiffusionConfig.__post_init__ = _patched_post_init
logger.warning("Patched OmniDiffusionConfig.__post_init__ with env-var overrides")

# ─── Run the original api_server main block ──────────────────────────────
if __name__ == "__main__":
    import asyncio
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm_omni.entrypoints.openai.api_server import omni_run_server
    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    parser = TrackingArgumentParser(description="vLLM-Omni OpenAI-Compatible REST API server (launcher)")
    parser = make_arg_parser(parser)
    registered_flags = set()
    for action in parser._actions:
        registered_flags.update(action.option_strings)

    if "--omni" not in registered_flags:
        parser.add_argument("--omni", action="store_true", default=False, help="Enable vLLM-Omni mode.")
    if "--enable-sleep-mode" not in registered_flags:
        parser.add_argument(
            "--enable-sleep-mode", action="store_true", default=False,
            help="Enable GPU memory pool for sleep mode.",
        )
    args = parser.parse_args()
    if not hasattr(args, "model_tag"):
        setattr(args, "model_tag", args.model)
    if hasattr(args, "model_tag") and args.model_tag is None:
        args.model_tag = args.model
    asyncio.run(omni_run_server(args))
