#!/usr/bin/env python3
"""Wrapper: patch multiprocessing to block fork, then launch vllm-omni API server.

Mount this into the container at /patches/omni_patch_fork.py and use it as
the entrypoint instead of python3 -m vllm_omni.entrypoints.openai.api_server.

Works around multiproc_executor.py:191 which does:
    mp.set_start_method("fork", force=True)
This is incompatible with CUDA when the parent process has already initialized CUDA.
"""
import os, sys, multiprocessing as mp

# ── Patch 1: block fork start method ──────────────────────────────────────
_orig_set_start = mp.set_start_method

def _no_fork_set_start(method, force=False):
    if method == "fork":
        # Silently swallow — "spawn" is compatible with CUDA subprocesses
        return
    return _orig_set_start(method, force)

mp.set_start_method = _no_fork_set_start
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# ── Patch 2: get_hf_file_to_dict for local paths ──────────────────────────
# vllm-omni's enrich_config() loads model_index.json / config.json from disk.
# The function is imported *inside* the method body at data.py:878, so we
# patch both the module-level and the re-export in the config module.
import json as _json
from vllm.transformers_utils import repo_utils

_orig_get_hf = repo_utils.get_hf_file_to_dict
def _patched_get_hf(config_name, model, revision=None, **kw):
    path = os.path.join(str(model), config_name)
    if os.path.isfile(path):
        with open(path) as f:
            return _json.load(f)
    return _orig_get_hf(config_name, model, revision=revision, **kw)

repo_utils.get_hf_file_to_dict = _patched_get_hf
import vllm.transformers_utils.config as t_config
t_config.get_hf_file_to_dict = _patched_get_hf

# ── Launch (mimics the __main__ block in api_server.py) ──────────────────
if __name__ == "__main__":
    import asyncio
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    parser = TrackingArgumentParser(description="vLLM-Omni OpenAI-Compatible REST API server")
    parser = make_arg_parser(parser)
    registered_flags = set()
    for action in parser._actions:
        registered_flags.update(action.option_strings)

    if "--omni" not in registered_flags:
        parser.add_argument("--omni", action="store_true", default=False, help="Enable vLLM-Omni mode.")
    if "--enable-sleep-mode" not in registered_flags:
        parser.add_argument("--enable-sleep-mode", action="store_true", default=False,
                            help="Enable GPU memory pool for sleep mode.")
    args = parser.parse_args()
    if not hasattr(args, "model_tag"):
        setattr(args, "model_tag", args.model)
    if hasattr(args, "model_tag") and args.model_tag is None:
        args.model_tag = args.model

    from vllm_omni.entrypoints.openai.api_server import omni_run_server
    asyncio.run(omni_run_server(args))
