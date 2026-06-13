"""Kimodo launcher — monkey-patches huggingface_hub before kimodo loads.

transformers' _patch_mistral_regex calls model_info() which hits the network.
On air-gapped pods this triggers OfflineModeIsEnabled and crashes.
This script patches model_info to a no-op before any of that runs.
"""
import os
import sys

# ── Patch: Kill network calls from huggingface_hub ──
class _FakeModelInfo:
    tags = []
    library_name = None
    def __init__(self, *a, **kw): pass

import huggingface_hub
import huggingface_hub.hf_api as _hfapi
_hfapi.model_info = lambda *a, **kw: _FakeModelInfo()
huggingface_hub.model_info = _hfapi.model_info

# ── Ensure offline mode ──
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── Start kimodo ──
import kimodo.demo
model = os.environ.get("KIMODO_MODEL", "kimodo-soma-rp")
sys.argv = ["kimodo_demo", "--model", model]
kimodo.demo.main()
