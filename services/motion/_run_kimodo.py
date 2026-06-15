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

# ── Patch: Bridge downloads to the parent editor's asset store ──────────────
# Kimodo's export buttons inject JS that triggers browser downloads. We
# intercept those injections and also fire window.parent.postMessage() so
# the Tech Noir editor (which hosts Kimodo in an iframe, same origin) can
# catch the payload and add it to the IndexedDB asset store automatically.

import base64 as _b64
import dataclasses as _dc
import json as _json

# Injected before `})();` in the bytes-download IIFE.
# `filename`, `mimeType`, and `b64` are already in scope there.
_BYTES_PM = """\
  // Bridge to Tech Noir asset store
  if (typeof mimeType === 'string' && (mimeType.startsWith('video/') || mimeType.startsWith('image/'))) {
    window.parent.postMessage({ type: 'kimodo-asset', filename: filename, mime: mimeType, b64: b64 }, '*');
  }
"""

# Injected inside the canvas.toBlob() callback, after URL.revokeObjectURL.
# `blob` and `filename` are in scope there.
_CANVAS_SEARCH = '    URL.revokeObjectURL(url);\n  }, "image/png");'
_CANVAS_REPLACE = """\
    URL.revokeObjectURL(url);
    // Bridge to Tech Noir asset store
    const _krd = new FileReader();
    _krd.onload = () => window.parent.postMessage({
      type: 'kimodo-asset', filename: filename, mime: 'image/png',
      b64: _krd.result.split(',')[1]
    }, '*');
    _krd.readAsDataURL(blob);
  }, "image/png");"""


def _patch_js_source(src: str) -> str:
    if 'canvas.toBlob' in src and _CANVAS_SEARCH in src:
        return src.replace(_CANVAS_SEARCH, _CANVAS_REPLACE, 1)
    if "const b64 = " in src and 'a.download' in src and 'canvas.toBlob' not in src:
        marker = '})();'
        idx = src.rfind(marker)
        if idx >= 0:
            return src[:idx] + _BYTES_PM + src[idx:]
    return src


import kimodo.demo.app as _kapp

_orig_on_client_connect = _kapp.Demo.on_client_connect


def _patched_on_client_connect(self, client):
    """Wrap each client's queue_message to intercept JS download injections."""
    from viser import _messages as _vm
    iface = client.gui._websock_interface
    _orig_qm = iface.queue_message

    def _wrapped_qm(msg):
        if isinstance(msg, _vm.RunJavascriptMessage):
            try:
                new_src = _patch_js_source(msg.source)
                if new_src != msg.source:
                    msg = _dc.replace(msg, source=new_src)
            except Exception:
                pass
        return _orig_qm(msg)

    iface.queue_message = _wrapped_qm
    _orig_on_client_connect(self, client)


_kapp.Demo.on_client_connect = _patched_on_client_connect


# send_file_download (video MP4) goes through _websock_connection and bypasses
# queue_message, so patch it separately at the class level.
import viser as _viser

_orig_send_file_download = _viser.ClientHandle.send_file_download

_FILE_DOWNLOAD_MIMES = {
    'mp4': 'video/mp4', 'webm': 'video/webm',
    'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
}


def _patched_send_file_download(self, filename, content, chunk_size=1024 * 1024, save_immediately=False):
    _orig_send_file_download(self, filename, content, chunk_size, save_immediately)
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    mime = _FILE_DOWNLOAD_MIMES.get(ext)
    if mime:
        b64 = _b64.b64encode(content).decode('ascii')
        js = (
            "window.parent.postMessage({"
            f"type:'kimodo-asset',"
            f"filename:{_json.dumps(filename)},"
            f"mime:{_json.dumps(mime)},"
            f"b64:{_json.dumps(b64)}"
            "},'*');"
        )
        from viser import _messages as _vm
        self.gui._websock_interface.queue_message(_vm.RunJavascriptMessage(source=js))


_viser.ClientHandle.send_file_download = _patched_send_file_download

# ── End bridge patch ─────────────────────────────────────────────────────────

model = os.environ.get("KIMODO_MODEL", "kimodo-soma-rp")
sys.argv = ["kimodo_demo", "--model", model]
kimodo.demo.main()
