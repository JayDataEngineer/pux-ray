"""MCP App Host — serves widget HTML and proxies tool calls.

Embedded in the ingress so no separate server is needed.
Handles the MCP Apps protocol used by assistant-ui widgets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

MCP_APP_MIME = "text/html;profile=mcp-app"

_TEMPLATE_DIR = Path(__file__).parent / "mcp_apps"

APPS = {
    "ui://apps/generate": {"name": "Generate", "description": "Run any GPU generation service"},
    "ui://apps/tts": {"name": "TTS Speech", "description": "Text-to-speech with voice design"},
    "ui://apps/audio": {"name": "Audio Studio", "description": "Transcription, sound effects, music"},
    "ui://apps/admin": {"name": "GPU Admin", "description": "GPU status, load/unload services"},
    "ui://apps/workflow": {"name": "Workflow Runner", "description": "Interactive pipeline runner"},
}

# ── TTS Engine catalog — single source of truth for the dynamic TTS endpoint ───

TTS_ENGINES: list[dict[str, Any]] = [
    {
        "id": "kokoro",
        "label": "Kokoro (CPU)",
        "category": "tts",
        "gpu": False,
        "service": "kokoro",
        "description": "Fast CPU text-to-speech, multi-voice. Best for quick generation.",
        "params": [
            {"name": "text", "type": "textarea", "label": "Text", "required": True,
             "placeholder": "Hello world"},
            {"name": "voice", "type": "select", "label": "Voice", "default": "af_bella",
             "options": [
                 "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
                 "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
                 "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
                 "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
                 "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
                 "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
                 "ef_dora", "em_alex", "em_santa", "ff_siwis",
                 "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
                 "if_sara", "im_nicola",
                 "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
                 "jm_kumo", "pf_dora", "pm_alex", "pm_santa",
                 "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                 "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
             ]},
        ],
    },
    {
        "id": "qwen3_tts",
        "label": "Qwen3-TTS (GPU)",
        "category": "tts",
        "gpu": True,
        "service": "wan2gp",
        "model": "faster_qwen3_tts",
        "description": "Qwen3-TTS with CUDA graph acceleration. Supports voice design and cloning.",
        "params": [
            {"name": "text", "type": "textarea", "label": "Text", "required": True,
             "placeholder": "Text to synthesize..."},
            {"name": "mode", "type": "select", "label": "Mode", "default": "custom_voice",
             "options": ["custom_voice", "voice_design", "voice_clone"],
             "description": "custom_voice: preset speaker. voice_design: describe a voice. voice_clone: from reference audio."},
            {"name": "voice", "type": "select", "label": "Voice", "default": "Aiden",
             "options": ["Aiden", "Chloe", "Ethan", "Marcus", "Ono_Anna",
                         "Sohee", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
             "show_when": {"mode": "custom_voice"}},
            {"name": "instruct", "type": "textarea", "label": "Voice Instruction",
             "placeholder": "A warm female voice with a gentle southern accent...",
             "show_when": {"mode": "voice_design"}},
            {"name": "language", "type": "select", "label": "Language", "default": "English",
             "options": ["English", "Chinese", "Japanese", "Korean"]},
        ],
    },
    {
        "id": "moss_tts",
        "label": "MOSS TTS (GPU)",
        "category": "tts",
        "gpu": True,
        "service": "wan2gp",
        "model": "moss-tts",
        "description": "MOSS TTS — text-to-speech with voice cloning via reference audio.",
        "params": [
            {"name": "text", "type": "textarea", "label": "Text", "required": True,
             "placeholder": "Hello world"},
            {"name": "instruct", "type": "textarea", "label": "Instruction",
             "placeholder": "warm, friendly, slightly husky",
             "description": "Optional emotion/style instruction for the voice."},
            {"name": "language", "type": "select", "label": "Language", "default": "English",
             "options": ["English", "Chinese", "Japanese", "Korean"]},
        ],
    },
    {
        "id": "espeak",
        "label": "eSpeak (CPU)",
        "category": "tts",
        "gpu": False,
        "service": "espeak",
        "description": "eSpeak-NG — lightweight phoneme TTS, many languages. Instant CPU inference.",
        "params": [
            {"name": "text", "type": "textarea", "label": "Text", "required": True,
             "placeholder": "Hello world"},
            {"name": "language", "type": "select", "label": "Language", "default": "en",
             "options": ["en", "fr", "de", "es", "it", "ja", "zh", "ko", "ru", "pt"]},
        ],
    },
    {
        "id": "index_tts",
        "label": "IndexTTS (GPU)",
        "category": "tts",
        "gpu": True,
        "service": "wan2gp",
        "model": "index_tts/v2",
        "description": "IndexTTS v2 — high-quality neural TTS with voice cloning.",
        "params": [
            {"name": "text", "type": "textarea", "label": "Text", "required": True,
             "placeholder": "Text to speak..."},
        ],
    },
]

def _get_html(uri: str) -> str | None:
    """Load HTML template for a widget URI."""
    name = uri.replace("ui://apps/", "")
    path = _TEMPLATE_DIR / f"{name}.html"
    if path.exists():
        return path.read_text()
    return None


async def handle_mcp_host(request: Request) -> JSONResponse:
    """POST /mcp/wan2gp-studio/host — MCP Apps protocol handler."""
    body = await request.json()
    method = body.get("method", "")
    params = body.get("params", {})

    # resources/list — list available widget apps
    if method == "resources/list":
        return JSONResponse({
            "resources": [
                {
                    "uri": uri,
                    "name": info["name"],
                    "description": info["description"],
                    "mimeType": MCP_APP_MIME,
                    "_meta": {"ui": {"prefersBorder": True}},
                }
                for uri, info in APPS.items()
            ],
        })

    # mcp-apps/read-resource — return widget HTML
    if method == "mcp-apps/read-resource":
        uri = params.get("uri", "")
        html = _get_html(uri)
        if html is None:
            return JSONResponse({"error": f"Resource not found: {uri}"}, status_code=404)
        return JSONResponse({
            "uri": uri,
            "mimeType": MCP_APP_MIME,
            "html": html,
            "meta": {"ui": {"prefersBorder": True}},
        })

    # tools/call — proxy MCP tool calls through the ingress dispatch
    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        # ── tts_voices: list available TTS engines and their parameters ──────
        if tool_name == "tts_voices":
            return JSONResponse({"engines": TTS_ENGINES})

        # ── tts_speak: unified dynamic TTS — one endpoint, all engines ──────
        if tool_name == "tts_speak":
            engine_id = tool_args.get("engine", "kokoro")
            engine = next((e for e in TTS_ENGINES if e["id"] == engine_id), None)
            if engine is None:
                return JSONResponse(
                    {"status": "error", "error": f"Unknown TTS engine: {engine_id}. "
                     f"Available: {[e['id'] for e in TTS_ENGINES]}"},
                    status_code=400,
                )

            from gateway.ingress import APIIngress
            ingress = APIIngress()

            try:
                if engine_id == "kokoro":
                    result = await ingress._dispatch_service("kokoro", {
                        "text": tool_args.get("text", ""),
                        "voice": tool_args.get("voice", "af_bella"),
                    })

                elif engine_id == "espeak":
                    result = await ingress._dispatch_service("espeak", {
                        "text": tool_args.get("text", ""),
                        "language": tool_args.get("language", "en"),
                    })

                elif engine_id == "qwen3_tts":
                    mode = tool_args.get("mode", "custom_voice")
                    payload: dict[str, Any] = {
                        "model": "faster_qwen3_tts",
                        "text": tool_args.get("text", ""),
                        "language": tool_args.get("language", "English"),
                        "mode": mode,
                    }
                    if mode == "voice_design":
                        payload["instruct"] = tool_args.get("instruct", "")
                    elif mode == "voice_clone":
                        payload["ref_audio_b64"] = tool_args.get("ref_audio_b64", "")
                        if not payload["ref_audio_b64"]:
                            return JSONResponse(
                                {"status": "error", "error": "ref_audio_b64 is required for voice_clone mode"},
                                status_code=400,
                            )
                    else:
                        payload["voice"] = tool_args.get("voice", "Aiden")
                    result = await ingress._dispatch_service("wan2gp", payload)

                elif engine_id == "moss_tts":
                    payload: dict[str, Any] = {
                        "model": "moss-tts",
                        "text": tool_args.get("text", ""),
                        "language": tool_args.get("language", "English"),
                    }
                    if tool_args.get("instruct"):
                        payload["instruction"] = tool_args["instruct"]
                    if tool_args.get("ref_audio_b64"):
                        payload["ref_audio_b64"] = tool_args["ref_audio_b64"]
                    result = await ingress._dispatch_service("wan2gp", payload)

                elif engine_id == "index_tts":
                    payload: dict[str, Any] = {
                        "model": "index_tts/v2",
                        "text": tool_args.get("text", ""),
                    }
                    result = await ingress._dispatch_service("wan2gp", payload)

                else:
                    return JSONResponse(
                        {"status": "error", "error": f"TTS engine {engine_id} not implemented"},
                        status_code=501,
                    )

                # Normalize result: extract audio_url for the widget
                audio_url = None
                if isinstance(result, dict):
                    audio_url = result.get("audio_url") or result.get("url")
                    # If raw base64 data, return as data URI
                    if not audio_url and result.get("data"):
                        fmt = result.get("format", "wav")
                        mime = f"audio/{fmt}" if fmt != "opus" else "audio/ogg"
                        audio_url = f"data:{mime};base64,{result['data']}"

                return JSONResponse({
                    "engine": engine_id,
                    "audio_url": audio_url,
                    "result": result,
                })

            except ValueError as e:
                return JSONResponse({"status": "error", "error": str(e)}, status_code=404)
            except Exception as e:
                logger.exception("tts_speak failed for engine=%s", engine_id)
                return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

        # The 'run' tool: { service, params, model? }
        if tool_name == "run":
            from gateway.ingress import APIIngress
            ingress = APIIngress()
            service = tool_args.get("service", "wan2gp")
            tool_params = tool_args.get("params", {})
            try:
                result = await ingress._dispatch_service(service, tool_params)
                return JSONResponse({"result": result})
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=404)

        # Generic tool: try all registered tools
        return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=404)

    return JSONResponse({"error": f"Unknown method: {method}"}, status_code=400)
