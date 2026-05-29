"""MCP App resource registry.

Maps resource URIs to HTML content for the assistant-ui MCP app protocol.
Each app is a self-contained HTML page that can call back to MCP tools.

URIs use the ui:// scheme per the MCP Apps specification.
"""
from __future__ import annotations

MCP_APP_MIME = "text/html;profile=mcp-app"

APPS: dict[str, dict] = {
    "ui://apps/workflow": {
        "name": "Workflow Runner",
        "description": "Interactive DAG workflow runner — select a pipeline, start a run, execute steps",
        "resourceUri": "ui://apps/workflow",
    },
    "ui://apps/tts": {
        "name": "TTS Speech",
        "description": "Text-to-speech with voice design and cloning",
        "resourceUri": "ui://apps/tts",
    },
    "ui://apps/audio": {
        "name": "Audio Studio",
        "description": "Transcription, sound effects, and music generation",
        "resourceUri": "ui://apps/audio",
    },
    "ui://apps/generate": {
        "name": "Generate",
        "description": "Run any GPU generation service",
        "resourceUri": "ui://apps/generate",
    },
    "ui://apps/admin": {
        "name": "GPU Admin",
        "description": "GPU status, load/unload services",
        "resourceUri": "ui://apps/admin",
    },
}

# The actual HTML for each app widget
HTML_TEMPLATES: dict[str, str] = {}


def _load_templates():
    """Load HTML templates from the templates directory."""
    import os
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    if not os.path.isdir(template_dir):
        return
    for fname in os.listdir(template_dir):
        if fname.endswith(".html"):
            uri = f"ui://apps/{fname[:-5]}"
            with open(os.path.join(template_dir, fname)) as f:
                HTML_TEMPLATES[uri] = f.read()


def get_app_html(uri: str) -> str | None:
    """Get the HTML content for an app resource URI."""
    if not HTML_TEMPLATES:
        _load_templates()
    return HTML_TEMPLATES.get(uri)


def list_apps() -> list[dict]:
    """List all registered MCP apps."""
    return list(APPS.values())
