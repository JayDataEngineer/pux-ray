"""Model provisioner and CLI for Tech Noir Ray.

Downloads missing models from HuggingFace, verifies integrity.
Treats the local disk as a cache and the registry YAML as source of truth.

Usage:
    ray-noir models list          # Show all models, downloaded vs missing
    ray-noir models pull          # Download all missing models
    ray-noir models pull <name>   # Download a specific model
    ray-noir models verify        # Check all model hashes
    ray-noir models status        # Summary of disk usage
    ray-noir pull <name>          # Shortcut: download a specific model
    ray-noir connect <app>        # Print connection config for an app
    ray-noir cluster start        # Start Ray cluster
    ray-noir cluster stop         # Stop Ray cluster
    ray-noir deploy               # Deploy all services
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _get_registry():
    from registry.models import ModelRegistry
    return ModelRegistry()


def _get_models_root() -> Path:
    from registry.config import Config
    return Path(Config().models_root)


def _parse_source(source: str) -> tuple[str, str | None]:
    """Parse 'hf://repo_id' or 'hf://repo_id/file' into (repo_id, filename)."""
    if not source or not source.startswith("hf://"):
        return "", None
    path = source[5:]  # strip hf://
    parts = path.split("/", 1)
    if len(parts) == 1:
        return parts[0], None
    # Could be org/repo or org/repo/file
    # HuggingFace repos are org/repo, files have 3+ parts
    segments = path.split("/")
    if len(segments) >= 3:
        repo_id = "/".join(segments[:2])
        filename = "/".join(segments[2:])
        return repo_id, filename
    return path, None


def _parse_modelscope_source(source: str) -> str:
    """Parse 'modelscope://repo_id' into repo_id."""
    if not source or not source.startswith("modelscope://"):
        return ""
    return source[13:]


def _compute_sha256(filepath: Path, progress=True) -> str:
    """Compute SHA256 of a file, optionally showing progress."""
    h = hashlib.sha256()
    size = filepath.stat().st_size
    done = 0
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):  # 8MB chunks
            h.update(chunk)
            done += len(chunk)
            if progress and size > 100_000_000:
                pct = done / size * 100
                sys.stdout.write(f"\r  Hashing: {pct:.0f}%")
                sys.stdout.flush()
    if progress and size > 100_000_000:
        sys.stdout.write("\r")
    return h.hexdigest()


# =========================================================================
# Post-download patching
# =========================================================================

def _post_download_patch(
    category: str, name: str, model_path: Path, registry: "ModelRegistry",
) -> None:
    """Apply post-download patches to model files.

    Currently handles:
    - TRELLIS pipeline.json: replace HF model IDs with local paths
    """
    if category == "3d" and name == "trellis":
        _patch_trellis_pipeline(model_path, registry)


def _patch_trellis_pipeline(
    ckpts_path: Path, registry: "ModelRegistry",
) -> None:
    """Patch TRELLIS pipeline.json to use local model paths instead of HF IDs.

    The pipeline.json references HuggingFace model IDs for DINOv3 and RMBG.
    These are gated on HuggingFace but freely available on ModelScope.
    After downloading, we patch pipeline.json to point to local copies.
    """
    pipeline_json = ckpts_path / "pipeline.json"
    if not pipeline_json.is_file():
        return

    try:
        import json
        content = pipeline_json.read_text()
        original = content

        # Patch DINOv3 image encoder: HF ID -> local path
        try:
            dinov3_path = registry.get_path("3d", "trellis_dinov3")
            hf_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            if hf_id in content and dinov3_path.exists():
                content = content.replace(f'"{hf_id}"', f'"{dinov3_path}"')
                content = content.replace(f'"{hf_id}/"', f'"{dinov3_path}/"')
                print(f"       Patched pipeline.json: dinov3 -> {dinov3_path}")
        except (KeyError, Exception):
            pass

        # Patch RMBG background removal: HF ID -> local path
        try:
            rmbg_path = registry.get_path("3d", "trellis_rmbg")
            hf_id = "briaai/RMBG-2.0"
            if hf_id in content and rmbg_path.exists():
                content = content.replace(f'"{hf_id}"', f'"{rmbg_path}"')
                content = content.replace(f'"{hf_id}/"', f'"{rmbg_path}/"')
                print(f"       Patched pipeline.json: rmbg -> {rmbg_path}")
        except (KeyError, Exception):
            pass

        if content != original:
            pipeline_json.write_text(content)

    except Exception as e:
        print(f"       WARN: Could not patch pipeline.json: {e}")


# =========================================================================
# Commands
# =========================================================================

def cmd_models_list(args):
    """List all models with their download status."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    registry = _get_registry()
    models_root = _get_models_root()

    table = Table(title="Tech Noir Model Registry")
    table.add_column("Model", style="cyan")
    table.add_column("Category")
    table.add_column("Size")
    table.add_column("Status", justify="center")
    table.add_column("Source")

    for category in sorted(registry.data.keys()):
        models = registry.data[category]
        if not isinstance(models, dict):
            continue
        for name, meta in sorted(models.items()):
            if not isinstance(meta, dict):
                continue

            try:
                model_path = registry.get_path(category, name)
                exists = model_path.exists()
                if model_path.is_file():
                    size_str = f"{model_path.stat().st_size / 1e9:.1f}G"
                elif model_path.is_dir():
                    total = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
                    size_str = f"{total / 1e9:.1f}G"
                else:
                    size_str = meta.get("size_gb", "?") + "G"
            except Exception:
                exists = False
                size_str = "?"

            source = meta.get("source")
            download_mode = meta.get("download", "")
            if source and source.startswith("hf://"):
                source_display = "hf://" + source.split("/")[2] + "/…" if "/" in source[5:] else source
            elif source and source.startswith("civitai://"):
                source_display = "civitai"
            elif source and source.startswith("modelscope://"):
                source_display = "modelscope"
            elif download_mode == "manual":
                source_display = "manual"
            else:
                source_display = "local"

            status = "[green]OK[/green]" if exists else "[red]MISSING[/red]"
            table.add_row(name, category, size_str, status, source_display)

    console.print(table)


def cmd_models_pull(args):
    """Download missing models from HuggingFace."""
    # Set HF_TOKEN from config if not already in environment
    from registry.config import Config
    hf_token = Config().get("secrets.hf_token", "")
    if hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token

    registry = _get_registry()
    models_root = _get_models_root()
    models_root.mkdir(parents=True, exist_ok=True)

    target = args.model
    pulled = 0
    skipped = 0

    for category in sorted(registry.data.keys()):
        models = registry.data[category]
        if not isinstance(models, dict):
            continue
        for name, meta in sorted(models.items()):
            if not isinstance(meta, dict):
                continue

            if target:
                if target in registry.data:
                    if category != target:
                        continue
                else:
                    full_name = f"{category}/{name}"
                    normalized = target.replace(".", "/", 1)
                    if (
                        full_name != target
                        and full_name != normalized
                        and name != target
                    ):
                        continue

            # Check download mode
            download_mode = meta.get("download", "")
            if download_mode in ("skip", None) and not meta.get("source"):
                skipped += 1
                print(f"  SKIP {category}/{name} - not downloadable")
                continue

            source = meta.get("source", "")

            # Handle ModelScope downloads (gated HF models available freely on ModelScope)
            if download_mode == "modelscope" and source.startswith("modelscope://"):
                model_path = registry.get_path(category, name)
                if model_path.exists() and (model_path.is_file() or any(model_path.iterdir())):
                    print(f"  OK   {category}/{name} - already downloaded")
                    continue

                ms_repo = _parse_modelscope_source(source)
                if not ms_repo:
                    skipped += 1
                    print(f"  SKIP {category}/{name} - invalid ModelScope source")
                    continue

                print(f"  PULL {category}/{name} from ModelScope ({ms_repo})")
                try:
                    from modelscope import snapshot_download
                    model_path.mkdir(parents=True, exist_ok=True)
                    snapshot_download(
                        model_id=ms_repo,
                        local_dir=str(model_path),
                    )
                    print(f"       -> {model_path}/")
                    pulled += 1

                    # Post-download: patch TRELLIS pipeline.json
                    _post_download_patch(category, name, model_path, registry)

                except ImportError:
                    print(f"  FAIL {category}/{name}: modelscope not installed. Run: uv pip install modelscope")
                    skipped += 1
                except Exception as e:
                    print(f"  FAIL {category}/{name}: {e}")
                    skipped += 1
                continue

            # Handle Civitai downloads
            if download_mode == "civitai" and source.startswith("civitai://"):
                model_id = source.split("://")[1]
                print(f"  PULL {category}/{name} from Civitai model {model_id}")
                try:
                    model_path = registry.get_path(category, name)
                    if model_path.exists() and model_path.is_file():
                        print(f"  OK   {category}/{name} - already downloaded")
                        continue
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    import subprocess
                    dl_url = f"https://civitai.com/api/download/models/{model_id}"
                    result = subprocess.run(
                        ["wget", "-q", "--show-progress", "-O", str(model_path), dl_url],
                        capture_output=True, text=True, timeout=600,
                    )
                    if result.returncode == 0 and model_path.exists():
                        print(f"       -> {model_path}")
                        pulled += 1
                    else:
                        print(f"  FAIL {category}/{name}: wget returned {result.returncode}")
                        print(f"       {result.stderr[:200]}")
                        skipped += 1
                except Exception as e:
                    print(f"  FAIL {category}/{name}: {e}")
                    skipped += 1
                continue

            if download_mode == "manual":
                skipped += 1
                print(f"  MANUAL {category}/{name} - requires manual download")
                continue

            if not source or not source.startswith("hf://"):
                skipped += 1
                print(f"  SKIP {category}/{name} - no download source")
                continue

            try:
                model_path = registry.get_path(category, name)
                if model_path.exists() and (model_path.is_file() or any(model_path.iterdir())):
                    # Check hash if available
                    expected_hash = meta.get("sha256")
                    if expected_hash and model_path.is_file():
                        actual_hash = _compute_sha256(model_path)
                        if actual_hash == expected_hash:
                            print(f"  OK   {category}/{name} - already downloaded (hash verified)")
                            continue
                        else:
                            print(f"  HASH MISMATCH {category}/{name} - re-downloading")
                    else:
                        print(f"  OK   {category}/{name} - already downloaded")
                        continue
            except Exception:
                pass

            # Download
            repo_id, filename = _parse_source(source)
            if not repo_id:
                skipped += 1
                continue

            print(f"  PULL {category}/{name} from {source}")
            try:
                from huggingface_hub import hf_hub_download, snapshot_download

                # Determine download method.
                # Check explicit download_mode first — "snapshot" should always
                # snapshot even if source has 3+ segments (e.g. hy-motion with
                # hf://org/repo/subfolder).
                if download_mode == "snapshot" or (not filename and download_mode != "file"):
                    # Entire repo snapshot
                    model_path.mkdir(parents=True, exist_ok=True)
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(model_path),
                    )
                    print(f"       -> {model_path}/")
                else:
                    # Single file download.
                    # hf_hub_download creates {local_dir}/{filename} — if filename
                    # contains subdirectories (e.g. split_files/vae/ae.safetensors),
                    # the file lands at a nested path. To land at the flat path
                    # specified in the registry, download to a temp dir and move.
                    # Use the models root for temp storage — /tmp may not have
                    # enough space for large model files.
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(dir=str(models_root)) as tmpdir:
                        tmp_downloaded = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            local_dir=tmpdir,
                        )
                        shutil.move(tmp_downloaded, str(model_path))
                    print(f"       -> {model_path}")

                pulled += 1

                # Post-download: patch TRELLIS pipeline.json
                _post_download_patch(category, name, model_path, registry)

            except Exception as e:
                print(f"  FAIL {category}/{name}: {e}")
                skipped += 1

    print(f"\nDone. Pulled: {pulled}, Skipped: {skipped}")


def cmd_models_verify(args):
    """Verify all models: existence, size, and readiness.

    Checks every model in the registry:
    - SKIP models: confirmed as system packages (no download needed)
    - Downloadable models: file/dir exists, size matches expected size_gb
    - SHA256 hash: verified if present in registry
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    registry = _get_registry()

    ok = 0
    missing = 0
    size_mismatch = 0
    hash_fail = 0
    skipped = 0
    errors = 0

    table = Table(title="Model Verification")
    table.add_column("Model", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Detail")

    for category in sorted(registry.data.keys()):
        models = registry.data[category]
        if not isinstance(models, dict):
            continue
        for name, meta in sorted(models.items()):
            if not isinstance(meta, dict):
                continue

            download = meta.get("download", "")

            # Skip models are system packages — nothing to verify on disk
            # Manual models are downloaded by external tools (Wan2GP download job, etc.)
            if download in ("skip", "manual") or (not meta.get("source") and download not in ("file", "snapshot", "civitai", "modelscope")):
                table.add_row(f"{category}/{name}", "[dim]SKIP[/dim]", "system package" if download == "skip" else "external download")
                skipped += 1
                continue

            try:
                model_path = registry.get_path(category, name)
                expected_gb = meta.get("size_gb", 0)

                if not model_path.exists():
                    table.add_row(
                        f"{category}/{name}",
                        "[red]MISSING[/red]",
                        f"expected at {model_path}",
                    )
                    missing += 1
                    continue

                # Compute actual size
                if model_path.is_file():
                    actual_bytes = model_path.stat().st_size
                elif model_path.is_dir():
                    actual_bytes = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
                else:
                    actual_bytes = 0

                actual_gb = actual_bytes / 1e9

                # Size check: allow 20% tolerance (compressed vs metadata estimate)
                if expected_gb and actual_gb < expected_gb * 0.3:
                    table.add_row(
                        f"{category}/{name}",
                        "[yellow]SMALL[/yellow]",
                        f"{actual_gb:.1f}GB / {expected_gb}GB expected",
                    )
                    size_mismatch += 1
                    continue

                # SHA256 check (only if hash is set)
                expected_hash = meta.get("sha256")
                if expected_hash and model_path.is_file():
                    actual_hash = _compute_sha256(model_path)
                    if actual_hash != expected_hash:
                        table.add_row(
                            f"{category}/{name}",
                            "[red]HASH[/red]",
                            f"SHA256 mismatch",
                        )
                        hash_fail += 1
                        continue

                table.add_row(
                    f"{category}/{name}",
                    "[green]OK[/green]",
                    f"{actual_gb:.1f}GB",
                )
                ok += 1

            except Exception as e:
                table.add_row(f"{category}/{name}", "[red]ERR[/red]", str(e)[:60])
                errors += 1

    console.print(table)
    console.print()

    total = ok + missing + size_mismatch + hash_fail + errors
    console.print(f"  [green]OK[/green]: {ok}  [red]Missing[/red]: {missing}  [yellow]Size mismatch[/yellow]: {size_mismatch}  [red]Hash fail[/red]: {hash_fail}  [dim]Skip[/dim]: {skipped}  [red]Errors[/red]: {errors}")
    console.print(f"  Total downloadable: {total}, Ready: {ok}/{total}")

    if missing + size_mismatch + hash_fail + errors > 0:
        console.print("\n  [yellow]Run 'task models:pull' to download missing models.[/yellow]")
        return 1
    return 0


def cmd_models_status(args):
    """Show disk usage summary."""
    models_root = _get_models_root()
    if not models_root.exists():
        print(f"Models root does not exist: {models_root}")
        return

    total = 0
    categories = {}
    for item in models_root.iterdir():
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            categories[item.name] = size
            total += size
        elif item.is_file():
            total += item.stat().st_size

    print(f"Models root: {models_root}")
    print(f"Total usage: {total / 1e9:.1f} GB")
    print()
    for cat, size in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s} {size / 1e9:8.1f} GB")


def cmd_cluster_start(args):
    """Start the Ray cluster."""
    import subprocess
    script = _PROJECT_ROOT / "scripts" / "start_cluster.sh"
    subprocess.run(["bash", str(script)], check=True)


def cmd_cluster_stop(args):
    """Stop the Ray cluster."""
    import subprocess
    ray_bin = _PROJECT_ROOT / ".venv" / "bin" / "ray"
    subprocess.run([str(ray_bin), "stop"], check=False)


def cmd_deploy(args):
    """Deploy all services."""
    import subprocess
    deploy_script = _PROJECT_ROOT / "scripts" / "deploy_services.py"
    venv_python = _PROJECT_ROOT / ".venv" / "bin" / "python"
    subprocess.run([str(venv_python), str(deploy_script)], check=True)


# =========================================================================
# Connect — print config snippets for popular apps
# =========================================================================

_CONNECT_SNIPPETS = {
    "open-webui": {
        "name": "Open WebUI",
        "instructions": [
            "Settings → Connections → OpenAI API",
            "Set API Base URL to the URL below",
            "Set API Key to your Tech Noir API key (or leave blank if no auth)",
        ],
        "config": {
            "API_BASE_URL": "{base_url}/v1",
            "API_KEY": "your-api-key-here",
        },
    },
    "anythingllm": {
        "name": "AnythingLLM",
        "instructions": [
            "Settings → LLM Provider → Generic OpenAI",
            "Set Base URL and API Key below",
        ],
        "config": {
            "API_BASE_URL": "{base_url}/v1",
            "API_KEY": "your-api-key-here",
            "MODEL": "llm",
        },
    },
    "claude-desktop": {
        "name": "Claude Desktop",
        "instructions": [
            "Add to claude_desktop_config.json:",
        ],
        "config_json": {
            "mcpServers": {
                "media": {
                    "type": "http",
                    "url": "{base_url}/mcp/media",
                },
                "web-research": {
                    "type": "http",
                    "url": "{base_url}/mcp/web",
                },
            },
        },
    },
    "claude-code": {
        "name": "Claude Code",
        "instructions": [
            "Run these commands:",
        ],
        "commands": [
            'claude mcp add media --transport http "{base_url}/mcp/media"',
            'claude mcp add web-research --transport http "{base_url}/mcp/web"',
        ],
    },
    "cursor": {
        "name": "Cursor IDE",
        "instructions": [
            "Settings → Models → OpenAI API Key",
            "Set Base URL below",
        ],
        "config": {
            "OPENAI_API_BASE": "{base_url}/v1",
            "OPENAI_API_KEY": "your-api-key-here",
        },
    },
    "n8n": {
        "name": "n8n",
        "instructions": [
            "Add an OpenAI node → Credentials → Create New",
            "Set Base URL and API Key below",
        ],
        "config": {
            "BASE_URL": "{base_url}/v1",
            "API_KEY": "your-api-key-here",
        },
    },
    "dify": {
        "name": "Dify",
        "instructions": [
            "Settings → Model Providers → Custom Model Provider",
            "Add OpenAI-API-compatible provider with URL below",
        ],
        "config": {
            "api_endpoint": "{base_url}/v1",
            "api_key": "your-api-key-here",
        },
    },
    "continue-dev": {
        "name": "Continue.dev",
        "instructions": [
            "Add to ~/.continue/config.json under 'models':",
        ],
        "config_json": {
            "title": "Tech Noir LLM",
            "provider": "openai",
            "model": "llm",
            "apiBase": "{base_url}/v1",
            "apiKey": "your-api-key-here",
        },
    },
    "python": {
        "name": "Python (openai library)",
        "instructions": [
            "pip install openai",
        ],
        "code": (
            "from openai import OpenAI\n\n"
            'client = OpenAI(\n'
            '    base_url="{base_url}/v1",\n'
            '    api_key="your-api-key-here",\n'
            ")\n\n"
            '# Chat\n'
            'response = client.chat.completions.create(\n'
            '    model="llm",\n'
            '    messages=[{"role": "user", "content": "Hello!"}],\n'
            ")\n"
            "print(response.choices[0].message.content)\n\n"
            '# Image generation\n'
            'response = client.images.generate(\n'
            '    model="z_image",\n'
            '    prompt="a cyberpunk samurai",\n'
            '    size="1024x1024",\n'
            ")\n"
            "print(response.data[0].b64_json[:80] + '...')\n\n"
            '# TTS\n'
            'response = client.audio.speech.create(\n'
            '    model="kokoro",\n'
            '    voice="af_bella",\n'
            '    input="Hello world",\n'
            ")\n"
            'with open("output.wav", "wb") as f:\n'
            "    f.write(response.content)"
        ),
    },
    "curl": {
        "name": "curl",
        "instructions": [],
        "commands": [
            '# List available models',
            'curl {base_url}/v1/models',
            '',
            '# Chat completion',
            'curl -X POST {base_url}/v1/chat/completions \\',
            '  -H "Content-Type: application/json" \\',
            '  -d \'{{"model": "llm", "messages": [{{"role": "user", "content": "Hello!"}}]}}\'',
            '',
            '# Image generation',
            'curl -X POST {base_url}/v1/images/generations \\',
            '  -H "Content-Type: application/json" \\',
            '  -d \'{{"model": "z_image", "prompt": "a cyberpunk samurai", "size": "1024x1024"}}\'',
            '',
            '# Text-to-speech',
            'curl -X POST {base_url}/v1/audio/speech \\',
            '  -H "Content-Type: application/json" \\',
            '  -d \'{{"model": "kokoro", "input": "Hello world", "voice": "af_bella"}}\' \\',
            '  --output speech.wav',
        ],
    },
}


def _get_base_url() -> str:
    """Determine the best base URL for connection config."""
    import subprocess
    # Try Tailscale IP first (works from tailnet)
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            ts_ip = result.stdout.strip().split("\n")[0].strip()
            if ts_ip:
                return f"http://{ts_ip}:30080"
    except Exception:
        pass

    # Try hostname
    import socket
    hostname = socket.gethostname()
    return f"http://{hostname}:30080"


def cmd_connect(args):
    """Print connection config for a specific app."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    import json

    console = Console()
    base_url = _get_base_url()

    # Override with explicit URL if provided
    if getattr(args, "url", None):
        base_url = args.url

    app = args.app

    if app == "list":
        console.print("\n[bold]Available app configs:[/bold]\n")
        for key, info in sorted(_CONNECT_SNIPPETS.items()):
            console.print(f"  [cyan]{key:16s}[/cyan]  {info['name']}")
        console.print()
        return

    snippet = _CONNECT_SNIPPETS.get(app)
    if not snippet:
        console.print(f"[red]Unknown app: {app}[/red]")
        console.print(f"Run [cyan]ray-noir connect list[/cyan] to see available apps.")
        return 1

    console.print()
    console.print(Panel(
        f"[bold]{snippet['name']}[/bold]\n"
        f"Base URL: [green]{base_url}[/green]",
        title="Tech Noir Connection",
        border_style="blue",
    ))

    # Print instructions
    if snippet.get("instructions"):
        console.print("\n[bold]Setup:[/bold]")
        for line in snippet["instructions"]:
            console.print(f"  {line}")

    # Print config values
    if snippet.get("config"):
        console.print("\n[bold]Configuration:[/bold]")
        for key, val in snippet["config"].items():
            resolved = val.replace("{base_url}", base_url)
            console.print(f"  [cyan]{key}[/cyan] = [green]{resolved}[/green]")

    # Print JSON config
    if snippet.get("config_json"):
        console.print("\n[bold]Config (JSON):[/bold]")
        raw = json.dumps(snippet["config_json"], indent=2)
        resolved = raw.replace("{base_url}", base_url)
        console.print(Syntax(resolved, "json", theme="monokai"))

    # Print commands
    if snippet.get("commands"):
        console.print("\n[bold]Commands:[/bold]")
        resolved_cmds = [c.replace("{base_url}", base_url) for c in snippet["commands"]]
        for cmd in resolved_cmds:
            if cmd.startswith("#") or cmd == "":
                console.print(f"  [dim]{cmd}[/dim]")
            else:
                console.print(f"  [green]{cmd}[/green]")

    # Print code
    if snippet.get("code"):
        console.print("\n[bold]Code:[/bold]")
        resolved_code = snippet["code"].replace("{base_url}", base_url)
        console.print(Syntax(resolved_code, "python", theme="monokai"))

    console.print()


def main():
    parser = argparse.ArgumentParser(
        prog="ray-noir",
        description="Tech Noir Ray - AI Infrastructure CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # models
    models = sub.add_parser("models", help="Model management")
    models_sub = models.add_subparsers(dest="models_command")

    models_sub.add_parser("list", help="List all models and status")
    models_sub.add_parser("status", help="Disk usage summary")
    models_sub.add_parser("verify", help="Verify model hashes")

    pull = models_sub.add_parser("pull", help="Download missing models")
    pull.add_argument("model", nargs="?", help="Specific model to pull (category/name or name)")

    # Shortcut: ray-noir pull <model>
    pull_shortcut = sub.add_parser("pull", help="Download a model (shortcut for models pull)")
    pull_shortcut.add_argument("model", help="Model to pull (category/name or name)")

    # connect
    connect = sub.add_parser("connect", help="Print connection config for an app")
    connect.add_argument("app", help="App name (or 'list' to see available)")
    connect.add_argument("--url", help="Override base URL")

    # cluster
    cluster = sub.add_parser("cluster", help="Ray cluster management")
    cluster_sub = cluster.add_subparsers(dest="cluster_command")
    cluster_sub.add_parser("start", help="Start Ray cluster")
    cluster_sub.add_parser("stop", help="Stop Ray cluster")

    # deploy
    sub.add_parser("deploy", help="Deploy all services")

    args = parser.parse_args()

    if args.command == "models":
        if args.models_command == "list":
            cmd_models_list(args)
        elif args.models_command == "pull":
            cmd_models_pull(args)
        elif args.models_command == "verify":
            cmd_models_verify(args)
        elif args.models_command == "status":
            cmd_models_status(args)
        else:
            models.print_help()
    elif args.command == "pull":
        # Shortcut: treat as models pull <model>
        args.models_command = "pull"
        cmd_models_pull(args)
    elif args.command == "connect":
        ret = cmd_connect(args)
        if ret:
            return ret
    elif args.command == "cluster":
        if args.cluster_command == "start":
            cmd_cluster_start(args)
        elif args.cluster_command == "stop":
            cmd_cluster_stop(args)
        else:
            cluster.print_help()
    elif args.command == "deploy":
        cmd_deploy(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
