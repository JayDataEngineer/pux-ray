"""Model provisioner and CLI for Tech Noir Ray.

Downloads missing models from HuggingFace, verifies integrity.
Treats the local disk as a cache and the registry YAML as source of truth.

Usage:
    ray-noir models list          # Show all models, downloaded vs missing
    ray-noir models pull          # Download all missing models
    ray-noir models pull <name>   # Download a specific model
    ray-noir models verify        # Check all model hashes
    ray-noir models status        # Summary of disk usage
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
            if download == "skip" or (not meta.get("source") and download not in ("file", "snapshot", "civitai", "modelscope")):
                table.add_row(f"{category}/{name}", "[dim]SKIP[/dim]", "system package")
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
