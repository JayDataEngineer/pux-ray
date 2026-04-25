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
import sys
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
            if source:
                source_display = source.split("/")[-1][:30]
            else:
                source_display = "local-only"

            status = "[green]OK[/green]" if exists else "[red]MISSING[/red]"
            table.add_row(name, category, size_str, status, source_display)

    console.print(table)


def cmd_models_pull(args):
    """Download missing models from HuggingFace."""
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

            if target and f"{category}/{name}" != target and name != target:
                continue

            source = meta.get("source")
            if not source:
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

                dest_dir = model_path.parent
                dest_dir.mkdir(parents=True, exist_ok=True)

                if filename:
                    # Single file download
                    downloaded = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(dest_dir),
                    )
                    print(f"       -> {downloaded}")
                else:
                    # Entire repo snapshot
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(model_path),
                    )
                    print(f"       -> {model_path}/")

                pulled += 1

            except Exception as e:
                print(f"  FAIL {category}/{name}: {e}")
                skipped += 1

    print(f"\nDone. Pulled: {pulled}, Skipped: {skipped}")


def cmd_models_verify(args):
    """Verify SHA256 hashes of downloaded models."""
    registry = _get_registry()
    ok = 0
    failed = 0
    no_hash = 0
    missing = 0

    for category in sorted(registry.data.keys()):
        models = registry.data[category]
        if not isinstance(models, dict):
            continue
        for name, meta in sorted(models.items()):
            if not isinstance(meta, dict):
                continue

            expected_hash = meta.get("sha256")
            if not expected_hash:
                no_hash += 1
                continue

            try:
                model_path = registry.get_path(category, name)
                if not model_path.exists() or not model_path.is_file():
                    print(f"  MISS {category}/{name} - file not found")
                    missing += 1
                    continue

                actual_hash = _compute_sha256(model_path)
                if actual_hash == expected_hash:
                    print(f"  OK   {category}/{name}")
                    ok += 1
                else:
                    print(f"  FAIL {category}/{name}")
                    print(f"       expected: {expected_hash}")
                    print(f"       actual:   {actual_hash}")
                    failed += 1
            except Exception as e:
                print(f"  ERR  {category}/{name}: {e}")
                failed += 1

    print(f"\nVerified: {ok}, Failed: {failed}, No hash: {no_hash}, Missing: {missing}")


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
