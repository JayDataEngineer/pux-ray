#!/usr/bin/env python3
"""
Tech Noir Universal Model Downloader — "one command to restore everything"

Reads model_registry.yaml (the single source of truth) and downloads every
auto-downloadable model to /mnt/data/models/. After a full wipe, this is the
only command needed to restore all models that can be automatically fetched.

Usage:
  python3 scripts/download/download_all.py                          # download everything
  python3 scripts/download/download_all.py --dry-run                # show what would download
  python3 scripts/download/download_all.py --section audio          # only audio models
  python3 scripts/download/download_all.py --list-models            # list all models with status
  python3 scripts/download/download_all.py --list-missing           # list only missing models
  python3 scripts/download/download_all.py --list-manual            # list models needing manual build
  python3 scripts/download/download_all.py --parallel 4             # download 4 files at once (default: 2)
  python3 scripts/download/download_all.py --hf-token <token>      # HF token for gated models

Design:
  - Source of truth is config/model_registry.yaml
  - Every physical entry with source + valid download method is auto-downloadable
  - Models with download: 'manual', 'skip', or no source are reported as requiring manual steps
  - Parallel downloads for speed, resume-download for reliability
  - Proper error handling: one failure doesn't stop the batch
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
REGISTRY_PATH = PROJECT_ROOT / "config" / "model_registry.yaml"
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/mnt/data/models"))

# Colors
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
NC = "\033[0m"

info = lambda msg: print(f"{GREEN}[INFO]{NC} {msg}")
warn = lambda msg: print(f"{YELLOW}[WARN]{NC} {msg}")
error = lambda msg: print(f"{RED}[ERROR]{NC} {msg}")
step = lambda msg: print(f"{BLUE}[STEP]{NC} {msg}")


# ─── Model discovery ─────────────────────────────────────────────────────────

def load_registry():
    """Load and return the model registry as a dict."""
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def discover_models(registry):
    """
    Discover all physical model entries from registry.
    Returns a list of dicts with: category, name, path, source, download, size_gb, status, full_path, on_disk
    """
    models = []
    for category, entries in registry.items():
        if category in ("_meta", "served-models"):
            continue
        if not isinstance(entries, dict):
            continue
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            path = entry.get("path", "")
            if not path or path == "None":
                continue

            full_path = MODELS_ROOT / path
            on_disk = full_path.exists()

            models.append({
                "category": category,
                "name": name,
                "path": path,
                "source": entry.get("source", ""),
                "download": entry.get("download", ""),
                "status": entry.get("status", ""),
                "size_gb": entry.get("size_gb", 0),
                "full_path": full_path,
                "on_disk": on_disk,
                "description": entry.get("description", ""),
            })
    return models


def classify_model(m):
    """
    Classify a model into: auto, manual, skip, gated, or nosource.
    Returns (class, reason) tuple.
    """
    source = m["source"]
    download = m["download"]
    
    if not source or source == "None":
        return ("nosource", "No source URL")
    if download in ("skip",):
        return ("skip", f"download={download}")
    if download in ("manual",):
        return ("manual", f"download={download}")
    if "gated" in source.lower():
        return ("gated", "Requires HF token with accepted terms")
    if source.startswith("hf://") or source.startswith("civitai://") or source.startswith("modelscope://"):
        return ("auto", "Can auto-download")
    return ("unknown", f"Unknown source format: {source}")


def parse_hf_source(source):
    """Parse hf://org/repo/file or hf://org/repo source URL."""
    path = source[5:]  # strip hf://
    parts = path.split("/")
    if len(parts) >= 3:
        # hf://org/repo/file
        repo = "/".join(parts[:2])
        file = "/".join(parts[2:])
        return repo, file
    elif len(parts) == 2:
        # hf://org/repo
        return path, None
    return None, None


def parse_civitai_source(source):
    """Parse civitai://model_id source URL."""
    return source[10:]  # strip civitai://


# ─── Download implementations ────────────────────────────────────────────────

def download_hf_snapshot(repo, target_dir, token=None):
    """Download full HF repo snapshot to target_dir."""
    cmd = ["hf", "download", repo, "--local-dir", str(target_dir)]
    if token:
        cmd.extend(["--token", token])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"HF download failed for {repo}: {result.stderr.strip()}")
    return True


def download_hf_file(repo, filename, target_dir, token=None):
    """Download single HF file to target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["hf", "download", "--quiet", repo,
           "--include", filename, "--local-dir", str(target_dir)]
    if token:
        cmd.extend(["--token", token])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"HF file download failed for {repo}/{filename}: {result.stderr.strip()}")
    return True


def download_model(m, token=None, dry_run=False):
    """Download a single model. Returns (success: bool, message: str)."""
    if dry_run:
        return (True, "[dry-run] WOULD DOWNLOAD")
    
    if m["on_disk"]:
        return (True, "Already on disk")
    
    source = m["source"]
    download = m["download"]
    target = m["full_path"]
    
    try:
        if source.startswith("hf://"):
            repo, file = parse_hf_source(source)
            if file and download == "file":
                # Single file download
                parent = target.parent
                download_hf_file(repo, file, parent, token=token)
                # Verify file landed
                expected = parent / file.split("/")[-1]
                if not expected.exists() and not target.exists():
                    # Try looking in subdirectory
                    for p in parent.rglob(file.split("/")[-1]):
                        if p.exists():
                            break
                    else:
                        warn(f"  Downloaded but file not found at expected location: {target}")
                        warn(f"  Checked in: {parent}")
                        return (True, "Downloaded (location uncertain)")
                return (True, f"Downloaded {file}")
            else:
                # Snapshot download to target dir
                target.mkdir(parents=True, exist_ok=True)
                download_hf_snapshot(repo, target, token=token)
                return (True, f"Downloaded snapshot to {target}")
        
        elif source.startswith("civitai://"):
            model_id = parse_civitai_source(source)
            return (False, f"CivitAI download not implemented for model ID {model_id}. "
                          f"Manual download from https://civitai.com/models/{model_id}")
        
        elif source.startswith("modelscope://"):
            ms_path = source[12:]  # strip modelscope://
            return (False, f"ModelScope download not yet implemented for {ms_path}. "
                          f"Manual: pip install modelscope && modelscope download {ms_path}")
        
        else:
            return (False, f"Unknown source type: {source}")
    
    except Exception as e:
        return (False, f"Download failed: {e}")


# ─── Reporting ───────────────────────────────────────────────────────────────

def print_model_table(models, show_status=True):
    """Print a formatted table of models."""
    print(f"\n{'Model':<50} {'Size':<8} {'Status':<12} {'On Disk':<8}")
    print("-" * 80)
    for m in sorted(models, key=lambda x: (x["category"], x["name"])):
        key = f"{m['category']}/{m['name']}"
        size = f"{m['size_gb']}G" if m["size_gb"] else "?"
        cls, reason = classify_model(m)
        status = cls if show_status else ""
        disk = "✓" if m["on_disk"] else "✗"
        print(f"{key:<50} {size:<8} {status:<12} {disk:<8}")
    print()


def print_manual_instructions(manual_models):
    """Print instructions for models that need manual builds."""
    print(f"\n{CYAN}═══ Manual Build Instructions ═══{NC}\n")
    for m in sorted(manual_models, key=lambda x: (x["category"], x["name"])):
        key = f"{m['category']}/{m['name']}"
        source = m["source"]
        path = m["path"]
        print(f"  {key}")
        print(f"    Path: {path}")
        print(f"    Source: {source}")
        
        # Specific instructions based on model
        if "qwen-image-edit" in key:
            print(f"    Build: python3 scripts/prepare_qwen_img_edit_fp8.py")
        elif "z-image" in key:
            print(f"    Build: python3 scripts/prepare_z_image_fp8.py")
        elif "vace" in key:
            print(f"    Build: python3 scripts/convert_vace_to_fp8.py")
        elif "ideogram" in key:
            print(f"    HF gated: hf.co/ideogram-ai/ideogram-4-nf4 (accept terms first)")
            print(f"    Download: huggingface-cli download ideogram-ai/ideogram-4-nf4 --local-dir {path}")
        elif "ltx-2.3-fp8" in key:
            print(f"    Already on disk via symlinks + transformer. Core files OK.")
        elif "lance" in key:
            print(f"    Clone manually: git clone https://huggingface.co/bytedance-research/Lance")
        elif "qwen3-tts-tokenizer" in key:
            print(f"    SKIPPED — qwen3-tts removed (MOSS VoiceGenerator + Kokoro replace)")
        print()


# ─── Main download orchestrator ──────────────────────────────────────────────

def download_all_auto(models, token=None, dry_run=False, parallel=2, section=None):
    """Download all auto-downloadable models, optionally filtered by section."""
    # Classify and filter
    auto_models = []
    for m in models:
        cls, reason = classify_model(m)
        if cls == "auto" and not m["on_disk"]:
            if section is None or m["category"] == section:
                auto_models.append(m)
    
    if not auto_models:
        info("No auto-downloadable models need downloading.")
        return 0
    
    if section:
        step(f"Downloading {len(auto_models)} models in section '{section}'...")
    else:
        step(f"Downloading {len(auto_models)} auto-downloadable models...")
    
    if dry_run:
        for m in auto_models:
            key = f"{m['category']}/{m['name']}"
            info(f"  WOULD DOWNLOAD: {key} -> {m['full_path']}")
        return 0
    
    # Download with parallel workers
    success = 0
    fail = 0
    
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(download_model, m, token=token): m for m in auto_models}
        for future in as_completed(futures):
            m = futures[future]
            key = f"{m['category']}/{m['name']}"
            ok, msg = future.result()
            if ok:
                info(f"  ✓ {key}: {msg}")
                success += 1
            else:
                error(f"  ✗ {key}: {msg}")
                fail += 1
    
    print()
    info(f"Downloaded: {success}, Failed: {fail}")
    return fail


def main():
    parser = argparse.ArgumentParser(
        description="Tech Noir Universal Model Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/download/download_all.py                    # download everything
  python3 scripts/download/download_all.py --dry-run           # show what would download
  python3 scripts/download/download_all.py --list-models       # list all models with status
  python3 scripts/download/download_all.py --list-missing      # list only missing auto-downloadable
  python3 scripts/download/download_all.py --list-manual       # list models needing manual setup
  python3 scripts/download/download_all.py --section audio     # only audio models
  python3 scripts/download/download_all.py --parallel 4        # parallel downloads
        """
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would download without executing")
    parser.add_argument("--section", "-s", help="Download only this section/category (e.g. audio, video)")
    parser.add_argument("--list-models", action="store_true", help="List all models with their status")
    parser.add_argument("--list-missing", action="store_true", help="List only missing auto-downloadable models")
    parser.add_argument("--list-manual", action="store_true", help="List models requiring manual setup")
    parser.add_argument("--parallel", "-p", type=int, default=2, help="Parallel downloads (default: 2)")
    parser.add_argument("--hf-token", help="HuggingFace token for gated models")
    parser.add_argument("--show-sections", action="store_true", help="List all sections/categories")
    
    args = parser.parse_args()
    
    # Load registry
    if not REGISTRY_PATH.exists():
        error(f"Registry not found: {REGISTRY_PATH}")
        sys.exit(1)
    
    registry = load_registry()
    models = discover_models(registry)
    
    # Classify all models
    for m in models:
        cls, reason = classify_model(m)
        m["_class"] = cls
        m["_reason"] = reason
    
    # ─── List modes ─────────────────────────────────────────────────────────
    if args.show_sections:
        categories = sorted(set(m["category"] for m in models))
        print(f"\nAvailable sections ({len(categories)}):")
        for cat in categories:
            count = sum(1 for m in models if m["category"] == cat)
            auto_count = sum(1 for m in models if m["category"] == cat and m["_class"] == "auto" and not m["on_disk"])
            disk_count = sum(1 for m in models if m["category"] == cat and m["on_disk"])
            print(f"  {cat:<20} {count:3d} entries, {disk_count:3d} on disk, {auto_count:3d} auto-downloadable missing")
        return 0
    
    if args.list_models:
        print(f"\n{CYAN}═══ Complete Model Inventory ({len(models)} entries) ═══{NC}")
        print_model_table(models)
        return 0
    
    if args.list_missing:
        missing = [m for m in models if m["_class"] == "auto" and not m["on_disk"]]
        print(f"\n{CYAN}═══ Missing Auto-Downloadable Models ({len(missing)}) ═══{NC}")
        if missing:
            print_model_table(missing)
        else:
            info("Nothing missing — all auto-downloadable models are on disk!")
        return 0
    
    if args.list_manual:
        manual = [m for m in models if m["_class"] == "manual"]
        gated = [m for m in models if m["_class"] == "gated"]
        print(f"\n{CYAN}═══ Models Requiring Manual Setup ═══{NC}")
        if manual:
            print(f"\n{GREEN}Manual build needed ({len(manual)}):{NC}")
            print_manual_instructions(manual)
        if gated:
            print(f"\n{YELLOW}Gated models ({len(gated)}):{NC}")
            for m in gated:
                key = f"{m['category']}/{m['name']}"
                print(f"  {key}: {m['source']}")
            print()
        return 0
    
    # ─── Summary before download ────────────────────────────────────────────
    auto_missing = [m for m in models if m["_class"] == "auto" and not m["on_disk"]]
    total_auto = [m for m in models if m["_class"] == "auto"]
    manual = [m for m in models if m["_class"] == "manual"]
    
    info(f"Registry: {REGISTRY_PATH}")
    info(f"Models root: {MODELS_ROOT}")
    info(f"Total entries: {len(models)}")
    info(f"Auto-downloadable: {len(total_auto)}")
    info(f"  On disk: {len(total_auto) - len(auto_missing)}")
    info(f"  Missing: {len(auto_missing)}")
    info(f"Manual build: {len(manual)}")
    
    if auto_missing:
        print()
        print(f"{YELLOW}Will download: {len(auto_missing)} models{NC}")
        for m in auto_missing:
            key = f"{m['category']}/{m['name']}"
            size = f"{m['size_gb']}G" if m["size_gb"] else "?"
            print(f"  {key:<50} {size:<8} -> {m['full_path']}")
    
    # ─── Download ────────────────────────────────────────────────────────────
    if args.dry_run:
        info("Dry run — no downloads executed")
        return 0
    
    if auto_missing:
        print()
        exit_code = download_all_auto(models, token=args.hf_token, parallel=args.parallel, section=args.section)
        if exit_code > 0:
            sys.exit(exit_code)
    else:
        info("Nothing to download!")
    
    # ─── Final report ────────────────────────────────────────────────────────
    print(f"\n{GREEN}═══ Download Complete ═══{NC}")
    
    # Check what's still missing
    for m in models:
        m["on_disk"] = m["full_path"].exists()  # refresh
        cls, reason = classify_model(m)
        m["_class"] = cls
    
    auto_missing = [m for m in models if m["_class"] == "auto" and not m["on_disk"]]
    manual = [m for m in models if m["_class"] == "manual"]
    gated = [m for m in models if m["_class"] == "gated"]
    
    if auto_missing:
        warn(f"{len(auto_missing)} auto-downloadable models still missing:")
        for m in auto_missing:
            print(f"  ✗ {m['category']}/{m['name']}")
    
    if manual:
        print(f"\n{YELLOW}Manual models ({len(manual)}):{NC}")
        for m in manual:
            print(f"  • {m['category']}/{m['name']} ({m['_reason']})")
        print(f"\n  Run with --list-manual for build instructions")
    
    if gated:
        print(f"\n{YELLOW}Gated models ({len(gated)}):{NC}")
        for m in gated:
            print(f"  • {m['category']}/{m['name']}")
    
    info("Done.")


if __name__ == "__main__":
    main()
