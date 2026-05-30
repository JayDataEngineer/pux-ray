"""Pre-download ALL default Wan2GP models without loading into GPU.

Runs inside the Wan2GP Docker container (K3s pod) where all
dependencies (mmgp, torch, etc.) are available.

Downloads go to /models/wan2gp/ on the PVC (RW mount).

Usage:
    kubectl cp scripts/download_wan2gp_defaults.py <pod>:/tmp/
    kubectl exec <pod> -- python3 /tmp/download_wan2gp_defaults.py
"""

import json
import os
import sys
import time
from pathlib import Path

WAN2GP = Path(os.environ.get("WAN2GP_ROOT", "/opt/wan2gp"))
if str(WAN2GP) not in sys.path:
    sys.path.insert(0, str(WAN2GP))

# wgp.py reads models/_settings.json at import time via CWD.
os.chdir(str(WAN2GP))

DEFAULTS_DIR = WAN2GP / "defaults"
DONE_MARKER = Path("/tmp/wan2gp_download_done")
PVC_MODELS = Path("/models")
PVC_WAN2GP = PVC_MODELS / "wan2gp"


def _is_mountpoint(p: Path) -> bool:
    """Check if path is an actual mountpoint (not a directory on overlay)."""
    try:
        return os.path.ismount(str(p))
    except OSError:
        return False


def check_env():
    issues = []
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        issues.append("HF_TOKEN not set — gated models (Llama, Qwen, etc.) will fail")
    else:
        print(f"  HF_TOKEN: {'SET (%s...%s)' % (hf_token[:4], hf_token[-4:])}")

    # HARD FAIL: /models MUST be a real mountpoint (PVC), NOT overlay filesystem.
    # Writing to overlay floods the primary disk and causes DiskPressure eviction.
    if not _is_mountpoint(PVC_MODELS):
        issues.append(
            f"FATAL: {PVC_MODELS} is NOT a mountpoint — downloads would write to "
            "container overlay and flood the primary disk. "
            "Fix the K8s volume mount and retry."
        )
        print(f"  PVC: {PVC_MODELS} (NOT A MOUNTPOINT — ABORTING)")
    else:
        # Check PVC is writable
        test_file = PVC_WAN2GP / ".write_test"
        try:
            PVC_WAN2GP.mkdir(parents=True, exist_ok=True)
            test_file.write_text("ok")
            test_file.unlink()
            print(f"  PVC: {PVC_WAN2GP} (MOUNTED + WRITABLE)")
        except OSError as e:
            issues.append(f"PVC not writable: {e}")
            print(f"  PVC: {PVC_WAN2GP} (MOUNTED but READ-ONLY — {e})")

    free_gb = "?"
    try:
        st = os.statvfs(str(PVC_MODELS))
        free_gb = f"{st.f_frsize * st.f_bavail / (1024**3):.0f}GB"
    except OSError:
        pass
    print(f"  PVC free: {free_gb}")

    # Redirect HF cache to PVC (not overlay). This prevents snapshot_download
    # from writing gigabytes of duplicate cache to /tmp/huggingface on overlay.
    hf_cache = PVC_MODELS / ".hf_cache"
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
    print(f"  HF cache: {hf_cache} (on PVC)")

    if issues:
        print("\n  ISSUES:")
        for i in issues:
            print(f"    - {i}")
    return len(issues) == 0


def init_wan2gp():
    # Wan2GP imports torch.cuda at module level — mock it for GPU-less pods
    import torch.cuda
    if not torch.cuda.is_available():
        torch.cuda.get_device_capability = lambda *a, **kw: (9, 0)
        torch.cuda.get_device_properties = lambda *a, **kw: type("P", (), {"total_memory": 24 * 1024**3})()
        torch.cuda.device_count = lambda: 1
        torch.cuda.set_device = lambda *a: None
        torch.cuda.current_device = lambda: 0
        print("  (GPU not available — using CUDA mocks for import)")

    import wgp
    checkpoints = [str(PVC_WAN2GP), str(PVC_MODELS)]
    if not wgp.server_config:
        wgp.server_config = {
            "checkpoints_paths": checkpoints,
            "attention_mode": "auto",
            "transformer_quantization": "int8",
            "text_encoder_quantization": "int8",
            "save_path": "outputs",
            "profile": 2, "video_profile": 2, "image_profile": 2, "audio_profile": 2,
        }
    else:
        for cp in checkpoints:
            if cp not in wgp.server_config.get("checkpoints_paths", []):
                wgp.server_config["checkpoints_paths"].append(cp)
    from shared.utils import files_locator as fl
    fl.set_checkpoints_paths(wgp.server_config["checkpoints_paths"])
    if not wgp.model_types_handlers:
        wgp.refresh_model_defs()
        wgp.map_family_handlers()
    return wgp


def get_all_archs(wgp):
    archs = set()
    if not DEFAULTS_DIR.is_dir():
        return archs
    for f in sorted(DEFAULTS_DIR.iterdir()):
        if not f.name.endswith(".json") or f.name == "ReadMe.txt":
            continue
        try:
            d = json.loads(f.read_text())
            arch = d.get("model", {}).get("architecture", f.stem)
            if arch in wgp.models_def:
                archs.add(arch)
        except Exception:
            pass
    return sorted(archs)


def download_model(wgp, model_type):
    model_def = wgp.models_def.get(model_type)
    if not model_def:
        return "SKIP (not in models_def)"

    filename = wgp.get_model_filename(model_type=model_type)
    if not filename:
        return "SKIP (no filename)"

    # Check if already downloaded (on PVC)
    local = wgp.fl.get_local_model_filename(filename)

    if local and os.path.isfile(local):
        mb = os.path.getsize(local) / (1024 * 1024)
        return f"EXISTS ({mb:.0f}MB, {local})"

    # Download
    try:
        wgp.download_models(filename, model_type, 0, 1)
    except Exception as e:
        return f"ERR: {e}"

    # Verify
    local = wgp.fl.get_local_model_filename(filename)
    if local and os.path.isfile(local):
        mb = os.path.getsize(local) / (1024 * 1024)
        return f"DOWNLOADED ({mb:.0f}MB, {local})"

    return "DOWNLOADED (file not found after download — check logs)"


def _populate_hub_cache(cache_dir: Path, models: list[tuple[str, str, str]]) -> None:
    """Create HF hub cache entries from local_dir downloads.

    ``from_pretrained("org/model-name")`` looks for ``models--Org--Model-name/``
    in the HF hub cache. ``snapshot_download(local_dir=...)`` creates a plain
    directory, NOT the hub cache format. This function converts local_dir
    downloads into proper hub cache entries so models load offline.

    Idempotent — safe to run multiple times (skips existing blobs).
    """
    import hashlib
    import shutil

    if not cache_dir.is_dir():
        print("  SKIP: cache dir not found")
        return

    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    for repo_id, _marker, rel_path in models:
        # rel_path is relative to PVC_MODELS (e.g. "cache/huggingface/meta-llama/Meta-Llama-3-8B-Instruct")
        # The actual local_dir is PVC_MODELS / rel_path, which may be outside cache_dir.
        local_dir = Path(rel_path) if Path(rel_path).is_absolute() else None
        if local_dir is None:
            # Try under cache_dir first (e.g. cache/huggingface/LLM2Vec-...)
            local_dir = cache_dir / Path(rel_path).name
            # For nested paths (e.g. meta-llama/Meta-Llama-3-8B-Instruct under cache/huggingface/)
            if not local_dir.is_dir():
                local_dir = cache_dir / rel_path
        if not local_dir.is_dir():
            print(f"  SKIP {repo_id}: local dir not found at {local_dir}")
            continue

        parts = repo_id.split("/")
        hub_dir = cache_dir / f"models--{parts[0]}--{parts[1]}"
        blobs_dir = hub_dir / "blobs"
        snapshots_dir = hub_dir / "snapshots"
        refs_dir = hub_dir / "refs"

        blobs_dir.mkdir(parents=True, exist_ok=True)
        refs_dir.mkdir(parents=True, exist_ok=True)

        # Use a deterministic revision hash based on repo_id
        rev_hash = hashlib.sha256(repo_id.encode()).hexdigest()[:40]
        snap_dir = snapshots_dir / rev_hash
        snap_dir.mkdir(parents=True, exist_ok=True)

        n_files = 0
        for f in sorted(local_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(local_dir)
            file_hash = _sha256(f)
            blob_path = blobs_dir / file_hash

            if not blob_path.exists():
                shutil.copy2(f, blob_path)

            snap_file = snap_dir / rel
            snap_file.parent.mkdir(parents=True, exist_ok=True)
            if not snap_file.exists():
                snap_file.symlink_to(blob_path)
            n_files += 1

        # Write ref (NO trailing newline — scan_cache_dir treats it as part of the hash)
        (refs_dir / "main").write_text(rev_hash)

        # Remove .no_exist markers that block local loading
        no_exist_dir = hub_dir / ".no_exist"
        if no_exist_dir.is_dir():
            shutil.rmtree(no_exist_dir)

        print(f"  {repo_id:55s} → hub cache ({n_files} files)")


def main():
    print("=" * 70)
    print("  Wan2GP Default Models Pre-Downloader")
    print("=" * 70)

    # ── Step 1: Environment Check
    print("\n── Environment ──")
    if not check_env():
        print("\n  ABORTING: environment check failed (see above). "
              "Fix PVC mount before running this script.")
        sys.exit(1)

    # ── Step 2: Init Wan2GP
    print("\n── Initializing Wan2GP ──")
    t0 = time.time()
    try:
        wgp = init_wan2gp()
        print(f"  Done: {len(wgp.models_def)} model defs, "
              f"{len(wgp.model_types_handlers)} handlers ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Step 3: List model architectures
    archs = get_all_archs(wgp)
    print(f"\n── Models to download ({len(archs)} architectures) ──")
    for mt in archs:
        print(f"  {mt}")

    # ── Step 4: Download each model
    print(f"\n── Downloading ──")
    results = {}
    start = time.time()
    for i, mt in enumerate(archs, 1):
        print(f"  [{i:3d}/{len(archs)}] {mt:35s} ... ", end="", flush=True)
        t0 = time.time()
        r = download_model(wgp, mt)
        dt = time.time() - t0
        print(f"[{dt:5.1f}s] {r}")
        results[mt] = r

    elapsed = time.time() - start
    exists = sum(1 for r in results.values() if r.startswith("EXISTS"))
    dled = sum(1 for r in results.values() if r.startswith("DOWNLOADED"))
    failed = sum(1 for r in results.values() if r.startswith("ERR"))
    skipped = sum(1 for r in results.values() if r.startswith("SKIP"))

    # ── Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Already on disk:   {exists}")
    print(f"  Downloaded:        {dled}")
    print(f"  Failed:            {failed}")
    print(f"  Skipped:           {skipped}")
    print(f"  Total:             {len(results)}")
    print(f"  Time:              {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 70}")

    # Save results
    DONE_MARKER.write_text(json.dumps(results, indent=2))
    print(f"\n  Results -> {DONE_MARKER}")

    # ── Step 5: Download custom models (not in Wan2GP models_def)
    print(f"\n── Custom models ──")
    custom_models = {
        "moss_soundeffect_v2": {
            "repo": "OpenMOSS-Team/MOSS-SoundEffect-v2.0",
            "marker": "model_index.json",
        },
    }
    for name, info in custom_models.items():
        dest = PVC_WAN2GP / name
        print(f"  {name:35s} ... ", end="", flush=True)
        if (dest / info["marker"]).exists():
            print("EXISTS")
            continue
        t0 = time.time()
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=info["repo"],
                local_dir=str(dest),
                cache_dir=os.environ.get("HF_HUB_CACHE"),
            )
            dt = time.time() - t0
            print(f"DOWNLOADED ({dt:.0f}s)")
        except Exception as e:
            print(f"ERR: {e}")
            failed += 1

    # ── Step 6: Download Kimodo text encoder (LLM2Vec / Llama-3-8B)
    # Kimodo uses LLM2VecEncoder which needs:
    #   1. meta-llama/Meta-Llama-3-8B-Instruct (~16GB, gated — base model)
    #   2. McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp (~170MB, PEFT adapter)
    #   3. McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised (~170MB, PEFT adapter)
    # Requires HF_TOKEN with meta-llama license accepted on huggingface.co.
    print(f"\n── Kimodo text encoder (LLM2Vec / Llama-3-8B) ──")
    llm2vec_models = [
        ("meta-llama/Meta-Llama-3-8B-Instruct", "config.json", "cache/huggingface/meta-llama/Meta-Llama-3-8B-Instruct"),
        ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "config.json", "cache/huggingface/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"),
        ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "adapter_config.json", "cache/huggingface/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"),
    ]
    for repo_id, marker, rel_path in llm2vec_models:
        model_name = repo_id.split("/")[-1]
        dest = PVC_MODELS / rel_path
        print(f"  {model_name:55s} ... ", end="", flush=True)
        if (dest / marker).exists():
            print("EXISTS")
            continue
        t0 = time.time()
        try:
            from huggingface_hub import snapshot_download
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                cache_dir=os.environ.get("HF_HUB_CACHE"),
            )
            dt = time.time() - t0
            print(f"DOWNLOADED ({dt:.0f}s)")
        except Exception as e:
            print(f"ERR: {e}")
            failed += 1

    # ── Step 7: Populate HF hub cache from local_dir downloads
    # from_pretrained("org/model") checks hub cache (models--Org--Model/) not
    # plain local_dir directories. This step creates the hub cache symlinks so
    # models load offline. Without this, kimodo's LLM2VecEncoder fails with
    # "couldn't connect to huggingface.co" on air-gapped pods.
    print(f"\n── Populating HF hub cache ──")
    _populate_hub_cache(PVC_MODELS / "cache" / "huggingface", llm2vec_models)

    # ── Step 8: Lance shared components (ViT + VAE from bytedance-research/Lance)
    print(f"\n── Lance shared components ──")
    lance_dest = PVC_MODELS / "lance"
    lance_components = [
        ("Qwen2.5-VL-ViT/config.json", "ViT encoder"),
        ("Wan2.2_VAE.pth", "VAE"),
    ]
    for rel_path, label in lance_components:
        target = lance_dest / rel_path
        print(f"  {rel_path:45s} ({label}) ... ", end="", flush=True)
        if target.exists():
            print("EXISTS")
            continue
        t0 = time.time()
        try:
            from huggingface_hub import hf_hub_download
            target.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id="bytedance-research/Lance",
                filename=rel_path,
                local_dir=str(lance_dest),
                local_dir_use_symlinks=False,
                cache_dir=os.environ.get("HF_HUB_CACHE"),
            )
            dt = time.time() - t0
            print(f"DOWNLOADED ({dt:.0f}s)")
        except Exception as e:
            print(f"ERR: {e}")
            failed += 1

    # ── Step 9: Lance GGUF models (preferred over AWQ)
    print(f"\n── Lance GGUF models ──")
    lance_gguf = [
        ("Lance_3B_Video-Q5_K_M.gguf", "samuelchristlie/Lance-GGUF", 5.53),
        ("Lance_3B-Q5_K_M.gguf", "samuelchristlie/Lance-GGUF", 4.52),
    ]
    for filename, repo_id, size_gb in lance_gguf:
        target = lance_dest / filename
        print(f"  {filename:40s} ({size_gb}GB) ... ", end="", flush=True)
        if target.exists() and target.stat().st_size > 100_000:
            print("EXISTS")
            continue
        t0 = time.time()
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(lance_dest),
                local_dir_use_symlinks=False,
                cache_dir=os.environ.get("HF_HUB_CACHE"),
            )
            dt = time.time() - t0
            print(f"DOWNLOADED ({dt:.0f}s)")
        except Exception as e:
            print(f"ERR: {e}")
            failed += 1

    if failed:
        print(f"\n  FAILED MODELS:")
        for mt, r in results.items():
            if r.startswith("ERR"):
                print(f"    ❌ {mt}: {r}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
