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


def check_env():
    issues = []
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        issues.append("HF_TOKEN not set — gated models (Llama, Qwen, etc.) will fail")
    else:
        print(f"  HF_TOKEN: {'SET (%s...%s)' % (hf_token[:4], hf_token[-4:])}")

    # Check PVC is writable
    test_file = PVC_WAN2GP / ".write_test"
    try:
        PVC_WAN2GP.mkdir(parents=True, exist_ok=True)
        test_file.write_text("ok")
        test_file.unlink()
        print(f"  PVC: {PVC_WAN2GP} (WRITABLE)")
    except OSError as e:
        issues.append(f"PVC not writable: {e}")
        print(f"  PVC: {PVC_WAN2GP} (READ-ONLY — {e})")

    free_gb = "?"
    try:
        st = os.statvfs(str(PVC_MODELS))
        free_gb = f"{st.f_frsize * st.f_bavail / (1024**3):.0f}GB"
    except OSError:
        pass
    print(f"  PVC free: {free_gb}")

    if issues:
        print("\n  ISSUES:")
        for i in issues:
            print(f"    - {i}")
    return len(issues) == 0


def init_wan2gp():
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


def main():
    print("=" * 70)
    print("  Wan2GP Default Models Pre-Downloader")
    print("=" * 70)

    # ── Step 1: Environment Check
    print("\n── Environment ──")
    check_env()

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

    if failed:
        print(f"\n  FAILED MODELS:")
        for mt, r in results.items():
            if r.startswith("ERR"):
                print(f"    ❌ {mt}: {r}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
