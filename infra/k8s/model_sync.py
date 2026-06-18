"""model_sync.py — KubeRay init container for model verification/download.

Checks if required models exist at MODELS_ROOT. Downloads missing models
from HuggingFace / ModelScope using the project's model_registry.yaml.

Locally (PVC): models already present → instant check, no downloads.
Cloud burst (empty disk): downloads only the models this worker needs.

Env vars:
  MODELS             Comma-separated registry keys (e.g. "3d.trellis,3d.trellis_dinov3")
  MODELS_CATEGORIES  Comma-separated categories (e.g. "tts,comfyui,3d") — expands to all keys
  MODELS_CATEGORY    Legacy: single category (e.g. "comfyui")
  MODELS_ROOT        Root directory for model storage (default: /models)
  HF_TOKEN           HuggingFace API token for gated models
"""
import os
import sys
import yaml
from pathlib import Path


def load_registry() -> dict:
    for p in [Path("/app/config/model_registry.yaml"),
              Path(__file__).parent.parent / "config" / "model_registry.yaml"]:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    sys.exit("ERROR: model_registry.yaml not found")


def find_entry(registry: dict, key: str) -> dict | None:
    cat, _, name = key.partition(".")
    return registry.get(cat, {}).get(name)


def model_exists(path: Path, download_type: str) -> bool:
    if download_type in ("skip", "manual"):
        return True
    if download_type == "snapshot":
        return path.is_dir() and any(path.iterdir())
    return path.exists()


def parse_hf(source: str, download_type: str) -> tuple[str, str | None]:
    rest = source.removeprefix("hf://")
    if download_type == "snapshot":
        return rest, None
    parts = rest.split("/")
    repo_id = f"{parts[0]}/{parts[1]}"
    filename = "/".join(parts[2:])
    return repo_id, filename


def download_model(key: str, meta: dict, models_root: str):
    source = meta.get("source", "")
    dl_type = meta.get("download", "skip")
    dest = Path(models_root) / meta["path"]

    if dl_type == "skip" or not source:
        print(f"  SKIP {key}: no download source")
        return

    if source.startswith("hf://"):
        from huggingface_hub import hf_hub_download, snapshot_download
        repo_id, filename = parse_hf(source, dl_type)
        if dl_type == "file":
            dest.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id, filename=filename,
                local_dir=str(dest.parent),
                local_dir_use_symlinks=False,
            )
            print(f"  DOWNLOADED {key} → {dest}")
        elif dl_type == "snapshot":
            snapshot_download(repo_id=repo_id, local_dir=str(dest))
            print(f"  DOWNLOADED {key} → {dest}/")

    elif source.startswith("modelscope://"):
        from modelscope import snapshot_download as ms_download
        repo_id = source.removeprefix("modelscope://")
        ms_download(repo_id, local_dir=str(dest))
        print(f"  DOWNLOADED {key} (ModelScope) → {dest}/")

    else:
        print(f"  WARN {key}: unsupported source '{source}'")


def resolve_models(registry: dict) -> list[str]:
    explicit = [k.strip() for k in os.environ.get("MODELS", "").split(",") if k.strip()]
    categories = [c.strip() for c in os.environ.get("MODELS_CATEGORIES", "").split(",") if c.strip()]
    if categories:
        for category in categories:
            if category in registry:
                cat_keys = [f"{category}.{k}" for k in registry[category]
                            if registry[category][k].get("download") not in ("skip", "manual")]
                explicit.extend(cat_keys)
    # Legacy single-category support
    category = os.environ.get("MODELS_CATEGORY", "")
    if category and category in registry:
        cat_keys = [f"{category}.{k}" for k in registry[category]
                    if registry[category][k].get("download") not in ("skip", "manual")]
        explicit.extend(cat_keys)
    return list(dict.fromkeys(explicit))


def main():
    models_root = os.environ.get("MODELS_ROOT", "/models")
    registry = load_registry()
    required = resolve_models(registry)

    if not required:
        print("No models required. Exiting.")
        return

    registry = load_registry()

    print(f"Checking {len(required)} models in {models_root}...")
    missing = []
    for key in required:
        meta = find_entry(registry, key)
        if not meta:
            print(f"  UNKNOWN: {key}")
            continue
        path = Path(models_root) / meta["path"]
        if model_exists(path, meta.get("download", "skip")):
            print(f"  OK: {key}")
        else:
            print(f"  MISSING: {key}")
            missing.append((key, meta))

    if not missing:
        print(f"All {len(required)} models present.")
        return

    print(f"\nDownloading {len(missing)} missing models...")
    for key, meta in missing:
        try:
            download_model(key, meta, models_root)
        except Exception as e:
            print(f"  ERROR {key}: {e}", file=sys.stderr)
            print(f"  WARN: {key} download failed — worker will start without it")
    # Don't exit on download errors — transient network/DNS failures should
    # not block the worker from starting. Missing models are handled at
    # runtime by the individual services.
    if missing:
        failed = len(missing)
        print(f"WARNING: {failed} model(s) failed to download — "
              f"services may be degraded")

    print("Model sync complete.")


if __name__ == "__main__":
    main()
