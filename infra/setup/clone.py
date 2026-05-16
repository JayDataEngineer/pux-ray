"""Clone or update creative tool repos and llama.cpp.

Uses git directly for host-side clones. Idempotent — skips repos
that are already cloned.

Docker-based tools (TRELLIS, AniGen, VibeVoice) are NOT cloned here —
their source is included in the Docker image during build.

ComfyUI custom extensions are managed by ComfyUIExtensionManager (Ray-native)
via config/comfyui_extensions.yaml — NOT cloned here.

Usage:
    python -m infra.setup.clone          # Clone all repos
    python -m infra.setup.clone ace-step # Clone ACE-Step only
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

INFRA_DIR = Path(__file__).resolve().parent.parent
REPOS_DIR = INFRA_DIR / "repos"

# Bare-metal tools only. Docker tools clone inside their Dockerfile.
REPOS = {
    "ace-step": ("https://github.com/ace-step/ACE-Step-1.5.git", "ACE-Step-1.5"),
    "qwen": ("https://github.com/JayDataEngineer/QwenTTS-Lora-Trainer.git", "qwen_img_expert"),
    "gpt-sovits": ("https://github.com/RVC-Boss/GPT-SoVITS.git", "GPT-SoVITS"),
    "comfyui": ("https://github.com/comfyanonymous/ComfyUI.git", "ComfyUI"),
    "llama": ("https://github.com/ggml-org/llama.cpp.git", "llama.cpp"),
}


def _log(msg: str) -> None:
    print(f"\033[0;32m[clone]\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"\033[1;33m[clone]\033[0m {msg}")


def _setup_comfyui_models(models_root: str | None = None) -> None:
    """Set up ComfyUI model paths: extra_model_paths.yaml + symlinks for custom types.

    extra_model_paths.yaml handles standard types (checkpoints, vae, loras, etc.).
    Extensions that use add_model_folder_path() for custom types (RMBG, sams) need
    symlinks from ComfyUI/models/<type> → shared models dir.
    """
    import yaml

    comfyui_dir = REPOS_DIR / "ComfyUI"
    config_path = comfyui_dir / "extra_model_paths.yaml"

    if models_root is None:
        from registry.config import Config
        models_root = Config().models_root + "/image-gen/comfyui"

    config = {
        "ray_models": {
            "base_path": models_root,
            "checkpoints": "checkpoints",
            "vae": "vae",
            "loras": "loras",
            "upscale_models": "latent_upscale_models",
            "controlnet": "controlnet",
            "clip": "clip",
            "clip_vision": "clip_vision",
            "unet": "unet",
            "diffusion_models": "diffusion_models",
            "text_encoders": "text_encoders",
        }
    }

    config_path.write_text(yaml.dump(config, default_flow_style=False))
    _log(f"ComfyUI extra_model_paths.yaml -> {models_root}")

    # Symlink custom model types from shared dir → ComfyUI/models/ so
    # extensions using add_model_folder_path() find pre-downloaded files.
    custom_types = {
        "RMBG": "Background removal (VNCCS sheet_manager)",
        "sams": "SAM segment-anything (VNCCS, controlnet_aux)",
        "ultralytics": "YOLO detectors (VNCCS QwenDetailer)",
        "HY-Motion": "HY-Motion text-to-3D human motion (ComfyUI-HY-Motion1)",
    }
    for folder, description in custom_types.items():
        shared_path = Path(models_root) / folder
        link_path = comfyui_dir / "models" / folder
        (comfyui_dir / "models").mkdir(parents=True, exist_ok=True)
        if shared_path.exists() and not link_path.exists():
            _log(f"Symlink ComfyUI/models/{folder} -> shared {folder} ({description})")
            link_path.symlink_to(shared_path)


def clone_repo(name: str, url: str, dest: str) -> bool:
    target = REPOS_DIR / dest
    if (target / ".git").is_dir():
        _log(f"{dest} already cloned, pulling latest...")
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(target), capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            _warn(f"  git pull failed for {dest}: {result.stderr[:100]}")
        return True

    _log(f"Cloning {dest}...")
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        _warn(f"  git clone failed: {result.stderr[:200]}")
        return False
    return True


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        _log("Cloning bare-metal tool repos...")
        for name, (url, dest) in REPOS.items():
            try:
                clone_repo(name, url, dest)
            except Exception as e:
                _warn(f"  {name} failed: {e}")

        _log("Setting up ComfyUI model paths...")
        _setup_comfyui_models()

        _log("All repos synced. ComfyUI extensions managed by ComfyUIExtensionManager (Ray-native).")
        return

    repo = REPOS.get(target)
    if repo:
        clone_repo(target, repo[0], repo[1])
    else:
        print(f"Usage: python -m infra.setup.clone [{'|'.join(REPOS)}|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
