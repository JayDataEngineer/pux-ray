"""Clone or update creative tool repos and llama.cpp.

Uses git directly for host-side clones. Idempotent — skips repos
that are already cloned.

Docker-based tools (TRELLIS, AniGen, VibeVoice) are NOT cloned here —
their source is included in the Docker image during build.

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
    "see-through": ("https://github.com/shitagaki-lab/see-through.git", "see-through"),
    "qwen": ("https://github.com/JayDataEngineer/QwenTTS-Lora-Trainer.git", "qwen_img_expert"),
    "gpt-sovits": ("https://github.com/RVC-Boss/GPT-SoVITS.git", "GPT-SoVITS"),
    "comfyui": ("https://github.com/comfyanonymous/ComfyUI.git", "ComfyUI"),
    "llama": ("https://github.com/ggml-org/llama.cpp.git", "llama.cpp"),
}

# ComfyUI custom extensions — cloned into ComfyUI/custom_nodes/
# Each entry is (url, ref_or_None). ref pins to a specific commit/branch/tag.
COMFYUI_EXTENSIONS = {
    "pose-director": ("https://github.com/JayDataEngineer/comfyui-pose-director.git", None),
    "comfyui_controlnet_aux": ("https://github.com/Fannovel16/comfyui_controlnet_aux.git", None),
    "ComfyUI-GGUF": ("https://github.com/city96/ComfyUI-GGUF.git", None),
    "ComfyUI-LTXVideo": ("https://github.com/Lightricks/ComfyUI-LTXVideo.git", "4c5add5"),
    "vnccs": ("https://github.com/AHEKOT/ComfyUI_VNCCS.git", None),
    "vnccs_utils": ("https://github.com/AHEKOT/ComfyUI_VNCCS_Utils.git", None),
}


def _log(msg: str) -> None:
    print(f"\033[0;32m[clone]\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"\033[1;33m[clone]\033[0m {msg}")


def _setup_comfyui_models() -> None:
    """Set up ComfyUI model paths: extra_model_paths.yaml + symlinks for custom types.

    extra_model_paths.yaml handles standard types (checkpoints, vae, loras, etc.).
    Extensions that use add_model_folder_path() for custom types (RMBG, sams) need
    symlinks from ComfyUI/models/<type> → shared models dir.
    """
    import yaml

    comfyui_dir = REPOS_DIR / "ComfyUI"
    config_path = comfyui_dir / "extra_model_paths.yaml"

    models_root = "/home/user/Documents/models/image-gen/comfyui"

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
    }
    for folder, description in custom_types.items():
        shared_path = Path(models_root) / folder
        link_path = comfyui_dir / "models" / folder
        (comfyui_dir / "models").mkdir(parents=True, exist_ok=True)
        if shared_path.exists() and not link_path.exists():
            _log(f"Symlink ComfyUI/models/{folder} -> shared {folder} ({description})")
            link_path.symlink_to(shared_path)


def _setup_comfyui_deps() -> None:
    """Install required pip packages in ComfyUI's own venv.

    ComfyUI runs under its own Python venv (torch 2.10+cu130). Extensions
    depend on packages that must be installed there, not in the ray venv.
    """
    comfyui_dir = REPOS_DIR / "ComfyUI"
    venv_python = comfyui_dir / ".venv" / "bin" / "python"

    if not venv_python.exists():
        _warn(f"ComfyUI venv not found at {venv_python} — skipping deps")
        return

    required = [
        "opencv-python-headless",  # comfyui_controlnet_aux, vnccs, vnccs_utils
        "gguf",                     # ComfyUI-GGUF
        "matplotlib",              # comfyui_controlnet_aux DWPose
        "scikit-image",            # comfyui_controlnet_aux DWPose
        "ltx-video",               # ComfyUI-LTXVideo (pinned to 4c5add5)
    ]

    for pkg in required:
        result = subprocess.run(
            [str(venv_python), "-c", f"import {pkg.replace('-','_')}"],
            capture_output=True,
        )
        if result.returncode != 0:
            _log(f"Installing {pkg} in ComfyUI venv...")
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", pkg],
                check=False,
                capture_output=True,
            )

    # ltx-video pins transformers<5 but ComfyUI needs >=5.
    # Re-upgrade after ltx-video install.
    result = subprocess.run(
        [str(venv_python), "-c", "import transformers; print(transformers.__version__)"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip().startswith("4."):
        _log("Re-upgrading transformers to >=5 (downgraded by ltx-video)")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "transformers>=5"],
            check=False, capture_output=True,
        )


def clone_comfyui_extension(name: str, url: str, ref: str | None = None) -> bool:
    """Clone a single ComfyUI custom extension into custom_nodes/.
    If ref is set, checkout that commit/tag/branch after clone/pull.
    """
    comfyui_dir = REPOS_DIR / "ComfyUI"
    target = comfyui_dir / "custom_nodes" / name
    if (target / ".git").is_dir():
        if ref:
            # Pinned: fetch and checkout the exact ref, skip pull
            result = subprocess.run(
                ["git", "fetch", "origin"],
                cwd=str(target), capture_output=True, text=True, timeout=60,
            )
            current = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(target), capture_output=True, text=True,
            ).stdout.strip()[:7]
            if current != ref[:7]:
                _log(f"  Extension {name} pinning to {ref} ...")
                subprocess.run(
                    ["git", "checkout", "-f", ref],
                    cwd=str(target), capture_output=True,
                )
                subprocess.run(
                    ["git", "clean", "-fd"],
                    cwd=str(target), capture_output=True,
                )
        else:
            _log(f"  Extension {name} already cloned, pulling...")
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=str(target), capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                _warn(f"  git pull failed for {name}: {result.stderr[:100]}")
        return True
    _log(f"  Cloning extension {name}...")
    target.parent.mkdir(parents=True, exist_ok=True)
    clone_cmd = ["git", "clone"]
    if not ref:
        clone_cmd.append("--depth=1")
    clone_cmd.extend([url, str(target)])
    result = subprocess.run(
        clone_cmd,
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        _warn(f"  git clone failed for {name}: {result.stderr[:200]}")
        return False
    if ref:
        _log(f"  Checking out {name} at {ref} ...")
        subprocess.run(
            ["git", "checkout", ref],
            cwd=str(target), capture_output=True,
        )
    return True


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

        _log("Cloning ComfyUI custom extensions...")
        for name, (url, ref) in COMFYUI_EXTENSIONS.items():
            try:
                clone_comfyui_extension(name, url, ref)
            except Exception as e:
                _warn(f"  extension {name} failed: {e}")

        _log("Setting up ComfyUI model paths...")
        _setup_comfyui_models()
        _setup_comfyui_deps()

        _log("All repos synced.")
        return

    repo = REPOS.get(target)
    if repo:
        clone_repo(target, repo[0], repo[1])
    else:
        print(f"Usage: python -m infra.setup.clone [{'|'.join(REPOS)}|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
