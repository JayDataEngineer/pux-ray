"""ComfyUI Docker setup — clones extensions, installs pip deps, configures paths."""
import subprocess
import os
import sys

# Ensure core deps (ai-dock base may be missing some for newer ComfyUI)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "psutil"], check=False)

# ai-dock may not have yaml installed; install if missing
try:
    import yaml
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"], check=True)
    import yaml

CUSTOM_NODES = "/opt/ComfyUI/custom_nodes"
EXTENSIONS_YAML = "/tmp/extensions.yaml"


def main():
    os.makedirs(CUSTOM_NODES, exist_ok=True)

    with open(EXTENSIONS_YAML) as f:
        cfg = yaml.safe_load(f)

    for ext in cfg.get("extensions", []):
        name = ext["name"]
        url = ext["url"]
        ref = ext.get("ref")
        target = os.path.join(CUSTOM_NODES, name)

        if os.path.isdir(os.path.join(target, ".git")):
            print(f"  [{name}] already cloned, skipping")
            continue

        print(f"  [{name}] cloning...")
        cmd = ["git", "clone"]
        if not ref:
            cmd.append("--depth=1")
        cmd.extend([url, target])
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  [{name}] CLONE FAILED", file=sys.stderr)
            continue

        if ref:
            print(f"  [{name}] pinning to {ref}...")
            subprocess.run(["git", "checkout", ref], cwd=target, check=True)

        for pkg in ext.get("pip", []):
            print(f"  [{name}] pip install {pkg}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg],
                check=False,
            )

    # Fix transformers if ltx-video downgraded it
    result = subprocess.run(
        [sys.executable, "-c", "import transformers; print(transformers.__version__)"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip().startswith("4."):
        print("Re-upgrading transformers to >=5 (downgraded by ltx-video)")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "transformers>=5"],
            check=False,
        )

    # Set up extra_model_paths.yaml - DOCKER VOLUME MOUNTS ONLY, NO SYMLINKS
    config = {
        "ray_models": {
            "base_path": "/models/image-gen/comfyui",
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
        },
        "wan2gp_base": {
            "base_path": "/opt/wan2gp/ckpts",
            "unet": "",  # Anima models at base_path
            "diffusion_models": "",  # Anima models at base_path
            "text_encoders": "Qwen3-0.6B",  # Qwen text encoder in subdirectory
            "clip": "Qwen3-0.6B",  # Also add to clip search path
        }
    }
    with open("/opt/ComfyUI/extra_model_paths.yaml", "w") as f:
        yaml.dump(config, f)
    print("extra_model_paths.yaml written")

    print("Setup complete - volume mounts handle model paths")


if __name__ == "__main__":
    main()
