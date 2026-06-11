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

    # Set up extra_model_paths.yaml
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
        "wan2gp_models": {
            "base_path": "/opt/wan2gp/ckpts",
            "checkpoints": "checkpoints",
            "vae": "vae",
            "loras": "loras",
            "upscale_models": "latent_upscale_models",
            "controlnet": "controlnet",
            "clip": "Qwen3-0.6B",
            "clip_vision": "clip_vision",
            "unet": "unet",
            "diffusion_models": "diffusion_models",
            "text_encoders": "Qwen3-0.6B",
        }
    }
    with open("/opt/ComfyUI/extra_model_paths.yaml", "w") as f:
        yaml.dump(config, f)
    print("extra_model_paths.yaml written")

    # Create custom model type symlinks (RMBG, sams, ultralytics for VNCCS)
    models_dir = "/opt/ComfyUI/models"
    os.makedirs(models_dir, exist_ok=True)
    for folder in ("RMBG", "sams", "ultralytics", "HY-Motion"):
        shared = os.path.join("/models/image-gen/comfyui", folder)
        link = os.path.join(models_dir, folder)
        if os.path.isdir(shared) and not os.path.exists(link):
            os.symlink(shared, link)
            print(f"Symlinked {link} -> {shared}")

    # Create symlinks for wan2gp models in ComfyUI
    wan2gp_ckpts = "/opt/wan2gp/ckpts"
    if os.path.isdir(wan2gp_ckpts):
        # Link anima model to unet directory (for diffusion models)
        anima_src = os.path.join(wan2gp_ckpts, "anima-base-v1.0.safetensors")
        anima_dst = os.path.join(models_dir, "unet", "anima-base-v1.0.safetensors")
        if os.path.isfile(anima_src) and not os.path.exists(anima_dst):
            os.makedirs(os.path.dirname(anima_dst), exist_ok=True)
            os.symlink(anima_src, anima_dst)
            print(f"Symlinked {anima_dst} -> {anima_src}")

        # Link text encoders for anima
        qwen_dir = os.path.join(wan2gp_ckpts, "Qwen3-0.6B")
        if os.path.isdir(qwen_dir):
            text_encoder_src = os.path.join(qwen_dir, "qwen_3_06b_base.safetensors")
            text_encoder_dst = os.path.join(models_dir, "text_encoders", "qwen_3_06b_base.safetensors")
            if os.path.isfile(text_encoder_src) and not os.path.exists(text_encoder_dst):
                os.makedirs(os.path.dirname(text_encoder_dst), exist_ok=True)
                os.symlink(text_encoder_src, text_encoder_dst)
                print(f"Symlinked {text_encoder_dst} -> {text_encoder_src}")

            # Also link to clip directory (ComfyUI checks both locations)
            clip_dst = os.path.join(models_dir, "clip", "qwen_3_06b_base.safetensors")
            if os.path.isfile(text_encoder_src) and not os.path.exists(clip_dst):
                os.makedirs(os.path.dirname(clip_dst), exist_ok=True)
                os.symlink(text_encoder_src, clip_dst)
                print(f"Symlinked {clip_dst} -> {text_encoder_src}")

    print("Setup complete")


if __name__ == "__main__":
    main()
