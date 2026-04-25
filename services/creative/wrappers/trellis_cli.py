#!/usr/bin/env python3
"""
TRELLIS.2 CLI Wrapper
=====================
Wraps TRELLIS.2 image-to-3D pipeline with a command-line interface.

IMPORTANT: Must be run with TRELLIS.2's own venv Python:
    ~/Documents/programs/TRELLIS.2/.venv/bin/python trellis_cli.py --image input.png --output output.glb

Usage:
    python trellis_cli.py --image input.png --output output.glb
    python trellis_cli.py --image input.png --output output.glb --texture-size 4096
"""
import argparse
import os
import sys
from pathlib import Path

TRELLIS_ROOT = Path(os.environ.get("TRELLIS_ROOT", os.getcwd()))
TRELLIS_PYTHON = TRELLIS_ROOT / ".venv" / "bin" / "python"

sys.path.insert(0, str(TRELLIS_ROOT))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 Image to 3D GLB exporter")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output .glb file")
    parser.add_argument("--resolution", type=int, default=512, help="Texture resolution (maps to texture_size)")
    parser.add_argument("--decimation", type=int, default=1_000_000, help="Target decimation face count")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B", help="HuggingFace model ID")
    parser.add_argument("--simplify", type=int, default=16_777_216, help="Simplify vertex limit")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Input image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Map resolution to texture size
    texture_size = {256: 2048, 512: 4096, 1024: 4096, 2048: 8192}.get(args.resolution, 4096)

    print(f"Loading TRELLIS.2 pipeline ({args.model})...")
    try:
        from PIL import Image
        import torch
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        import o_voxel
    except ImportError as e:
        print(f"Error: Missing TRELLIS.2 dependency: {e}", file=sys.stderr)
        print("Hint: Make sure TRELLIS.2 is installed with all dependencies:", file=sys.stderr)
        print("  cd ~/Documents/programs/TRELLIS.2 && bash setup.sh --basic --o-voxel", file=sys.stderr)
        sys.exit(1)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    print(f"Processing image: {args.image}")
    image = Image.open(args.image)
    mesh = pipeline.run(image)[0]
    mesh.simplify(args.simplify)

    print(f"Exporting GLB -> {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    glb.export(args.output, extension_webp=True)
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
