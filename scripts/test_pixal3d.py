"""Pixal3D real inference test — loads model on GPU and generates a GLB.

Usage: uv run python scripts/test_pixal3d.py
"""
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "opt" / "wan2gp"))

TEST_IMAGE = Path(__file__).resolve().parent.parent / "vendor" / "pixal3d" / "assets" / "images" / "0_img.png"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    if not TEST_IMAGE.exists():
        print(f"Test image not found: {TEST_IMAGE}")
        sys.exit(1)

    print(f"Test image: {TEST_IMAGE}")
    image_bytes = TEST_IMAGE.read_bytes()
    print(f"Image size: {len(image_bytes) / 1024:.0f} KB")

    # Step 1: Load handler
    print("\n[1/3] Loading handler...")
    from models.pixal3d.pixal3d_handler import family_handler

    print(f"  Supported types: {family_handler.query_supported_types()}")
    print(f"  Family: {family_handler.query_model_family()}")
    print(f"  Infos: {family_handler.query_family_infos()}")

    # Step 2: Load model
    print("\n[2/3] Loading model (this downloads weights if needed)...")
    t0 = time.time()
    pipeline, result = family_handler.load_model(
        model_filename="pixal3d",
        model_type="pixal3d",
        base_model_type="pixal3d",
        model_def={},
    )
    load_time = time.time() - t0
    pipe = result["pipe"]
    co_tenants = result["coTenantsMap"]

    print(f"  Load time: {load_time:.1f}s")
    print(f"  Pipe modules ({len(pipe)}):")
    for name, module in pipe.items():
        params = sum(p.numel() for p in module.parameters()) if hasattr(module, 'parameters') else 0
        size_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / 1e6 if hasattr(module, 'parameters') else 0
        print(f"    {name}: {params:,} params ({size_mb:.1f} MB)")
    print(f"  Co-tenants: {co_tenants}")

    # Step 3: Generate
    print("\n[3/3] Running inference...")
    t0 = time.time()
    output = pipeline.generate(
        image=image_bytes,
        seed=42,
        steps=12,
        guidance=7.5,
        resolution="1024_cascade",
        camera_angle_x=0.8575,
        camera_distance=2.0,
        mesh_scale=1.0,
    )
    infer_time = time.time() - t0

    print(f"  Status: {output.get('status')}")
    print(f"  Inference time: {infer_time:.1f}s")

    if output.get("media_type"):
        print(f"  Media type: {output['media_type']}")

    if output.get("data"):
        data = base64.b64decode(output["data"])
        out_path = OUTPUT_DIR / "pixal3d_test.glb"
        out_path.write_bytes(data)
        print(f"  Output size: {len(data) / 1024:.0f} KB")
        print(f"  Saved to: {out_path}")
    else:
        print(f"  ERROR: No data in output")
        print(f"  Full output: {output}")
        sys.exit(1)

    print(f"\nTotal time: {load_time + infer_time:.1f}s")
    print("SUCCESS")


if __name__ == "__main__":
    main()
