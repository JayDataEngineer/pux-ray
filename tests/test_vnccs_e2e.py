"""Real E2E test — calls the live GPU server, saves actual output images to disk.

Run: python tests/test_vnccs_e2e.py

Requires the API server running at http://tech-noir-ray-serve-svc.ai-services:8000
(or set FORGE_URL env var).
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import httpx

API_URL = os.environ.get("FORGE_URL", "http://tech-noir-ray-serve-svc.ai-services:8000").rstrip("/")
OUT_DIR = Path(__file__).parent / "e2e_output"

def invoke(payload: dict, timeout: float = 300.0) -> dict:
    r = httpx.post(f"{API_URL}/v1/run", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def save_image(b64: str, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_bytes(base64.b64decode(b64))
    print(f"  → saved {path} ({path.stat().st_size:,} bytes)")
    return path

def test_char_sheet():
    print("\n═══ char_sheet E2E ═══")
    result = invoke({
        "pipeline": "vnccs/char-sheet",
        "params": {
            "sex": "female",
            "age": 20,
            "eyes": "blue eyes",
            "hair": "black long",
            "race": "human",
            "background_color": "green",
            "quality": "turbo",
            "seed": 42,
        },
    }, timeout=600.0)

    print(f"  status: {result.get('status')}")
    if result.get("_pipeline"):
        print(f"  pipeline: {result['_pipeline']}")
    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        return

    if result.get("data"):
        path = save_image(result["data"], "char_sheet.png")
        # Verify it's a real image (not tiny/blank)
        from PIL import Image
        img = Image.open(path)
        print(f"  size: {img.size}, mode: {img.mode}")
        import numpy as np
        arr = np.array(img)
        white_pct = np.mean(np.all(arr == 255, axis=2))
        print(f"  white pixel %: {white_pct:.1%}")
        if white_pct > 0.99:
            print("  ⚠ IMAGE IS NEARLY ALL WHITE")
        if path.stat().st_size < 10000:
            print("  ⚠ FILE IS SUSPICIOUSLY SMALL")

    if result.get("face"):
        save_image(result["face"], "char_sheet_face.png")


def test_pose_edit_mesh():
    print("\n═══ pose_edit (capture mode) E2E ═══")
    # First generate a character to re-pose
    char_result = invoke({
        "pipeline": "vnccs/char-sheet",
        "params": {
            "sex": "female", "age": 20, "seed": 42,
            "eyes": "blue eyes", "hair": "black long",
            "quality": "turbo",
        },
    }, timeout=600.0)

    if char_result.get("error"):
        print(f"  char_sheet failed: {char_result['error']}")
        return

    char_b64 = char_result.get("data")
    if not char_b64:
        print("  no char data")
        return

    save_image(char_b64, "pose_edit_input.png")

    # Use capture mode with the char sheet as both character and pose reference
    # (tests the QWEN re-pose path without needing BodyMesh/anny)
    result = invoke({
        "pipeline": "vnccs/pose-edit",
        "params": {
            "character_image_b64": char_b64,
            "pose_image_b64": char_b64,  # Use same image as pose reference
            "seed": 42,
        },
    }, timeout=600.0)

    print(f"  status: {result.get('status')}")
    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        return

    if result.get("data"):
        path = save_image(result["data"], "pose_edit_mesh.png")
        from PIL import Image
        import numpy as np
        img = Image.open(path)
        print(f"  size: {img.size}, mode: {img.mode}")
        arr = np.array(img)
        white_pct = np.mean(np.all(arr == 255, axis=2))
        print(f"  white pixel %: {white_pct:.1%}")
        if white_pct > 0.99:
            print("  ⚠ IMAGE IS NEARLY ALL WHITE")


if __name__ == "__main__":
    test = sys.argv[1] if len(sys.argv) > 1 else "all"

    t0 = time.time()
    if test in ("all", "char_sheet"):
        test_char_sheet()
    if test in ("all", "pose_edit"):
        test_pose_edit_mesh()
    print(f"\nDone in {time.time()-t0:.1f}s — images in {OUT_DIR}")
