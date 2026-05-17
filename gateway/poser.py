"""Poser — OpenPose preset skeleton renderer.

Serves pre-defined pose presets as both metadata (JSON) and rendered
skeleton images (PNG). Uses the same COCO-18 bone/keypoint color scheme
as controlnet_aux for ControlNet compatibility.
"""

import io
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# ── COCO-18 OpenPose constants ────────────────────────────────────────────

SKELETON_BONES = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (0, 1), (0, 14), (14, 16), (0, 15), (15, 17),
]

BONE_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]

KEYPOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85),
]

# ── Preset poses (COCO-18 keypoints: x, y, confidence × 18) ──────────────

PRESETS = {
    "standing_neutral": {
        "name": "Standing Neutral",
        "description": "Arms at sides, weight even, facing forward",
        "tags": ["standing", "neutral", "default"],
        "keypoints": [
            512, 152, 0.9, 512, 192, 0.9, 412, 232, 0.9, 372, 352, 0.9,
            392, 472, 0.9, 612, 232, 0.9, 652, 352, 0.9, 632, 472, 0.9,
            452, 392, 0.9, 442, 572, 0.9, 432, 752, 0.9,
            572, 392, 0.9, 582, 572, 0.9, 592, 752, 0.9,
            502, 132, 0.9, 522, 132, 0.9, 482, 142, 0.9, 542, 142, 0.9,
        ],
    },
    "t_pose": {
        "name": "T-Pose",
        "description": "Arms extended horizontally — calibration stance",
        "tags": ["t-pose", "calibration", "arms out"],
        "keypoints": [
            512, 152, 0.9, 512, 192, 0.9, 412, 232, 0.9, 312, 232, 0.9,
            212, 232, 0.9, 612, 232, 0.9, 712, 232, 0.9, 812, 232, 0.9,
            452, 392, 0.9, 442, 572, 0.9, 432, 752, 0.9,
            572, 392, 0.9, 582, 572, 0.9, 592, 752, 0.9,
            502, 132, 0.9, 522, 132, 0.9, 482, 142, 0.9, 542, 142, 0.9,
        ],
    },
    "sitting": {
        "name": "Sitting",
        "description": "Seated, knees bent at 90°, hands on knees",
        "tags": ["sitting", "seated", "chair", "relaxed"],
        "keypoints": [
            512, 252, 0.9, 512, 292, 0.9, 412, 332, 0.9, 392, 432, 0.9,
            432, 512, 0.9, 612, 332, 0.9, 632, 432, 0.9, 592, 512, 0.9,
            452, 492, 0.9, 412, 592, 0.9, 462, 692, 0.9,
            572, 492, 0.9, 612, 592, 0.9, 562, 692, 0.9,
            502, 232, 0.9, 522, 232, 0.9, 482, 242, 0.9, 542, 242, 0.9,
        ],
    },
    "walking": {
        "name": "Walking",
        "description": "Mid-stride, right leg forward, natural gait",
        "tags": ["walking", "stride", "gait"],
        "keypoints": [
            512, 152, 0.9, 512, 192, 0.9, 412, 232, 0.9, 392, 332, 0.9,
            412, 432, 0.9, 612, 232, 0.9, 642, 332, 0.9, 622, 432, 0.9,
            462, 392, 0.9, 512, 542, 0.9, 562, 722, 0.9,
            562, 392, 0.9, 492, 572, 0.9, 442, 752, 0.9,
            502, 132, 0.9, 522, 132, 0.9, 482, 142, 0.9, 542, 142, 0.9,
        ],
    },
    "running": {
        "name": "Running",
        "description": "Full sprint, extended stride, arms pumping",
        "tags": ["running", "sprinting", "athletic"],
        "keypoints": [
            512, 142, 0.9, 522, 182, 0.9, 422, 212, 0.9, 482, 312, 0.9,
            542, 392, 0.9, 622, 212, 0.9, 562, 132, 0.9, 482, 82, 0.9,
            472, 382, 0.9, 562, 512, 0.9, 632, 682, 0.9,
            552, 382, 0.9, 442, 552, 0.9, 352, 722, 0.9,
            502, 122, 0.9, 522, 122, 0.9, 482, 132, 0.9, 542, 132, 0.9,
        ],
    },
    "dancing": {
        "name": "Dancing",
        "description": "Dynamic dance pose, one arm raised, weight shifted",
        "tags": ["dancing", "dynamic", "energetic", "performance"],
        "keypoints": [
            472, 152, 0.9, 482, 192, 0.9, 392, 222, 0.9, 342, 142, 0.9,
            312, 62, 0.9, 572, 222, 0.9, 622, 342, 0.9, 592, 442, 0.9,
            432, 372, 0.9, 412, 542, 0.9, 442, 732, 0.9,
            532, 372, 0.9, 592, 512, 0.9, 612, 712, 0.9,
            462, 132, 0.9, 482, 132, 0.9, 442, 142, 0.9, 502, 142, 0.9,
        ],
    },
    "waving": {
        "name": "Waving",
        "description": "Right arm raised, waving — friendly greeting",
        "tags": ["waving", "greeting", "hello", "friendly"],
        "keypoints": [
            512, 152, 0.9, 512, 192, 0.9, 412, 232, 0.9, 372, 162, 0.9,
            432, 82, 0.9, 612, 232, 0.9, 652, 352, 0.9, 632, 472, 0.9,
            452, 392, 0.9, 442, 572, 0.9, 432, 752, 0.9,
            572, 392, 0.9, 582, 572, 0.9, 592, 752, 0.9,
            502, 132, 0.9, 522, 132, 0.9, 482, 142, 0.9, 542, 142, 0.9,
        ],
    },
}

# ── Rendering ─────────────────────────────────────────────────────────────


def _render_skeleton(
    keypoints: list[float],
    width: int = 1024,
    height: int = 1024,
    line_width: int = 4,
    point_radius: int = 4,
) -> bytes:
    """Render OpenPose keypoints into a PNG skeleton image."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw bones with per-bone gradient colors
    for bone_idx, (i, j) in enumerate(SKELETON_BONES):
        if i * 3 + 2 >= len(keypoints) or j * 3 + 2 >= len(keypoints):
            continue
        c1 = keypoints[i * 3 + 2]
        c2 = keypoints[j * 3 + 2]
        if c1 <= 0.1 or c2 <= 0.1:
            continue
        x1, y1 = keypoints[i * 3], keypoints[i * 3 + 1]
        x2, y2 = keypoints[j * 3], keypoints[j * 3 + 1]
        draw.line([(x1, y1), (x2, y2)], fill=BONE_COLORS[bone_idx], width=line_width)

    # Draw keypoints with per-keypoint gradient colors
    for idx in range(min(len(keypoints) // 3, len(KEYPOINT_COLORS))):
        conf = keypoints[idx * 3 + 2]
        if conf <= 0.1:
            continue
        x, y = keypoints[idx * 3], keypoints[idx * 3 + 1]
        r = point_radius
        draw.ellipse([x - r, y - r, x + r, y + r], fill=KEYPOINT_COLORS[idx])

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Route handlers ────────────────────────────────────────────────────────

# Cache rendered PNGs (small — ~2KB each)
_render_cache: dict[str, bytes] = {}


async def poser_presets(request: Request) -> Response:
    """GET /poser/presets — list all available pose presets."""
    preset_list = []
    for key, data in PRESETS.items():
        preset_list.append({
            "id": key,
            "name": data["name"],
            "description": data["description"],
            "tags": data["tags"],
            "render_url": f"/poser/presets/{key}/render",
        })
    return JSONResponse(preset_list)


async def poser_preset_render(request: Request) -> Response:
    """GET /poser/presets/{name}/render — render a pose preset as PNG."""
    name = request.path_params["name"]
    if name not in PRESETS:
        return JSONResponse({"error": f"Unknown preset: {name}"}, status_code=404)

    # Query params for render customization
    w = int(request.query_params.get("width", 1024))
    h = int(request.query_params.get("height", 1024))
    lw = int(request.query_params.get("line_width", 4))
    pr = int(request.query_params.get("point_radius", 4))

    # Cache key includes render params
    cache_key = f"{name}:{w}:{h}:{lw}:{pr}"
    if cache_key not in _render_cache:
        _render_cache[cache_key] = _render_skeleton(
            PRESETS[name]["keypoints"],
            width=w, height=h, line_width=lw, point_radius=pr,
        )

    return Response(
        content=_render_cache[cache_key],
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )
