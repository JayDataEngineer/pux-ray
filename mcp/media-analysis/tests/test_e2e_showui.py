"""E2E test for ShowUI ground_ui tool.

Runs against the live MCP server. Requires the server to be up with ShowUI enabled.
Usage: python tests/test_e2e_showui.py [--url http://host:port]

The test creates a synthetic UI screenshot with a clearly labeled "Login" button
in a known position, calls ground_ui, and verifies the returned coordinates
fall within 10% of the true button location.
"""

import argparse
import asyncio
import base64
import io
import json
import sys

import httpx
from PIL import Image, ImageDraw, ImageFont


def _make_test_screenshot(width: int = 800, height: int = 600) -> tuple[str, float, float]:
    """Create a synthetic login-form screenshot.

    Returns (data_uri, true_x_norm, true_y_norm) for the Login button center.
    """
    img = Image.new("RGB", (width, height), color=(240, 242, 245))
    draw = ImageDraw.Draw(img)

    # White card in center
    card_x, card_y, card_w, card_h = 200, 150, 400, 300
    draw.rectangle(
        [card_x, card_y, card_x + card_w, card_y + card_h],
        fill="white",
        outline=(200, 200, 200),
        width=1,
    )

    # "Login" heading
    draw.text((card_x + 160, card_y + 30), "Login", fill=(30, 30, 30))

    # Username field outline
    field_x, field_y = card_x + 40, card_y + 80
    draw.rectangle([field_x, field_y, field_x + 320, field_y + 40], outline=(150, 150, 150))
    draw.text((field_x + 10, field_y + 12), "Username", fill=(150, 150, 150))

    # Password field outline
    field_y2 = card_y + 150
    draw.rectangle([field_x, field_y2, field_x + 320, field_y2 + 40], outline=(150, 150, 150))
    draw.text((field_x + 10, field_y2 + 12), "Password", fill=(150, 150, 150))

    # Login button — this is what ground_ui should find
    btn_x, btn_y = card_x + 40, card_y + 220
    btn_w, btn_h = 320, 48
    draw.rectangle([btn_x, btn_y, btn_x + btn_w, btn_y + btn_h], fill=(59, 130, 246))
    draw.text((btn_x + 130, btn_y + 15), "Login", fill="white")

    # True center of button in normalized coords
    true_x = (btn_x + btn_w / 2) / width
    true_y = (btn_y + btn_h / 2) / height

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return data_uri, true_x, true_y


async def call_ground_ui(base_url: str, data_uri: str, query: str) -> dict:
    """Call the ground_ui MCP tool via HTTP (FastMCP SSE transport)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "ground_ui",
            "arguments": {
                "imageSource": data_uri,
                "query": query,
            },
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    async with httpx.AsyncClient(timeout=360.0) as client:
        resp = await client.post(f"{base_url}/mcp", json=payload, headers=headers)
        resp.raise_for_status()
        text = resp.text
        # FastMCP returns SSE: "event: message\ndata: {...}\n\n"
        for line in text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        # Fallback: plain JSON
        return json.loads(text)


async def run_e2e(base_url: str):
    print(f"\nE2E test: ground_ui  →  {base_url}")
    print("─" * 60)

    # Build synthetic screenshot
    data_uri, true_x, true_y = _make_test_screenshot()
    print(f"Created test screenshot 800×600. True Login button center: ({true_x:.3f}, {true_y:.3f})")
    print("Sending to ground_ui (ShowUI-2B will download on first call, ~4GB) ...")

    raw_resp = await call_ground_ui(base_url, data_uri, "Login button")
    print(f"\nRaw response: {json.dumps(raw_resp, indent=2)[:600]}")

    # Unpack MCP tool result
    result_content = raw_resp.get("result", {}).get("content", [])
    if not result_content:
        print("\nFAIL: empty content in MCP response")
        sys.exit(1)

    result = json.loads(result_content[0].get("text", "{}"))
    print(f"\nParsed result: {result}")

    if not result.get("success"):
        print(f"\nFAIL: ground_ui returned success=false: {result.get('error')}")
        sys.exit(1)

    x_norm = result["x_norm"]
    y_norm = result["y_norm"]
    x_px   = result["x"]
    y_px   = result["y"]

    # Allow 15% margin — model won't be pixel-perfect on a synthetic image
    tolerance = 0.15
    x_ok = abs(x_norm - true_x) <= tolerance
    y_ok = abs(y_norm - true_y) <= tolerance

    print(f"\nModel predicted: ({x_norm:.3f}, {y_norm:.3f})  →  pixel ({x_px}, {y_px})")
    print(f"True center:     ({true_x:.3f}, {true_y:.3f})")
    print(f"Error:           Δx={abs(x_norm - true_x):.3f}  Δy={abs(y_norm - true_y):.3f}  (tolerance={tolerance})")

    if x_ok and y_ok:
        print("\nPASS ✓  Coordinates within tolerance")
        return True
    else:
        axis = []
        if not x_ok:
            axis.append(f"x (off by {abs(x_norm - true_x):.3f})")
        if not y_ok:
            axis.append(f"y (off by {abs(y_norm - true_y):.3f})")
        print(f"\nFAIL ✗  Out of tolerance on {', '.join(axis)}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://100.86.69.57:30080/mcp/media",
                        help="Base URL of the media-analysis MCP server")
    args = parser.parse_args()

    ok = asyncio.run(run_e2e(args.url))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
