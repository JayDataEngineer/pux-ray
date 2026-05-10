#!/usr/bin/env python3
"""
E2E MCP Cache Test

This test verifies that the ResponseCachingMiddleware correctly caches
tool responses, resulting in significantly faster response times for cached calls.

Run with: docker exec mcp-server python /app/test_cache_e2e.py
"""

import httpx
import json
import time


def test_cache_with_sessions():
    """Test caching by making actual MCP HTTP requests"""

    print("=" * 70)
    print("MCP Cache E2E Test")
    print("=" * 70)

    client = httpx.Client(timeout=120.0)
    session_id = None

    # Initialize session
    print("\n[1] Initializing session...")
    resp = client.post(
        "http://localhost:8000/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json"
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "cache-test", "version": "1.0"}
            }
        }
    )

    # Extract session ID from response headers
    session_id = resp.headers.get("mcp-session-id")
    if not session_id:
        print("    ❌ Failed to get session ID")
        return False

    print(f"    Session ID: {session_id[:16]}...")

    # Test URL
    test_url = "https://httpbin.org/html"

    # First call - CACHE MISS
    print(f"\n[2] First call (cache MISS)...")
    print(f"    URL: {test_url}")

    start1 = time.time()
    resp1 = client.post(
        "http://localhost:8000/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-session-id": session_id
        },
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "scrape_url",
                "arguments": {"url": test_url}
            }
        }
    )
    time1 = time.time() - start1

    # Parse SSE response
    result1 = None
    for line in resp1.text.split('\n'):
        if line.startswith('data:'):
            data = line[5:].strip()
            if data and data != '[DONE]':
                try:
                    parsed = json.loads(data)
                    if "result" in parsed:
                        result1 = parsed["result"]
                    elif "error" in parsed:
                        print(f"    ❌ Error: {parsed['error'].get('message')}")
                        return False
                    break
                except json.JSONDecodeError:
                    pass

    if result1:
        success1 = result1.get("success", False)
        content1 = result1.get("content", "")
        print(f"    Success: {success1}")
        print(f"    Content: {len(content1)} chars")
        print(f"    Time: {time1*1000:.0f}ms")

    # Wait a moment
    time.sleep(0.5)

    # Second call - CACHE HIT
    print(f"\n[3] Second call (cache HIT)...")

    start2 = time.time()
    resp2 = client.post(
        "http://localhost:8000/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-session-id": session_id
        },
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "scrape_url",
                "arguments": {"url": test_url}
            }
        }
    )
    time2 = time.time() - start2

    # Parse SSE response
    result2 = None
    for line in resp2.text.split('\n'):
        if line.startswith('data:'):
            data = line[5:].strip()
            if data and data != '[DONE]':
                try:
                    parsed = json.loads(data)
                    if "result" in parsed:
                        result2 = parsed["result"]
                    elif "error" in parsed:
                        print(f"    ❌ Error: {parsed['error'].get('message')}")
                        return False
                    break
                except json.JSONDecodeError:
                    pass

    if result2:
        success2 = result2.get("success", False)
        print(f"    Success: {success2}")
        print(f"    Time: {time2*1000:.0f}ms")

    # Third call - confirm cache
    print(f"\n[4] Third call (confirm cache)...")

    start3 = time.time()
    resp3 = client.post(
        "http://localhost:8000/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-session-id": session_id
        },
        json={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "scrape_url",
                "arguments": {"url": test_url}
            }
        }
    )
    time3 = time.time() - start3

    # Parse SSE response
    result3 = None
    for line in resp3.text.split('\n'):
        if line.startswith('data:'):
            data = line[5:].strip()
            if data and data != '[DONE]':
                try:
                    parsed = json.loads(data)
                    if "result" in parsed:
                        result3 = parsed["result"]
                    break
                except json.JSONDecodeError:
                    pass

    if result3:
        print(f"    Time: {time3*1000:.0f}ms")

    client.close()

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"First call:  {time1*1000:.0f}ms (cache MISS)")
    print(f"Second call: {time2*1000:.0f}ms (cache HIT)")
    print(f"Third call:  {time3*1000:.0f}ms (cache HIT)")

    # Verify cache is working
    cache_working = False

    # Cached should be significantly faster
    if time2 < time1 * 0.8:  # 20% faster or more
        speedup = time1 / time2
        cache_working = True
        print(f"\n✅ CACHING CONFIRMED!")
        print(f"   Cache speedup: {speedup:.1f}x faster")
    elif time2 < 1.0 and time1 > 2.0:
        cache_working = True
        print(f"\n✅ CACHING CONFIRMED!")
        print(f"   Cached response < 1s, original > 2s")
    elif time3 < 1.0 and time1 > 2.0:
        cache_working = True
        print(f"\n✅ CACHING CONFIRMED!")
        print(f"   Third cached response < 1s, original > 2s")
    else:
        print(f"\n⚠️  Cache not clearly faster")
        print(f"   This could indicate:")
        print(f"   - Cache key mismatch")
        print(f"   - Response is large (serialization overhead)")
        print(f"   - Network latency (if testing over Tailscale)")

    print("\n" + "=" * 70)
    print("Expected behavior:")
    print("  - First call:  Slow (2-10s for actual scraping)")
    print("  - Cached calls:  Fast (< 100ms on localhost)")
    print("=" * 70)

    return cache_working


if __name__ == "__main__":
    result = test_cache_with_sessions()
    exit(0 if result else 1)
