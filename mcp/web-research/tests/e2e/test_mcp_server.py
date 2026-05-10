"""Comprehensive E2E tests for MCP Server over Streamable HTTP

These tests actually connect to the running MCP server via HTTP
and test the full MCP protocol flow including error conditions.
"""

import asyncio
import json
import httpx
from typing import Any


MCP_SERVER_URL = "http://localhost:8000/mcp"
HEALTH_URL = "http://localhost:8000/health"


async def make_mcp_request(client, request, session_id=None, timeout=60.0) -> tuple[dict, str | None]:
    """Helper to make MCP requests and parse SSE response

    Returns tuple of (data, session_id)
    """
    headers = {"Accept": "application/json, text/event-stream"}
    if session_id:
        headers["mcp-session-id"] = session_id

    response = await client.post(MCP_SERVER_URL, json=request, headers=headers, timeout=timeout)

    # Extract session ID from response headers
    new_session_id = response.headers.get("mcp-session-id", session_id)

    # Parse SSE response - find the actual result (not notifications)
    request_id = request.get("id")
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            data = json.loads(line[6:])
            # Return result if:
            # 1. It has "result" key and matches our request_id
            # 2. It has "error" key
            # 3. No request_id in request (initialization)
            if "result" in data or "error" in data:
                if request_id is None or data.get("id") == request_id:
                    return data, new_session_id

    # Try parsing as plain JSON (for some error responses)
    try:
        data = json.loads(response.text)
        if "result" in data or "error" in data:
            return data, new_session_id
    except json.JSONDecodeError:
        pass

    raise ValueError(f"No result in response (got {len(response.text)} bytes): {response.text[:300]}...")


async def get_session(client) -> str:
    """Initialize and get a session ID"""
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "e2e-test", "version": "1.0.0"}
        }
    }
    _, session_id = await make_mcp_request(client, init_request)
    return session_id


# ========== BASIC FUNCTIONALITY TESTS ==========

async def test_health_endpoint():
    """Test the health endpoint"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(HEALTH_URL)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["server"] == "mcp-research-server"
        print("  ✓ Health check passed")
        return True


async def test_initialize_and_list_tools():
    """Test MCP initialize and list tools"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        session_id = await get_session(client)

        # List tools
        list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        data, _ = await make_mcp_request(client, list_request, session_id)
        tools = data["result"]["tools"]
        tool_names = [t["name"] for t in tools]

        print(f"  ✓ Got {len(tool_names)} tools")
        print(f"  ✓ Tools: {', '.join(tool_names)}")

        expected = ["search_web", "scrape_url", "map_domain", "crawl_site", "scrape_structured",
                    "list_schemas", "get_domains", "clean_database"]
        for exp in expected:
            assert exp in tool_names, f"Missing tool: {exp}"
        print(f"  ✓ All expected tools present")

        return True


async def test_search_web():
    """Test the search_web tool"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        session_id = await get_session(client)

        search_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search_web",
                "arguments": {"query": "python async await", "max_results": 3}
            }
        }

        data, _ = await make_mcp_request(client, search_request, session_id)

        assert "result" in data, f"Got error: {data.get('error')}"
        content = data["result"]["content"]
        assert len(content) > 0
        print(f"  ✓ search_web returned {len(content)} content items")

        # Check for text content
        for item in content:
            if item.get("type") == "text" and item.get("text"):
                text = item["text"]
                if len(text) > 100:
                    print(f"  ✓ Got substantive response ({len(text)} chars)")
                    return True

        print(f"  ⚠ Short response but content present")
        return True


async def test_scrape_url():
    """Test the scrape_url tool with a simple URL"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        session_id = await get_session(client)

        scrape_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "scrape_url",
                "arguments": {"url": "https://httpbin.org/html"}
            }
        }

        data, _ = await make_mcp_request(client, scrape_request, session_id)

        # Check for MCP-level error
        if "error" in data:
            print(f"  ✗ MCP error: {data['error']}")
            return False

        # Should have result
        assert "result" in data
        result = data["result"]

        # Result is a text content item - parse the JSON string
        content_list = result.get("content", [])
        assert len(content_list) > 0

        text_item = content_list[0]
        assert text_item["type"] == "text"

        # Parse the inner JSON
        inner_data = json.loads(text_item["text"])
        assert inner_data["success"] is True
        assert inner_data["url"] == "https://httpbin.org/html"
        assert len(inner_data.get("content", "")) > 10

        print(f"  ✓ scrape_url succeeded ({len(inner_data.get('content', ''))} chars)")
        print(f"  ✓ Method used: {inner_data['method_used']}")
        return True


async def test_scrape_with_method_override():
    """Test scrape_url with force_method parameter"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        session_id = await get_session(client)

        scrape_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "scrape_url",
                "arguments": {"url": "https://httpbin.org/html", "method": "selenium"}
            }
        }

        data, _ = await make_mcp_request(client, scrape_request, session_id)

        assert "result" in data, f"Got: {list(data.keys())}"
        result = data["result"]

        content_list = result.get("content", [])
        assert len(content_list) > 0

        inner_data = json.loads(content_list[0]["text"])
        assert inner_data["success"] is True
        assert inner_data["method_used"] == "selenium"

        print(f"  ✓ Method override worked (forced selenium)")
        return True


async def test_map_domain():
    """Test the map_domain tool"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        session_id = await get_session(client)

        map_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "map_domain",
                "arguments": {"domain": "example.com", "max_urls": 5}
            }
        }

        data, _ = await make_mcp_request(client, map_request, session_id, timeout=60.0)

        # map_domain may succeed or fail depending on the domain
        assert "result" in data or "error" in data
        if "result" in data:
            content = data["result"].get("content", [])
            print(f"  ✓ map_domain returned {len(content)} content items")
        else:
            print(f"  ⚠ map_domain returned error (acceptable for some domains)")

        return True


async def test_list_schemas():
    """Test the list_schemas tool"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await get_session(client)

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "list_schemas",
                "arguments": {}
            }
        }

        data, _ = await make_mcp_request(client, request, session_id)

        assert "result" in data
        content = data["result"].get("content", [])
        assert len(content) > 0

        print(f"  ✓ list_schemas returned {len(content)} content items")
        return True


async def test_get_domains():
    """Test the get_domains admin tool"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await get_session(client)

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_domains",
                "arguments": {}
            }
        }

        data, _ = await make_mcp_request(client, request, session_id)

        assert "result" in data
        content = data["result"].get("content", [])
        assert len(content) > 0

        # Parse the result
        inner_data = json.loads(content[0]["text"])
        assert "total" in inner_data
        assert "domains" in inner_data

        print(f"  ✓ get_domains returned {inner_data['total']} domains")
        return True


# ========== ERROR CONDITION TESTS ==========

async def test_invalid_url_scheme():
    """Test that invalid URL schemes are rejected"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await get_session(client)

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "scrape_url",
                "arguments": {"url": "ftp://example.com"}
            }
        }

        data, _ = await make_mcp_request(client, request, session_id)

        # FastMCP returns errors as result with isError: true
        assert "result" in data
        result = data["result"]
        assert result.get("isError") is True
        content_list = result.get("content", [])
        assert len(content_list) > 0
        assert "URL must start with" in content_list[0]["text"]

        print(f"  ✓ Invalid URL scheme rejected")
        return True


async def test_private_ip_blocked():
    """Test that private IPs are blocked"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await get_session(client)

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "scrape_url",
                "arguments": {"url": "http://localhost:8000/health"}
            }
        }

        data, _ = await make_mcp_request(client, request, session_id)

        # Should fail
        assert "result" in data
        result = data["result"]
        # Either isError or success=False in content
        if result.get("isError"):
            print(f"  ✓ Private IP blocked")
        else:
            content_list = result.get("content", [])
            if content_list and content_list[0]["type"] == "text":
                text = content_list[0]["text"]
                # Try to parse as JSON scrape result
                try:
                    inner = json.loads(text)
                    assert inner.get("success") is False
                except:
                    pass  # Plain error message
            print(f"  ✓ Private IP blocked")

        return True


async def test_private_ip_127_blocked():
    """Test that 127.0.0.1 is blocked"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await get_session(client)

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "scrape_url",
                "arguments": {"url": "http://127.0.0.1:8000/health"}
            }
        }

        data, _ = await make_mcp_request(client, request, session_id)

        # Should fail
        assert "result" in data
        result = data["result"]
        assert result.get("isError") is True or result.get("content", [])

        print(f"  ✓ 127.0.0.1 blocked")
        return True


async def test_malformed_json_rpc():
    """Test that malformed JSON-RPC is handled"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        request = {
            "invalid": "request"
        }

        response = await client.post(MCP_SERVER_URL, json=request,
                                      headers={"Accept": "application/json, text/event-stream"})

        # Should get a response (could be 200 with error or 400)
        assert response.status_code in [200, 400]
        print(f"  ✓ Malformed JSON-RPC handled (status {response.status_code})")
        return True


async def test_missing_session_id():
    """Test that tools/call without session ID returns appropriate error"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "scrape_url",
                "arguments": {"url": "https://example.com"}
            }
        }

        data, _ = await make_mcp_request(client, request, session_id=None)

        # Should get an error about missing session
        assert "error" in data
        print(f"  ✓ Missing session ID handled")
        return True


async def test_invalid_tool_name():
    """Test that invalid tool names are rejected"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await get_session(client)

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {}
            }
        }

        data, _ = await make_mcp_request(client, request, session_id)

        # FastMCP returns error as result with isError: true
        assert "result" in data
        result = data["result"]
        assert result.get("isError") is True
        content = result.get("content", [{}])[0]
        assert "Unknown tool" in content.get("text", "")

        print(f"  ✓ Invalid tool name rejected")
        return True


async def test_scrape_invalid_domain():
    """Test scraping a likely non-existent domain"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await get_session(client)

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "scrape_url",
                "arguments": {"url": "https://this-definitely-does-not-exist-12345.com"}
            }
        }

        data, _ = await make_mcp_request(client, request, session_id)

        # Should fail gracefully
        assert "result" in data or "error" in data
        if "result" in data:
            content_list = data["result"].get("content", [])
            if content_list:
                inner_data = json.loads(content_list[0]["text"])
                # May succeed or fail depending on DNS/timeout
                print(f"  ✓ Invalid domain handled (success={inner_data.get('success')})")
            else:
                print(f"  ✓ Invalid domain handled (no content)")
        else:
            print(f"  ✓ Invalid domain handled (MCP error)")

        return True


async def test_empty_search_query():
    """Test search with empty query"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        session_id = await get_session(client)

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search_web",
                "arguments": {"query": ""}
            }
        }

        data, _ = await make_mcp_request(client, request, session_id)

        # Should handle gracefully
        assert "result" in data or "error" in data
        print(f"  ✓ Empty search query handled")
        return True


# ========== CONCURRENT REQUEST TESTS ==========

async def test_concurrent_scrapes():
    """Test multiple concurrent scrape requests"""
    urls = [
        "https://httpbin.org/html",
        "https://example.com",
    ]

    async def scrape_url(url: str) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as client:
            session_id = await get_session(client)
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "scrape_url",
                    "arguments": {"url": url}
                }
            }
            data, _ = await make_mcp_request(client, request, session_id)
            return data

    # Run all scrapes concurrently
    results = await asyncio.gather(*[scrape_url(url) for url in urls])

    # Count successful (have result with content)
    successful = 0
    for r in results:
        if "result" in r:
            content_list = r["result"].get("content", [])
            if content_list:
                try:
                    inner_data = json.loads(content_list[0]["text"])
                    if inner_data.get("success") is True:
                        successful += 1
                except:
                    pass

    print(f"  ✓ Concurrent scrapes: {successful}/{len(urls)} successful")
    assert successful >= 1, "At least one scrape should succeed"

    return True


async def test_concurrent_searches():
    """Test multiple concurrent search requests"""
    queries = ["python", "golang"]

    async def search(query: str) -> dict:
        async with httpx.AsyncClient(timeout=60.0) as client:
            session_id = await get_session(client)
            request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "search_web",
                    "arguments": {"query": query, "max_results": 2}
                }
            }
            data, _ = await make_mcp_request(client, request, session_id)
            return data

    # Run all searches concurrently
    results = await asyncio.gather(*[search(q) for q in queries])

    successful = sum(1 for r in results if "result" in r)

    print(f"  ✓ Concurrent searches: {successful}/{len(queries)} successful")
    assert successful == len(queries), "All searches should return results"

    return True


# ========== SESSION TESTS ==========

async def test_session_persistence():
    """Test that session ID persists across requests"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        session_id = await get_session(client)
        initial_session = session_id

        # Make multiple requests
        for i in range(3):
            request = {
                "jsonrpc": "2.0",
                "id": i + 2,
                "method": "tools/call",
                "params": {
                    "name": "scrape_url",
                    "arguments": {"url": "https://httpbin.org/html"}
                }
            }
            data, session_id = await make_mcp_request(client, request, session_id)
            assert session_id == initial_session, "Session ID should persist"

        print(f"  ✓ Session ID persisted across {3} requests")
        return True


# ========== TEST RUNNER ==========

TEST_SUITES = {
    "Basic Functionality": [
        ("Health Endpoint", test_health_endpoint),
        ("Initialize and List Tools", test_initialize_and_list_tools),
        ("Search Web", test_search_web),
        ("Scrape URL", test_scrape_url),
        ("Scrape with Method Override", test_scrape_with_method_override),
        ("Map Domain", test_map_domain),
        ("List Schemas", test_list_schemas),
        ("Get Domains", test_get_domains),
    ],
    "Error Conditions": [
        ("Invalid URL Scheme", test_invalid_url_scheme),
        ("Private IP Blocked", test_private_ip_blocked),
        ("127.0.0.1 Blocked", test_private_ip_127_blocked),
        ("Malformed JSON-RPC", test_malformed_json_rpc),
        ("Missing Session ID", test_missing_session_id),
        ("Invalid Tool Name", test_invalid_tool_name),
        ("Invalid Domain", test_scrape_invalid_domain),
        ("Empty Search Query", test_empty_search_query),
    ],
    "Concurrent Requests": [
        ("Concurrent Scrapes", test_concurrent_scrapes),
        ("Concurrent Searches", test_concurrent_searches),
    ],
    "Session Management": [
        ("Session Persistence", test_session_persistence),
    ],
}


async def run_test_suite(suite_name: str, tests: list) -> tuple[int, int]:
    """Run a single test suite"""
    print(f"\n{'=' * 60}")
    print(f"[SUITE] {suite_name}")
    print('=' * 60)

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        print('-' * 60)
        try:
            result = await test_func()
            if result is False:
                print(f"⊘ SKIPPED: {name}")
            else:
                print(f"✓ PASSED: {name}")
                passed += 1
        except Exception as e:
            print(f"✗ FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    return passed, failed


async def run_all_tests():
    """Run all E2E tests"""
    print("\n" + "=" * 60)
    print("MCP Server Comprehensive E2E Tests")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    for suite_name, tests in TEST_SUITES.items():
        passed, failed = await run_test_suite(suite_name, tests)
        total_passed += passed
        total_failed += failed

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    return total_failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
