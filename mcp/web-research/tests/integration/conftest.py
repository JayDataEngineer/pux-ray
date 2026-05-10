"""Pytest configuration and fixtures for integration tests.

These tests run against live services inside Docker:
- MCP server on localhost:8000
- PostgreSQL on postgres:5432
- Redis on redis:6379
- SearXNG on searxng:8080

Run inside container:
    docker exec mcp-server python -m pytest tests/integration/ -v
    docker exec mcp-server python -m pytest tests/integration/ -v -k test_search
"""

import json
import os
import pytest
import httpx


# ---------- Service URLs ----------

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
HEALTH_URL = os.getenv("MCP_HEALTH_URL", "http://localhost:8000/health")
DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@postgres:5432/mcp_server",
    ),
)


# ---------- MCP Protocol Helpers ----------


async def _mcp_request(
    client: httpx.AsyncClient,
    request: dict,
    session_id: str | None = None,
    timeout: float = 60.0,
) -> tuple[dict, str | None]:
    """Send an MCP JSON-RPC request and parse the response.

    Returns (data_dict, session_id).
    """
    headers = {"Accept": "application/json, text/event-stream"}
    if session_id:
        headers["mcp-session-id"] = session_id

    response = await client.post(
        MCP_SERVER_URL, json=request, headers=headers, timeout=timeout
    )
    new_session_id = response.headers.get("mcp-session-id", session_id)

    # Parse SSE stream
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if "result" in data or "error" in data:
                return data, new_session_id

    # Fallback: try plain JSON
    try:
        data = json.loads(response.text)
        if "result" in data or "error" in data:
            return data, new_session_id
    except json.JSONDecodeError:
        pass

    raise ValueError(f"No result in response: {response.text[:300]}")


async def _init_session(client: httpx.AsyncClient) -> str:
    """Initialize MCP session and return session ID."""
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "integration-test", "version": "1.0.0"},
        },
    }
    _, session_id = await _mcp_request(client, init_req)
    return session_id


# ---------- Fixtures ----------


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def mcp_client():
    """HTTP client connected to the MCP server."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        yield client


@pytest.fixture
async def session(mcp_client):
    """Initialized MCP session ID."""
    return await _init_session(mcp_client)


@pytest.fixture
async def db():
    """Direct database session for setup/teardown."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

    engine = create_async_engine(DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession)

    async with session_factory() as sess:
        yield sess

    await engine.dispose()


@pytest.fixture
async def clean_db(db):
    """Database fixture that cleans up all domain records after the test."""
    from sqlalchemy import delete
    from src.db.models import Domain, ScrapeMetric

    yield db

    # Cleanup: remove all test data
    async with db.begin():
        await db.execute(delete(ScrapeMetric))
        await db.execute(delete(Domain))


# ---------- Tool Call Helpers ----------


async def call_tool(
    client: httpx.AsyncClient,
    session_id: str,
    tool_name: str,
    arguments: dict,
    timeout: float = 60.0,
) -> dict:
    """Call an MCP tool and return the parsed result content.

    Raises AssertionError on MCP-level errors.
    """
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    data, _ = await _mcp_request(client, request, session_id, timeout=timeout)

    if "error" in data:
        pytest.fail(f"MCP error calling {tool_name}: {data['error']}")

    result = data["result"]

    # FastMCP wraps errors as isError=True
    if result.get("isError"):
        error_text = result.get("content", [{}])[0].get("text", "Unknown error")
        pytest.fail(f"Tool {tool_name} returned error: {error_text}")

    # Parse JSON content if present
    content_list = result.get("content", [])
    if content_list and content_list[0].get("type") == "text":
        text = content_list[0]["text"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}

    return result


async def call_tool_raw(
    client: httpx.AsyncClient,
    session_id: str,
    tool_name: str,
    arguments: dict,
    timeout: float = 60.0,
) -> dict:
    """Call an MCP tool and return the raw MCP response (including isError)."""
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    data, _ = await _mcp_request(client, request, session_id, timeout=timeout)
    return data
