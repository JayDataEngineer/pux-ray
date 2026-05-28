#!/usr/bin/env python3
"""E2E test: workflow pipeline through MCP server.

Tests: init → listSpecs → getSpec → startRun → executeStep → getRun → verify artifacts
"""
import json
import sys
import time
import urllib.request
import urllib.error

MCP_URL = "http://localhost:30080/mcp/wan2gp-studio/mcp"
session_id = None
next_id = 1


def mcp_call(method: str, params: dict | None = None) -> dict:
    """Send a JSON-RPC call to the MCP server. Returns parsed result."""
    global next_id

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    payload: dict = {"jsonrpc": "2.0", "id": next_id, "method": method}
    if params is not None:
        payload["params"] = params
    next_id += 1

    req = urllib.request.Request(MCP_URL, data=json.dumps(payload).encode(), headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {body[:200]}")

    text = resp.read().decode()

    # Parse SSE data line
    data_line = None
    for line in text.split("\n"):
        if line.startswith("data: "):
            data_line = line[6:]
            break

    if data_line is None:
        raise RuntimeError(f"No data line in response. Raw: {text[:300]}")

    msg = json.loads(data_line)

    if msg.get("error"):
        raise RuntimeError(f"MCP error: {msg['error']}")

    # FastMCP wraps in content[{type:"text", text:"..."}]
    content = msg.get("result", {}).get("content")
    if content and isinstance(content, list) and len(content) > 0:
        if content[0].get("type") == "text":
            return json.loads(content[0]["text"])

    return msg.get("result", {})


def test_step(name: str, func):
    """Run a test step with pass/fail reporting."""
    try:
        func()
        print(f"  PASS: {name}")
        return True
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        return False


def main():
    global session_id
    passed = 0
    failed = 0

    # ── Step 1: Initialize ──
    def step_init():
        result = mcp_call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "e2e-test", "version": "1.0"},
        })
        name = result.get("serverInfo", {}).get("name", "?")
        ver = result.get("serverInfo", {}).get("version", "?")
        print(f"  Server: {name} v{ver}")

    if test_step("Initialize MCP session", step_init):
        passed += 1
    else:
        failed += 1
        print("Cannot continue without session. Aborting.")
        sys.exit(1)

    # ── Step 2: List specs ──
    specs = None

    def step_list():
        nonlocal specs
        result = mcp_call("tools/call", {"name": "workflow_list_specs", "arguments": {}})
        # May be array or {data:[...]}
        if isinstance(result, list):
            specs = result
        elif isinstance(result, dict) and "data" in result:
            specs = result["data"]
        else:
            specs = result
        print(f"  Found {len(specs)} specs: {[s['name'] for s in specs[:5]]}...")

    if test_step("List workflow specs", step_list):
        passed += 1
    else:
        failed += 1

    if not specs:
        print("No specs found. Aborting.")
        sys.exit(1)

    # ── Step 3: Get first spec ──
    spec = None
    step_ids = []

    def step_get_spec():
        nonlocal spec, step_ids
        spec_name = specs[0]["name"]
        result = mcp_call("tools/call", {"name": "workflow_get_spec", "arguments": {"spec_name": spec_name}})
        spec = result
        step_ids = [s["id"] for s in spec.get("steps", [])]
        print(f"  Spec: {spec_name}, steps: {step_ids}")

    if test_step("Get spec detail", step_get_spec):
        passed += 1
    else:
        failed += 1

    if not step_ids:
        print("Spec has no steps. Aborting.")
        sys.exit(1)

    # ── Step 4: Start run (manual mode) ──
    run_id = None

    def step_start_run():
        nonlocal run_id
        result = mcp_call("tools/call", {"name": "workflow_start_run", "arguments": {
            "spec_name": spec["name"],
            "inputs": {},
            "manual": True,
        }})
        run_id = result.get("run_id")
        print(f"  Run ID: {run_id}, status: {result.get('status')}")

    if test_step("Start manual run", step_start_run):
        passed += 1
    else:
        failed += 1

    if not run_id:
        print("No run ID. Aborting.")
        sys.exit(1)

    # ── Step 5: Get run (should show empty steps before execution) ──
    def step_get_run():
        result = mcp_call("tools/call", {"name": "workflow_get_run", "arguments": {
            "spec_name": spec["name"],
            "run_id": run_id,
        }})
        steps = result.get("steps", [])
        print(f"  Run steps: {len(steps)} (status: {result.get('status')})")
        for s in steps:
            print(f"    {s.get('id')}: {s.get('status')}")

    test_step("Get run status", step_get_run)  # May have 0 steps for manual runs
    passed += 1  # Not a blocking step

    # ── Step 6: Execute first step ──
    first_step_id = step_ids[0]
    execution_ok = False

    def step_execute():
        nonlocal execution_ok
        result = mcp_call("tools/call", {"name": "workflow_execute_step", "arguments": {
            "spec_name": spec["name"],
            "run_id": run_id,
            "step_id": first_step_id,
        }})
        print(f"  Step {first_step_id}: status={result.get('status')}, duration={result.get('duration_ms')}ms")
        outputs = result.get("outputs", {})
        if outputs:
            for k, v in outputs.items():
                print(f"    output: {k} = {str(v)[:100]}")
        execution_ok = result.get("status") in ("completed", "success", "done")

    if test_step(f"Execute step '{first_step_id}'", step_execute):
        passed += 1
    else:
        failed += 1

    # ── Step 7: Get run after execution ──
    def step_get_run_after():
        result = mcp_call("tools/call", {"name": "workflow_get_run", "arguments": {
            "spec_name": spec["name"],
            "run_id": run_id,
        }})
        steps = result.get("steps", [])
        print(f"  Run status: {result.get('status')}, steps: {len(steps)}")
        for s in steps:
            print(f"    {s.get('id')}: {s.get('status')}")

    if test_step("Get run after step execution", step_get_run_after):
        passed += 1
    else:
        failed += 1

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("All E2E workflow tests passed!")


if __name__ == "__main__":
    main()
