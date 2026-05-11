"""End-to-end integration tests for LLM /configure + inference.

Tests the full path: ingress → master_router → LLM deployment → llama-server.

Usage:
  python scripts/test_llm_e2e.py [--base-url http://100.86.69.57:18080]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

BASE = "http://100.86.69.57:18080"
PASS = 0
FAIL = 0
SKIP = 0


def _report(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def _skip(name: str, reason: str):
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {name} — {reason}")


def test_health(client: httpx.Client):
    r = client.get("/health")
    _report("GET /health returns 200", r.status_code == 200, f"status={r.status_code}")


def test_status(client: httpx.Client):
    r = client.get("/status")
    _report("GET /status returns 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        _report("Status has vram key", "vram" in data, f"keys={list(data.keys())}")


def test_list_services(client: httpx.Client):
    r = client.get("/v1/services")
    _report("GET /v1/services returns 200", r.status_code == 200)
    if r.status_code == 200:
        services = r.json()
        names = [s["name"] for s in services]
        _report("LLM service registered", "llm" in names, f"names={sorted(names)}")


def test_service_info(client: httpx.Client):
    r = client.get("/v1/services/llm")
    _report("GET /v1/services/llm returns 200", r.status_code == 200,
            f"status={r.status_code} body={r.text[:200] if r.status_code != 200 else ''}")
    if r.status_code == 200:
        info = r.json()
        _report("LLM service info has model_aliases", "model_aliases" in info)


def test_configure_default(client: httpx.Client):
    """Configure with default model (Q5_K_S)."""
    r = client.post("/v1/llm/configure", json={"model": "qwen3.6-27b-q5_k_s"}, timeout=300)
    _report("POST /v1/llm/configure (q5_k_s) returns 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        _report("Configure returns status=ok", data.get("status") == "ok", f"status={data.get('status')}")
        _report("Configure returns engine", "engine" in data, f"engine={data.get('engine')}")
        _report("Configure returns model name", data.get("model") == "qwen3.6-27b-q5_k_s",
                f"model={data.get('model')}")
        _report("Engine is beellama", data.get("engine") == "beellama", f"engine={data.get('engine')}")
        _report("Configure returns session_defaults", "session_defaults" in data)
        if "session_defaults" in data:
            sd = data["session_defaults"]
            _report("Session defaults has temperature", "temperature" in sd,
                    f"keys={sorted(sd.keys())}")
            _report("Session defaults has top_p", "top_p" in sd)
            _report("Session defaults has chat_template_kwargs", "chat_template_kwargs" in sd,
                    f"sd={json.dumps(sd, indent=2)}")
    return r.status_code == 200


def test_configure_idempotent(client: httpx.Client):
    """Second configure with same model should not restart (changed=false)."""
    r = client.post("/v1/llm/configure", json={"model": "qwen3.6-27b-q5_k_s"}, timeout=300)
    _report("Idempotent configure returns 200", r.status_code == 200)
    if r.status_code == 200:
        data = r.json()
        _report("Idempotent configure: changed=false", data.get("changed") is False,
                f"changed={data.get('changed')}")


def test_configure_with_overrides(client: httpx.Client):
    """Configure with session defaults override."""
    body = {
        "model": "qwen3.6-27b-q5_k_s",
        "session_defaults": {
            "temperature": 0.3,
            "top_p": 0.9,
        },
    }
    r = client.post("/v1/llm/configure", json=body, timeout=300)
    _report("Configure with overrides returns 200", r.status_code == 200)
    if r.status_code == 200:
        data = r.json()
        sd = data.get("session_defaults", {})
        _report("Override temperature applied", sd.get("temperature") == 0.3,
                f"temp={sd.get('temperature')}")
        _report("Override top_p applied", sd.get("top_p") == 0.9, f"top_p={sd.get('top_p')}")


def test_configure_upstream_engine(client: httpx.Client):
    """Configure with upstream engine (no DFlash) — skipped by default (slow restart)."""
    _skip("Upstream engine test", "requires 300+ second restart, run with --test-upstream")


def test_chat_completions(client: httpx.Client):
    """OpenAI-compatible /v1/chat/completions."""
    body = {
        "messages": [{"role": "user", "content": "Say exactly: hello world"}],
        "max_tokens": 100,
        "temperature": 0.0,
    }
    r = client.post("/v1/chat/completions", json=body, timeout=120)
    _report("POST /v1/chat/completions returns 200", r.status_code == 200,
            f"status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        _report("Response has choices", "choices" in data, f"keys={list(data.keys())}")
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            has_output = len(content) > 0 or len(reasoning) > 0
            _report("Response has content or reasoning", has_output,
                    f"content={content[:60]}, reasoning={reasoning[:60]}")
    elif r.status_code == 500:
        try:
            err = r.json()
            print(f"         Error: {json.dumps(err, indent=2)}")
        except Exception:
            print(f"         Raw: {r.text[:500]}")


def test_chat_completions_thinking_off(client: httpx.Client):
    """Chat with thinking disabled per-request."""
    body = {
        "messages": [{"role": "user", "content": "Say exactly: test pass"}],
        "max_tokens": 20,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = client.post("/v1/chat/completions", json=body, timeout=120)
    _report("POST /v1/chat/completions (thinking off) returns 200",
            r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            _report("Thinking-off response has content", len(content) > 0,
                    f"content={content[:80]}")


def test_tnap_generate(client: httpx.Client):
    """TNAP format: POST /v1/{service}/generate."""
    body = {
        "input": {
            "messages": [{"role": "user", "content": "Say exactly: tnap works"}],
            "max_tokens": 20,
            "temperature": 0.0,
        },
    }
    r = client.post("/v1/llm/generate", json=body, timeout=120)
    _report("POST /v1/llm/generate returns 200", r.status_code == 200,
            f"status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        _report("TNAP response has output", "output" in data or "choices" in data,
                f"keys={list(data.keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE)
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference tests (only test configure)")
    args = parser.parse_args()

    print(f"\nEnd-to-end LLM tests — {args.base_url}\n")

    with httpx.Client(base_url=args.base_url, timeout=300) as client:
        # Phase 1: Health and discovery
        print("Phase 1: Health & Discovery")
        test_health(client)
        test_status(client)
        test_list_services(client)
        test_service_info(client)

        # Phase 2: Configure
        print("\nPhase 2: Configure")
        ok = test_configure_default(client)
        if not ok:
            print("\nConfigure failed — skipping remaining tests")
            _summarize()
            return 1
        test_configure_idempotent(client)
        test_configure_with_overrides(client)
        test_configure_upstream_engine(client)

        if args.skip_inference:
            print("\n--skip-inference: skipping inference tests")
            _summarize()
            return 0

        # Phase 3: Inference
        print("\nPhase 3: Inference")
        test_chat_completions(client)
        test_chat_completions_thinking_off(client)
        test_tnap_generate(client)

    _summarize()
    return 1 if FAIL > 0 else 0


def _summarize():
    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print(f"{'='*50}")


if __name__ == "__main__":
    sys.exit(main())
