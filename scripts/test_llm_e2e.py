"""End-to-end integration tests for LLM service.

Tests the full path: client → ingress → Forge → LLM → llama-server.

The main flow is one call: POST /v1/chat/completions with {"model": "...", "messages": [...]}.
The server auto-configures (smart diff restart) if the model changed, then infers.

Usage:
  python scripts/test_llm_e2e.py [--base-url http://100.86.69.57:30080]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

BASE = "http://100.86.69.57:30080"
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


# ── Phase 1: Health & Discovery ──────────────────────────────────────

def test_health(client: httpx.Client):
    r = client.get("/health")
    _report("GET /health returns 200", r.status_code == 200, f"status={r.status_code}")


def test_status(client: httpx.Client):
    r = client.get("/status")
    _report("GET /status returns 200", r.status_code == 200)
    if r.status_code == 200:
        data = r.json()
        _report("Status has vram key", "vram" in data, f"keys={list(data.keys())}")


def test_list_services(client: httpx.Client):
    r = client.get("/v1/services")
    _report("GET /v1/services returns 200", r.status_code == 200)
    if r.status_code == 200:
        services = r.json()
        names = [s["name"] for s in services]
        _report("LLM service registered", "llm" in names)


def test_service_info(client: httpx.Client):
    r = client.get("/v1/services/llm")
    _report("GET /v1/services/llm returns 200", r.status_code == 200,
            f"status={r.status_code} body={r.text[:200] if r.status_code != 200 else ''}")
    if r.status_code == 200:
        info = r.json()
        _report("LLM service info has model_aliases", "model_aliases" in info)


# ── Phase 2: One-call inference (auto-configures model) ──────────────

def test_chat_q5(client: httpx.Client):
    """First call — server cold-starts Q5_K_S model."""
    body = {
        "model": "qwen3.6-27b-q5_k_s",
        "messages": [{"role": "user", "content": "Say exactly: hello world"}],
        "max_tokens": 100,
        "temperature": 0.0,
    }
    r = client.post("/v1/chat/completions", json=body, timeout=300)
    _report("Q5_K_S chat completions returns 200", r.status_code == 200,
            f"status={r.status_code}")
    if r.status_code == 200:
        _check_response(r, "Q5_K_S")


def test_chat_q5_thinking_off(client: httpx.Client):
    """Same model, thinking disabled — no restart."""
    body = {
        "model": "qwen3.6-27b-q5_k_s",
        "messages": [{"role": "user", "content": "Say exactly: test pass"}],
        "max_tokens": 20,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = client.post("/v1/chat/completions", json=body, timeout=120)
    _report("Q5_K_S thinking-off returns 200", r.status_code == 200,
            f"status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            _report("Thinking-off has content", len(content) > 0,
                    f"content={content[:80]}")


def test_chat_q6_switch(client: httpx.Client):
    """Switch to Q6_K default (125K) — triggers restart via smart diff."""
    body = {
        "model": "qwen3.6-27b-q6_k",
        "messages": [{"role": "user", "content": "Say exactly: switched"}],
        "max_tokens": 100,
        "temperature": 0.0,
    }
    r = client.post("/v1/chat/completions", json=body, timeout=300)
    _report("Q6_K model switch returns 200", r.status_code == 200,
            f"status={r.status_code}")
    if r.status_code == 200:
        _check_response(r, "Q6_K")


def test_chat_q6_same(client: httpx.Client):
    """Same Q6_K model — no restart, instant inference."""
    body = {
        "model": "qwen3.6-27b-q6_k",
        "messages": [{"role": "user", "content": "Say exactly: cached"}],
        "max_tokens": 100,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = client.post("/v1/chat/completions", json=body, timeout=120)
    _report("Q6_K repeat returns 200 (no restart)", r.status_code == 200,
            f"status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            _report("Repeat has content", len(content) > 0, f"content={content[:80]}")


# ── Phase 3: TNAP + Explicit configure ────────────────────────────────

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


def test_explicit_configure(client: httpx.Client):
    """Explicit /configure for session defaults (no inference)."""
    r = client.post("/v1/llm/configure", json={
        "model": "qwen3.6-27b-q6_k",
        "session_defaults": {"temperature": 0.3, "top_p": 0.9},
    }, timeout=300)
    _report("Explicit configure returns 200", r.status_code == 200)
    if r.status_code == 200:
        data = r.json()
        _report("Configure status=ok", data.get("status") == "ok")
        sd = data.get("session_defaults", {})
        _report("Session defaults overridden", sd.get("temperature") == 0.3,
                f"temp={sd.get('temperature')}")


def test_configure_idempotent(client: httpx.Client):
    """Same config twice — changed=false, no restart."""
    r = client.post("/v1/llm/configure", json={
        "model": "qwen3.6-27b-q6_k",
    }, timeout=300)
    _report("Idempotent configure returns 200", r.status_code == 200)
    if r.status_code == 200:
        data = r.json()
        _report("Idempotent: changed=false", data.get("changed") is False,
                f"changed={data.get('changed')}")


# ── Helpers ───────────────────────────────────────────────────────────

def _check_response(r, label: str):
    data = r.json()
    _report(f"{label} has choices", "choices" in data, f"keys={list(data.keys())}")
    if "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        has_output = len(content) > 0 or len(reasoning) > 0
        _report(f"{label} has output", has_output,
                f"content={content[:60]}, reasoning={reasoning[:60]}")
    elif r.status_code == 500:
        try:
            err = r.json()
            print(f"         Error: {json.dumps(err, indent=2)}")
        except Exception:
            print(f"         Raw: {r.text[:500]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE)
    args = parser.parse_args()

    print(f"\nLLM e2e tests — {args.base_url}\n")

    with httpx.Client(base_url=args.base_url, timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        print("Phase 1: Health & Discovery")
        test_health(client)
        test_status(client)
        test_list_services(client)
        test_service_info(client)

        print("\nPhase 2: One-call inference (model in request)")
        test_chat_q5(client)
        test_chat_q5_thinking_off(client)
        test_chat_q6_switch(client)
        test_chat_q6_same(client)

        print("\nPhase 3: TNAP + Explicit configure")
        test_tnap_generate(client)
        test_explicit_configure(client)
        test_configure_idempotent(client)

    _summarize()
    return 1 if FAIL > 0 else 0


def _summarize():
    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print(f"{'='*50}")


if __name__ == "__main__":
    sys.exit(main())
