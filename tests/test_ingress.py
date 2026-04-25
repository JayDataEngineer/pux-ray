"""Unit tests for gateway.ingress — API routing and auth.

Ray Serve is NOT running during these tests.
We mock ray.get_actor and serve.get_deployment_handle to isolate ingress logic.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest
from starlette.testclient import TestClient

from gateway.ingress import create_app
from registry.config import Config

# Prevent real Ray connections during unit tests
_RAY_MOCK = mock.patch("ray.get_actor", side_effect=ValueError("no cluster in test"))

def _mock_serve_handle(name=None, app_name=None):
    """Return an async-compatible mock with .remote() that can be awaited."""
    h = mock.AsyncMock()
    h.remote = mock.AsyncMock(return_value={"choices": [{"message": {"content": "test"}}]})
    return h

_SERVE_HANDLE_MOCK = mock.patch("ray.serve.get_deployment_handle", side_effect=_mock_serve_handle)
_CONFIG_RESET = mock.patch.object(Config, "reload", lambda s: setattr(s, "_data", None))


@pytest.fixture(autouse=True)
def _mock_ray():
    with _RAY_MOCK, _SERVE_HANDLE_MOCK:
        yield


def _fresh_config():
    """Reload config singleton so env vars take effect."""
    Config().reload()
    Config._instance._data = {}


class TestAPIIngress:
    """Test Starlette ingress routes without Ray."""

    @pytest.fixture
    def client(self):
        with mock.patch.dict(os.environ, {"TECH_NOIR_API_KEY": ""}, clear=False):
            Config().reload()
            app = create_app()
            with TestClient(app) as c:
                yield c

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_status_returns_gpu_info(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "vram" in data or "gpu" in data

    def test_chat_returns_response_with_minimal_input(self, client):
        """Chat endpoint should return a response with minimal valid input."""
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data

    def test_jobs_list_returns_response(self, client):
        resp = client.get("/jobs")
        assert resp.status_code in (200, 503)

    def test_job_submit_unknown_type(self, client):
        resp = client.post("/jobs/unknown", json={"x": 1})
        assert resp.status_code in (400, 503)

    def test_job_status_not_found(self, client):
        resp = client.get("/jobs/nonexistent")
        assert resp.status_code in (200, 503)


class TestAPIKeyMiddleware:
    """Auth middleware: skips /health and /status, blocks others without key."""

    @pytest.fixture
    def auth_client(self):
        with mock.patch.dict(os.environ, {"TECH_NOIR_API_KEY": "test-secret"}, clear=False):
            Config().reload()
            app = create_app()
            with TestClient(app) as c:
                yield c

    def test_health_bypasses_auth(self, auth_client):
        resp = auth_client.get("/health")
        assert resp.status_code == 200

    def test_status_bypasses_auth(self, auth_client):
        resp = auth_client.get("/status")
        assert resp.status_code == 200

    def test_protected_route_without_key_returns_401(self, auth_client):
        resp = auth_client.post("/jobs/trellis", json={"image": "dummy"})
        assert resp.status_code == 401

    def test_protected_route_with_wrong_key_returns_401(self, auth_client):
        resp = auth_client.post(
            "/jobs/trellis",
            json={"image": "dummy"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_protected_route_with_valid_key(self, auth_client):
        resp = auth_client.post(
            "/jobs/trellis",
            json={"image": "dummy"},
            headers={"X-API-Key": "test-secret"},
        )
        assert resp.status_code != 401

    def test_query_param_key(self, auth_client):
        resp = auth_client.post(
            "/jobs/trellis?api_key=test-secret",
            json={"image": "dummy"},
        )
        assert resp.status_code != 401
