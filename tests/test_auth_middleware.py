"""Tests for APIKeyMiddleware — auth enforcement and hmac.compare_digest.

Uses Starlette TestClient against create_app(). No Ray cluster needed.
"""
from __future__ import annotations

import hmac
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient


class TestAPIKeyMiddlewareNoAuth:

    def test_health_no_key_needed(self):
        with patch("gateway.ingress._get_api_key", return_value=""):
            from gateway.ingress import create_app
            client = TestClient(create_app())
            r = client.get("/health")
            assert r.status_code == 200

    def test_all_routes_pass_without_key(self):
        with patch("gateway.ingress._get_api_key", return_value=""):
            from gateway.ingress import create_app
            client = TestClient(create_app())
            r = client.get("/v1/services")
            assert r.status_code == 200


class TestAPIKeyMiddlewareWithAuth:

    def _authed_client(self):
        """Create TestClient WITH auth middleware active."""
        with patch("gateway.ingress._get_api_key", return_value="test-secret-key"):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                yield client

    def test_valid_key_in_header(self):
        with patch("gateway.ingress._get_api_key", return_value="test-secret-key"):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                r = client.get("/v1/services", headers={"X-API-Key": "test-secret-key"})
                assert r.status_code == 200

    def test_valid_key_in_query_param(self):
        with patch("gateway.ingress._get_api_key", return_value="test-secret-key"):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                r = client.get("/v1/services?api_key=test-secret-key")
                assert r.status_code == 200

    def test_invalid_key_returns_401(self):
        with patch("gateway.ingress._get_api_key", return_value="test-secret-key"):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                r = client.get("/v1/services", headers={"X-API-Key": "wrong"})
                assert r.status_code == 401
                assert r.json()["error"] == "unauthorized"

    def test_missing_key_returns_401(self):
        with patch("gateway.ingress._get_api_key", return_value="test-secret-key"):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                r = client.get("/v1/services")
                assert r.status_code == 401

    def test_empty_key_string_disables_auth(self):
        with patch("gateway.ingress._get_api_key", return_value=""):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                r = client.get("/v1/services")
                assert r.status_code == 200

    def test_health_skips_auth(self):
        with patch("gateway.ingress._get_api_key", return_value="test-secret-key"):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                r = client.get("/health")
                assert r.status_code == 200

    def test_status_skips_auth(self):
        with patch("gateway.ingress._get_api_key", return_value="test-secret-key"):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                r = client.get("/status")
                assert r.status_code == 200

    def test_tnap_generate_requires_key(self):
        with patch("gateway.ingress._get_api_key", return_value="test-secret-key"):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                r = client.post("/v1/espeak/generate", json={"action": "generate"})
                assert r.status_code == 401

    def test_dashboard_requires_key(self):
        with patch("gateway.ingress._get_api_key", return_value="test-secret-key"):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                r = client.get("/dashboard")
                assert r.status_code == 401


class TestHMACComparison:

    def test_uses_compare_digest(self):
        with patch("gateway.ingress._get_api_key", return_value="test-secret-key"):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                with patch("gateway.ingress.hmac.compare_digest", wraps=hmac.compare_digest) as spy:
                    r = client.get("/v1/services", headers={"X-API-Key": "test-secret-key"})
                    spy.assert_called_once_with("test-secret-key", "test-secret-key")
                    assert r.status_code == 200

    def test_wrong_key_calls_compare_digest(self):
        with patch("gateway.ingress._get_api_key", return_value="test-secret-key"):
            from gateway.ingress import create_app
            with TestClient(create_app()) as client:
                with patch("gateway.ingress.hmac.compare_digest", wraps=hmac.compare_digest) as spy:
                    r = client.get("/v1/services", headers={"X-API-Key": "wrong"})
                    spy.assert_called_once_with("wrong", "test-secret-key")
                    assert r.status_code == 401
