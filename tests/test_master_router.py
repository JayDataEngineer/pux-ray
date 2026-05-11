"""Tests for MasterRouter — _InnerRequest, HEAVY_SERVICES, service lifecycle.

No Ray cluster needed. Service imports are mocked.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── _InnerRequest Tests ───────────────────────────────────────────────────


class TestInnerRequest:

    def test_json_returns_data(self):
        from services.creative.master_router import _InnerRequest
        req = _InnerRequest({"text": "hello"}, path="/trellis")
        assert asyncio.get_event_loop().run_until_complete(req.json()) == {"text": "hello"}

    def test_body_returns_json_bytes(self):
        from services.creative.master_router import _InnerRequest
        req = _InnerRequest({"key": "val"}, path="/test")
        body = asyncio.get_event_loop().run_until_complete(req.body())
        assert json.loads(body) == {"key": "val"}

    def test_url_path_set(self):
        from services.creative.master_router import _InnerRequest
        req = _InnerRequest({}, path="/ace_step")
        assert req.url.path == "/ace_step"

    def test_method_and_headers(self):
        from services.creative.master_router import _InnerRequest
        req = _InnerRequest({})
        assert req.method == "POST"
        assert req.headers["content-type"] == "application/json"


# ─── HEAVY_SERVICES Validation ─────────────────────────────────────────────


class TestHeavyServicesConfig:

    def test_heavy_services_complete(self):
        from services.creative.master_router import HEAVY_SERVICES
        expected = {
            "trellis", "ace_step", "comfyui", "hy_motion",
            "moss_soundeffect", "anigen", "see_through", "llm",
            "vibevoice_microsoft", "vibevoice_community_tts", "phi4mm",
        }
        assert set(HEAVY_SERVICES) == expected

    def test_excludes_cpu_services(self):
        from services.creative.master_router import HEAVY_SERVICES
        cpu = {"kokoro", "espeak", "faster_whisper", "vibevoice_cpp_gpu", "vibevoice_cpp_cpu"}
        assert not (cpu & set(HEAVY_SERVICES)), f"CPU services in HEAVY_SERVICES: {cpu & set(HEAVY_SERVICES)}"

    def test_load_kwargs_for_llm(self):
        from services.creative.master_router import LOAD_KWARGS
        assert "llm" in LOAD_KWARGS
        assert "model_name" in LOAD_KWARGS["llm"]


# ─── MasterRouter Validation ───────────────────────────────────────────────


class TestMasterRouterValidation:

    def _make_router(self):
        from services.creative.master_router import MasterRouter
        cls = MasterRouter.func_or_class if hasattr(MasterRouter, 'func_or_class') else MasterRouter
        router = cls.__new__(cls)
        router.active_service = None
        router._services = {}
        router._loaded = {}
        router._loaded_model = {}
        return router

    def test_unknown_service_returns_400(self):
        router = self._make_router()
        mock_req = MagicMock()
        mock_req.json = AsyncMock(return_value={"service": "unknown_thing"})

        with patch.object(router, '_load_service'):
            resp = asyncio.get_event_loop().run_until_complete(router(mock_req))
            assert resp.status_code == 400

    def test_missing_service_returns_400(self):
        router = self._make_router()
        mock_req = MagicMock()
        mock_req.json = AsyncMock(return_value={"prompt": "hello"})

        resp = asyncio.get_event_loop().run_until_complete(router(mock_req))
        assert resp.status_code == 400

    def test_get_service_unknown_raises(self):
        router = self._make_router()
        with pytest.raises(ValueError, match="Unknown heavy service"):
            router._get_service("nonexistent")

    def test_get_service_creates_and_caches(self):
        router = self._make_router()
        with patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_cls = MagicMock()
            mock_mod.TRELLISDeployment = mock_cls
            mock_import.return_value = mock_mod

            svc = router._get_service("trellis")
            assert "trellis" in router._services
            assert router._loaded["trellis"] is False
            assert router._loaded_model["trellis"] is None

    def test_unload_active_clears_state(self):
        router = self._make_router()
        router.active_service = "trellis"
        router._loaded["trellis"] = True
        router._loaded_model["trellis"] = "model-v1"
        mock_svc = MagicMock()
        router._services["trellis"] = mock_svc

        with patch("torch.cuda.empty_cache"), patch("torch.cuda.synchronize"):
            router._unload_active()

        assert router.active_service is None
        assert router._loaded["trellis"] is False
        assert router._loaded_model["trellis"] is None

    def test_load_service_unloads_previous(self):
        router = self._make_router()
        router.active_service = "trellis"
        router._loaded["trellis"] = True
        router._services["trellis"] = MagicMock()

        with patch("torch.cuda.empty_cache"), patch("torch.cuda.synchronize"), \
             patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_svc_instance = MagicMock()
            mock_mod.ACEStepDeployment = MagicMock(return_value=mock_svc_instance)
            mock_import.return_value = mock_mod

            router._load_service("ace_step")
            assert router.active_service == "ace_step"

    def test_load_service_skips_if_same_model(self):
        router = self._make_router()
        router.active_service = "llm"
        router._loaded["llm"] = True
        router._loaded_model["llm"] = "qwen3.6-27b-q5_k_s"
        router._services["llm"] = MagicMock()

        # Should not call _unload_active or _load
        router._load_service("llm")
        # Still llm, still loaded with same model
        assert router.active_service == "llm"

    def test_load_kwargs_applied(self):
        router = self._make_router()
        mock_svc = MagicMock()
        mock_deployment = MagicMock()
        mock_deployment.func_or_class = MagicMock(return_value=mock_svc)
        mock_mod = MagicMock()
        mock_mod.LLMDeployment = mock_deployment

        with patch("importlib.import_module", return_value=mock_mod), \
             patch.object(type(router), '_unload_active'), \
             patch("services.creative.master_router.torch.cuda.memory_allocated", return_value=0):
            router._load_service("llm")
        mock_svc._load.assert_called_with(model_name="qwen3.6-27b-q5_k_s")

    def test_model_override_takes_precedence(self):
        router = self._make_router()
        mock_svc = MagicMock()
        mock_deployment = MagicMock()
        mock_deployment.func_or_class = MagicMock(return_value=mock_svc)
        mock_mod = MagicMock()
        mock_mod.LLMDeployment = mock_deployment

        with patch("importlib.import_module", return_value=mock_mod), \
             patch.object(type(router), '_unload_active'), \
             patch("services.creative.master_router.torch.cuda.memory_allocated", return_value=0):
            router._load_service("llm", model_override="custom-model")
        mock_svc._load.assert_called_with(model_name="custom-model")


class TestMasterRouterRouting:

    def _make_router(self):
        from services.creative.master_router import MasterRouter
        cls = MasterRouter.func_or_class if hasattr(MasterRouter, 'func_or_class') else MasterRouter
        router = cls.__new__(cls)
        router.active_service = None
        router._services = {}
        router._loaded = {}
        router._loaded_model = {}
        return router

    def test_strips_service_key_from_body(self):
        router = self._make_router()
        mock_svc = AsyncMock(return_value=MagicMock(status_code=200))
        router._services["trellis"] = mock_svc
        router._loaded["trellis"] = True
        router.active_service = "trellis"

        mock_req = MagicMock()
        mock_req.json = AsyncMock(return_value={"service": "trellis", "prompt": "a cat"})

        with patch("asyncio.to_thread", side_effect=lambda f, *a, **kw: None):
            resp = asyncio.get_event_loop().run_until_complete(router(mock_req))

        call_args = mock_svc.call_args
        if call_args:
            inner_req = call_args[0][0]
            inner_data = asyncio.get_event_loop().run_until_complete(inner_req.json())
            assert "service" not in inner_data
            assert inner_data["prompt"] == "a cat"
