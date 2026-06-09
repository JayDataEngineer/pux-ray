"""HTTP client for the Workflow Engine.

Calls the same /v1/wf/* gateway routes the frontend uses, but from the MCP
server. Reuses the working gateway → Ray Serve → WorkflowEngine pipeline.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

DEFAULT_BASE_URL = "http://tech-noir-ray-serve-svc.ai-services:8000"
DEFAULT_TIMEOUT = 300.0  # GPU inference is slow


class WorkflowClient:
    """Async HTTP client for the workflow engine gateway routes."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or os.environ.get("WORKFLOW_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout or float(os.environ.get("WORKFLOW_TIMEOUT", DEFAULT_TIMEOUT))
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def list_specs(self) -> dict[str, Any]:
        client = await self._get_client()
        resp = await client.get(f"{self.base_url}/v1/wf")
        resp.raise_for_status()
        return resp.json()

    async def get_spec(self, spec_name: str) -> dict[str, Any]:
        client = await self._get_client()
        resp = await client.get(f"{self.base_url}/v1/wf/{spec_name}")
        resp.raise_for_status()
        return resp.json()

    async def start_run(self, spec_name: str, inputs: dict, manual: bool = True) -> dict[str, Any]:
        client = await self._get_client()
        body = {**inputs, "_manual": manual}
        resp = await client.post(f"{self.base_url}/v1/wf/{spec_name}/runs", json=body)
        resp.raise_for_status()
        return resp.json()

    async def get_run(self, spec_name: str, run_id: str) -> dict[str, Any]:
        client = await self._get_client()
        resp = await client.get(f"{self.base_url}/v1/wf/{spec_name}/runs/{run_id}")
        resp.raise_for_status()
        return resp.json()

    async def cancel_run(self, spec_name: str, run_id: str) -> dict[str, Any]:
        client = await self._get_client()
        resp = await client.delete(f"{self.base_url}/v1/wf/{spec_name}/runs/{run_id}")
        resp.raise_for_status()
        return resp.json()

    async def execute_step(
        self, spec_name: str, run_id: str, step_id: str, params: dict | None = None,
    ) -> dict[str, Any]:
        client = await self._get_client()
        resp = await client.post(
            f"{self.base_url}/v1/wf/{spec_name}/runs/{run_id}/steps/{step_id}/execute",
            json=params or {},
        )
        resp.raise_for_status()
        return resp.json()

    async def approve_step(
        self, spec_name: str, run_id: str, step_id: str, data: dict | None = None,
    ) -> dict[str, Any]:
        client = await self._get_client()
        resp = await client.post(
            f"{self.base_url}/v1/wf/{spec_name}/runs/{run_id}/steps/{step_id}/approve",
            json=data or {},
        )
        resp.raise_for_status()
        return resp.json()

    async def rerun_step(
        self, spec_name: str, run_id: str, step_id: str, params: dict | None = None,
    ) -> dict[str, Any]:
        client = await self._get_client()
        resp = await client.post(
            f"{self.base_url}/v1/wf/{spec_name}/runs/{run_id}/steps/{step_id}/rerun",
            json=params or {},
        )
        resp.raise_for_status()
        return resp.json()

    async def run_and_wait(
        self, spec_name: str, inputs: dict, timeout: float | None = None,
    ) -> dict[str, Any]:
        """Start a non-manual run and poll until completion.

        Returns the final run state dict with step_states and artifacts.
        """
        import asyncio

        deadline = (timeout or self.timeout)
        result = await self.start_run(spec_name, inputs, manual=False)
        run_id = result["run_id"]

        t0 = asyncio.get_event_loop().time()
        while True:
            run = await self.get_run(spec_name, run_id)
            status = run.get("status")
            if status in ("completed", "failed", "cancelled"):
                return run
            elapsed = asyncio.get_event_loop().time() - t0
            if elapsed > deadline:
                await self.cancel_run(spec_name, run_id)
                return {"status": "failed", "error": "Workflow timed out",
                        "run_id": run_id}
            await asyncio.sleep(2.0)

    async def get_artifact_data(
        self, spec_name: str, run_id: str, step_id: str, filename: str,
    ) -> bytes:
        """Download artifact binary data from the workflow engine."""
        client = await self._get_client()
        resp = await client.get(
            f"{self.base_url}/v1/wf/{spec_name}/runs/{run_id}"
            f"/artifacts/{step_id}/{filename}",
        )
        resp.raise_for_status()
        return resp.content
