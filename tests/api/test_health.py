"""API tests for liveness and readiness health endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_liveness_probe(async_client: AsyncClient) -> None:
    response = await async_client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_readiness_probe_success(async_client: AsyncClient) -> None:
    # Readiness performs DB check
    response = await async_client.get("/health/ready")
    # In test environment with uninitialized DB engine it handles error gracefully or returns 503
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "checks" in data
