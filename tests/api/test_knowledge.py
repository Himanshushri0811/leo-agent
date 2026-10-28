"""API tests for protected knowledge management endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_source_unauthorized_without_key(async_client: AsyncClient) -> None:
    payload = {
        "name": "Bitmax Documentation",
        "source_url": "https://bitmaxgroup.com/docs",
    }
    response = await async_client.post("/api/v1/knowledge/sources", json=payload)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_register_source_forbidden_with_wrong_key(async_client: AsyncClient) -> None:
    payload = {
        "name": "Bitmax Documentation",
        "source_url": "https://bitmaxgroup.com/docs",
    }
    response = await async_client.post(
        "/api/v1/knowledge/sources",
        json=payload,
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_register_source_success_with_valid_key(async_client: AsyncClient) -> None:
    payload = {
        "name": "Bitmax Overview",
        "source_url": "https://bitmaxgroup.com/about",
        "notes": "Verified company about page",
    }
    response = await async_client.post(
        "/api/v1/knowledge/sources",
        json=payload,
        headers={"X-API-Key": "test-admin-api-key"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Bitmax Overview"
    assert data["status"] == "pending"
    assert "source_id" in data
