"""API tests for POST /api/v1/chat endpoint."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_chat_successful_mock_response(async_client: AsyncClient) -> None:
    payload = {
        "session_id": "test_session_123",
        "message": "What services does Bitmax Group offer?",
        "conversation_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
        ],
    }
    response = await async_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["session_id"] == "test_session_123"
    assert "response" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert "metadata" in data
    assert data["metadata"]["provider"] == "mock"
    assert "prompt_version" in data["metadata"]


@pytest.mark.asyncio
async def test_chat_empty_message_validation_error(async_client: AsyncClient) -> None:
    payload = {
        "session_id": "test_session_123",
        "message": "",  # min_length is 1
    }
    response = await async_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_request_id_header_present(async_client: AsyncClient) -> None:
    payload = {
        "session_id": "test_session_456",
        "message": "Hello LEO",
    }
    response = await async_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    assert "x-request-id" in response.headers
