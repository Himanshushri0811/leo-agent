"""
Pytest root fixtures for LEO AI Backend.

Configures test settings, overrides FastAPI dependencies, and sets up
mock services so API tests execute reliably and instantaneously.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

os.environ["APP_ENV"] = "testing"
os.environ["APP_SECRET_KEY"] = "test-secret-key-for-unit-testing-32-chars-min"
os.environ["ADMIN_API_KEY"] = "test-admin-api-key"
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://leo_user:leo_password@localhost:5432/leo_db"
)
if os.environ.get("DATABASE_URL") == "postgresql+asyncpg://leo_user:leo_pass@localhost:5432/leo_db":
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://leo_user:leo_password@localhost:5432/leo_db"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDINGS_PROVIDER"] = "mock"

from app.ai.orchestrator import ChatResponse as OrchChatResponse
from app.core.config import Settings, get_settings
from app.main import app
from app.services.chat_service import get_chat_service
from app.services.knowledge_service import get_knowledge_service


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(
        app_env="testing",
        app_secret_key="test-secret-key-for-unit-testing-32-chars-min",
        admin_api_key="test-admin-api-key",
        database_url=os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://leo_user:leo_password@localhost:5432/leo_db",
        ),
        llm_provider="mock",
        embeddings_provider="mock",
    )


@pytest.fixture
def mock_chat_service() -> AsyncMock:
    service = AsyncMock()
    service.process_chat.return_value = OrchChatResponse(
        session_id="test_session_123",
        response_text="[Mock LEO response] Bitmax Group provides cloud solutions.",
        sources=[],
        model="mock-model",
        provider="mock",
        prompt_tokens=10,
        completion_tokens=15,
        prompt_version="1.0.0",
    )
    return service


@pytest.fixture
def mock_knowledge_service() -> AsyncMock:
    service = AsyncMock()
    mock_source = MagicMock()
    mock_source.id = uuid.uuid4()
    mock_source.name = "Bitmax Overview"
    mock_source.source_type = "url"
    mock_source.source_url = "https://bitmaxgroup.com/about"
    mock_source.status = "pending"
    mock_source.version = 1
    mock_source.created_at = datetime.now(UTC)
    mock_source.approved_at = None
    mock_source.approved_by = None

    service.register_source.return_value = mock_source
    return service


@pytest.fixture
async def async_client(
    test_settings: Settings,
    mock_chat_service: AsyncMock,
    mock_knowledge_service: AsyncMock,
) -> AsyncGenerator[AsyncClient, None]:
    """Asynchronous HTTP test client with mocked dependencies."""
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service
    app.dependency_overrides[get_knowledge_service] = lambda: mock_knowledge_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
