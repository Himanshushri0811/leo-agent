"""
Integration test for knowledge ingestion deduplication and approval flow.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest
from app.core.config import Settings
from app.db.models.knowledge import KnowledgeSource, SourceStatus
from app.knowledge.ingestion import _ingest


@pytest.mark.asyncio
async def test_ingest_deduplication_returns_existing_source_id():
    """Verify that duplicate content hashes return the real existing source ID without crashing."""
    session = AsyncMock()
    embeddings_provider = AsyncMock()
    settings = Settings(
        app_env="testing",
        app_secret_key="test-secret-key-for-unit-testing-32-chars-min",
        admin_api_key="test-admin-api-key",
        llm_provider="mock",
        embeddings_provider="mock",
    )

    existing_id = uuid.uuid4()
    mock_source = MagicMock(spec=KnowledgeSource)
    mock_source.id = existing_id
    mock_source.status = SourceStatus.REVIEW

    # Mock execute result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_source
    session.execute.return_value = mock_result

    result = await _ingest(
        text="Bitmax Group enterprise knowledge sample.",
        content_hash="test_sha256_hash_12345",
        name="Bitmax Duplicate Test Doc",
        source_type="text",
        source_url=None,
        session=session,
        embeddings_provider=embeddings_provider,
        settings=settings,
    )

    assert result.is_duplicate is True
    assert result.chunks_created == 0
    # Must match the real existing source ID, not a newly generated random uuid
    assert result.source_id == existing_id
    # Verify scalar_one_or_none was called only once on mock_result
    assert mock_result.scalar_one_or_none.call_count == 1
