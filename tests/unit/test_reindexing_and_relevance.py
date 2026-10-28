"""Unit tests for knowledge re-indexing, dimension safety, and RAG relevance filtering."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.ai.interfaces import EmbeddingResponse
from app.core.config import AppEnv, Settings
from app.core.exceptions import KnowledgeIngestionError
from app.db.models.knowledge import KnowledgeChunk, KnowledgeSource
from app.knowledge.ingestion import reindex_all_sources
from app.knowledge.retrieval import RetrievalResult, retrieve_relevant_chunks


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        app_env=AppEnv.TESTING,
        app_secret_key="secret-key-at-least-32-chars-long",
        admin_api_key="valid-admin-key",
        database_url="postgresql+asyncpg://user:pass@localhost:5432/db",
        embeddings_dimensions=1536,
        min_similarity_score=0.70,
    )


@pytest.mark.asyncio
async def test_reindex_dimension_mismatch_raises_error(test_settings: Settings) -> None:
    # Source & chunk in DB
    mock_source = KnowledgeSource(
        id=uuid.uuid4(),
        name="Test Doc",
        source_type="document",
        status="approved",
    )
    mock_chunk = KnowledgeChunk(
        id=uuid.uuid4(),
        source_id=mock_source.id,
        chunk_index=0,
        content="Test content",
        embedding=[0.0] * 1536,
    )

    mock_session = AsyncMock()
    mock_sources_result = MagicMock()
    mock_sources_result.scalars.return_value.all.return_value = [mock_source]
    mock_chunks_result = MagicMock()
    mock_chunks_result.scalars.return_value.all.return_value = [mock_chunk]

    mock_session.execute.side_effect = [mock_sources_result, mock_chunks_result]

    # Incompatible provider returning 768 dimensions instead of 1536
    incompatible_provider = MagicMock()
    incompatible_provider.provider_name = "incompatible_test"
    incompatible_provider.embed = AsyncMock(
        return_value=EmbeddingResponse(
            embeddings=[[0.1] * 768],
            model="768-dim-model",
            token_count=10,
            provider="incompatible_test",
        )
    )

    with pytest.raises(KnowledgeIngestionError, match="Vector dimension mismatch"):
        await reindex_all_sources(
            session=mock_session,
            embeddings_provider=incompatible_provider,
            settings=test_settings,
        )


@pytest.mark.asyncio
async def test_reindex_success_updates_chunks(test_settings: Settings) -> None:
    mock_source = KnowledgeSource(
        id=uuid.uuid4(),
        name="Company Overview",
        source_type="document",
        status="approved",
    )
    mock_chunk = KnowledgeChunk(
        id=uuid.uuid4(),
        source_id=mock_source.id,
        chunk_index=0,
        content="Bitmax Group overview",
        embedding=[0.0] * 1536,
    )

    mock_session = AsyncMock()
    mock_sources_result = MagicMock()
    mock_sources_result.scalars.return_value.all.return_value = [mock_source]
    mock_chunks_result = MagicMock()
    mock_chunks_result.scalars.return_value.all.return_value = [mock_chunk]

    mock_session.execute.side_effect = [mock_sources_result, mock_chunks_result]

    matching_provider = MagicMock()
    matching_provider.provider_name = "matching_1536"
    new_vector = [0.05] * 1536
    matching_provider.embed = AsyncMock(
        return_value=EmbeddingResponse(
            embeddings=[new_vector],
            model="1536-model",
            token_count=5,
            provider="matching_1536",
        )
    )

    result = await reindex_all_sources(
        session=mock_session,
        embeddings_provider=matching_provider,
        settings=test_settings,
    )

    assert result["sources_reindexed"] == 1
    assert result["chunks_reembedded"] == 1
    assert mock_chunk.embedding == new_vector
    mock_session.flush.assert_called_once()
