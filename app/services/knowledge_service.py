"""
Application business service for managing knowledge ingestion and review workflows.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.providers import get_embeddings_provider
from app.core.config import Settings, get_settings
from app.core.exceptions import NotFoundError, ValidationError
from app.db.models.knowledge import KnowledgeSource, SourceStatus
from app.db.session import get_async_session
from app.knowledge.ingestion import IngestionResult, ingest_document, ingest_url
from app.repositories.knowledge_repository import KnowledgeRepository


class KnowledgeService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings
        self._repo = KnowledgeRepository(session)
        self._embeddings = get_embeddings_provider(settings)

    async def register_source(
        self, name: str, source_url: str | None = None, notes: str = ""
    ) -> KnowledgeSource:
        source = KnowledgeSource(
            name=name,
            source_type="url" if source_url else "manual",
            source_url=source_url,
            status=SourceStatus.PENDING,
            metadata_={"notes": notes},
        )
        self._session.add(source)
        await self._session.flush()
        return source

    async def ingest_url_source(self, url: str, name: str) -> IngestionResult:
        return await ingest_url(
            url=url,
            name=name,
            session=self._session,
            embeddings_provider=self._embeddings,
            settings=self._settings,
        )

    async def ingest_file_source(
        self, path: Path, name: str, auto_approve: bool = False
    ) -> IngestionResult:
        if not path.exists():
            raise NotFoundError(f"File not found: {path}")
        return await ingest_document(
            path=path,
            name=name,
            session=self._session,
            embeddings_provider=self._embeddings,
            settings=self._settings,
            auto_approve=auto_approve,
        )

    async def approve_source(self, source_id: uuid.UUID, approved_by: str) -> KnowledgeSource:
        source = await self._repo.get_source_by_id(source_id)
        if not source:
            raise NotFoundError(f"KnowledgeSource {source_id} not found.")
        if source.status != SourceStatus.REVIEW:
            raise ValidationError(
                f"Source in status '{source.status}' cannot be approved. Must be 'review'."
            )
        updated = await self._repo.update_status(
            source_id=source_id,
            new_status=SourceStatus.APPROVED,
            approved_by=approved_by,
        )
        assert updated is not None
        return updated

    async def reject_source(self, source_id: uuid.UUID) -> KnowledgeSource:
        source = await self._repo.get_source_by_id(source_id)
        if not source:
            raise NotFoundError(f"KnowledgeSource {source_id} not found.")
        updated = await self._repo.update_status(
            source_id=source_id, new_status=SourceStatus.REJECTED
        )
        assert updated is not None
        return updated

    async def list_sources(self, status: str | None = None) -> Sequence[KnowledgeSource]:
        return await self._repo.list_sources(status=status)


def get_knowledge_service(
    session: AsyncSession = Depends(get_async_session),
    settings: Settings = Depends(get_settings),
) -> KnowledgeService:
    return KnowledgeService(session=session, settings=settings)
