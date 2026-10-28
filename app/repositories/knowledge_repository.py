"""
Repository for KnowledgeSource and KnowledgeChunk entities.
Encapsulates database operations and approval workflows for company knowledge.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.knowledge import KnowledgeSource, SourceStatus


class KnowledgeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_source_by_id(self, source_id: uuid.UUID) -> KnowledgeSource | None:
        stmt = (
            select(KnowledgeSource)
            .where(KnowledgeSource.id == source_id)
            .options(selectinload(KnowledgeSource.chunks))
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_source_by_hash(self, content_hash: str) -> KnowledgeSource | None:
        stmt = select(KnowledgeSource).where(KnowledgeSource.content_hash == content_hash)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_sources(
        self, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> Sequence[KnowledgeSource]:
        stmt = select(KnowledgeSource)
        if status:
            stmt = stmt.where(KnowledgeSource.status == status)
        stmt = stmt.order_by(KnowledgeSource.created_at.desc()).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def update_status(
        self, source_id: uuid.UUID, new_status: str, approved_by: str | None = None
    ) -> KnowledgeSource | None:
        source = await self.get_source_by_id(source_id)
        if not source:
            return None

        source.status = new_status
        if new_status == SourceStatus.APPROVED:
            source.approved_at = datetime.now(UTC)
            source.approved_by = approved_by
        await self._session.flush()
        return source

    async def delete_source(self, source_id: uuid.UUID) -> bool:
        source = await self.get_source_by_id(source_id)
        if not source:
            return False
        await self._session.delete(source)
        await self._session.flush()
        return True
