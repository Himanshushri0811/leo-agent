"""
Knowledge ingestion pipeline.

Flow:
  1. Parse source content (URL or document).
  2. Check for duplicate by content hash.
  3. Chunk the text.
  4. Embed chunks.
  5. Store chunks in pending state.
  6. Source remains in REVIEW status until manually approved.

Ingestion never automatically promotes content to APPROVED.
A separate approval action (not implemented in this phase) promotes sources.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.interfaces import EmbeddingsProvider
from app.core.config import Settings
from app.core.exceptions import KnowledgeIngestionError
from app.core.logging import get_logger
from app.db.models.knowledge import KnowledgeChunk, KnowledgeSource, SourceStatus
from app.knowledge.chunking import chunk_text
from app.knowledge.embeddings import embed_chunks
from app.knowledge.parsing import SourceType, parse_pdf, parse_text, parse_url

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    source_id: uuid.UUID
    source_name: str
    chunks_created: int
    is_duplicate: bool
    status: str


async def ingest_url(
    url: str,
    name: str,
    session: AsyncSession,
    embeddings_provider: EmbeddingsProvider,
    settings: Settings,
) -> IngestionResult:
    """Parse a URL, chunk it, embed it, and store in REVIEW state."""
    try:
        text, content_hash = parse_url(url)
    except Exception as exc:
        raise KnowledgeIngestionError(f"Failed to fetch URL: {url}: {exc}") from exc

    return await _ingest(
        text=text,
        content_hash=content_hash,
        name=name,
        source_type=SourceType.URL,
        source_url=url,
        session=session,
        embeddings_provider=embeddings_provider,
        settings=settings,
    )


from app.knowledge.parsing import (
    SourceType,
    parse_document,
    parse_pdf,
    parse_text,
    parse_url,
)
from sqlalchemy import delete


async def ingest_document(
    path: Path,
    name: str | None = None,
    session: AsyncSession | None = None,
    embeddings_provider: EmbeddingsProvider | None = None,
    settings: Settings | None = None,
    auto_approve: bool = False,
) -> IngestionResult:
    """Parse a document (.md, .txt, .pdf, .docx), chunk it, embed it, and store in DB."""
    if not path.exists():
        raise KnowledgeIngestionError(f"Document file does not exist: {path}")

    doc_name = name or path.stem.replace("_", " ").title()

    try:
        text, content_hash, source_type = parse_document(path)
    except Exception as exc:
        raise KnowledgeIngestionError(f"Failed to parse document: {path}: {exc}") from exc

    metadata = {
        "filename": path.name,
        "file_path": str(path.resolve()),
        "file_size": path.stat().st_size,
        "extension": path.suffix.lower(),
    }

    assert session is not None, "session is required"
    assert embeddings_provider is not None, "embeddings_provider is required"
    assert settings is not None, "settings is required"

    return await _ingest(
        text=text,
        content_hash=content_hash,
        name=doc_name,
        source_type=source_type,
        source_url=None,
        session=session,
        embeddings_provider=embeddings_provider,
        settings=settings,
        auto_approve=auto_approve,
        metadata_=metadata,
    )


async def _ingest(
    *,
    text: str,
    content_hash: str,
    name: str,
    source_type: str,
    source_url: str | None,
    session: AsyncSession,
    embeddings_provider: EmbeddingsProvider,
    settings: Settings,
    auto_approve: bool = False,
    metadata_: dict | None = None,
) -> IngestionResult:
    # 1. Deduplication check by content hash
    existing = await session.execute(
        select(KnowledgeSource).where(KnowledgeSource.content_hash == content_hash)
    )
    dup_source = existing.scalar_one_or_none()
    if dup_source is not None:
        logger.info(
            "ingestion_skipped_duplicate",
            content_hash=content_hash,
            source_id=str(dup_source.id),
        )
        return IngestionResult(
            source_id=dup_source.id,
            source_name=dup_source.name or name,
            chunks_created=0,
            is_duplicate=True,
            status=dup_source.status,
        )

    # 2. Check for updated document with identical name
    target_status = SourceStatus.APPROVED if auto_approve else SourceStatus.REVIEW
    existing_named = None
    if name:
        res = await session.execute(
            select(KnowledgeSource).where(KnowledgeSource.name == name)
        )
        existing_named = res.scalar_one_or_none()

    if existing_named is not None:
        logger.info(
            "ingestion_updating_existing_source",
            source_name=name,
            source_id=str(existing_named.id),
        )
        # Remove previous chunks safely
        await session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.source_id == existing_named.id)
        )

        chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        if not chunks:
            raise KnowledgeIngestionError("Source produced no text after parsing.")

        embeddings = await embed_chunks(chunks, embeddings_provider)

        existing_named.version += 1
        existing_named.content_hash = content_hash
        existing_named.source_type = source_type
        existing_named.source_url = source_url
        existing_named.updated_at = datetime.now(UTC)
        if auto_approve or existing_named.status == SourceStatus.APPROVED:
            existing_named.status = SourceStatus.APPROVED
            existing_named.approved_at = datetime.now(UTC)
            existing_named.approved_by = existing_named.approved_by or "system-auto-approve"
        else:
            existing_named.status = SourceStatus.REVIEW

        meta = existing_named.metadata_ or {}
        if metadata_:
            meta.update(metadata_)
        existing_named.metadata_ = meta

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            db_chunk = KnowledgeChunk(
                source_id=existing_named.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunk.token_estimate,
                embedding=embedding,
                chunk_metadata={
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    **(metadata_ or {}),
                },
            )
            session.add(db_chunk)

        await session.flush()
        return IngestionResult(
            source_id=existing_named.id,
            source_name=name,
            chunks_created=len(chunks),
            is_duplicate=False,
            status=existing_named.status,
        )

    # 3. New source creation
    chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        raise KnowledgeIngestionError("Source produced no text after parsing.")

    embeddings = await embed_chunks(chunks, embeddings_provider)

    source = KnowledgeSource(
        name=name,
        source_type=source_type,
        source_url=source_url,
        content_hash=content_hash,
        status=target_status,
        ingested_at=datetime.now(UTC),
        approved_at=datetime.now(UTC) if target_status == SourceStatus.APPROVED else None,
        approved_by="system-auto-approve" if target_status == SourceStatus.APPROVED else None,
        metadata_=metadata_ or {},
    )
    session.add(source)
    await session.flush()

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        db_chunk = KnowledgeChunk(
            source_id=source.id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            token_count=chunk.token_estimate,
            embedding=embedding,
            chunk_metadata={
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **(metadata_ or {}),
            },
        )
        session.add(db_chunk)

    logger.info(
        "ingestion_complete",
        source_name=name,
        source_id=str(source.id),
        chunks=len(chunks),
        status=target_status,
    )

    return IngestionResult(
        source_id=source.id,
        source_name=name,
        chunks_created=len(chunks),
        is_duplicate=False,
        status=target_status,
    )


async def reindex_all_sources(
    session: AsyncSession,
    embeddings_provider: EmbeddingsProvider,
    settings: Settings,
) -> dict[str, int]:
    """
    Re-embed all existing knowledge chunks using the active embeddings provider.
    Verifies vector dimensionality against settings.embeddings_dimensions before writing.
    Preserves existing source records and IDs without duplicating entries.
    """
    sources_result = await session.execute(
        select(KnowledgeSource).order_by(KnowledgeSource.name)
    )
    sources = sources_result.scalars().all()
    if not sources:
        return {"sources_reindexed": 0, "chunks_reembedded": 0}

    total_chunks_reembedded = 0
    from app.ai.interfaces import EmbeddingRequest

    for source in sources:
        chunks_res = await session.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.source_id == source.id)
            .order_by(KnowledgeChunk.chunk_index)
        )
        chunks = chunks_res.scalars().all()
        if not chunks:
            continue

        texts = [c.content for c in chunks]
        embed_resp = await embeddings_provider.embed(EmbeddingRequest(texts=texts))

        if len(embed_resp.embeddings) != len(chunks):
            raise KnowledgeIngestionError(
                f"Embedding count mismatch for source '{source.name}': "
                f"expected {len(chunks)}, got {len(embed_resp.embeddings)}"
            )

        for chunk, vec in zip(chunks, embed_resp.embeddings, strict=True):
            if len(vec) != settings.embeddings_dimensions:
                raise KnowledgeIngestionError(
                    f"Vector dimension mismatch: embedding provider returned {len(vec)} dimensions, "
                    f"but database column expects {settings.embeddings_dimensions}. "
                    "Ensure embedding model output dimension matches PostgreSQL schema before indexing."
                )
            chunk.embedding = vec
            total_chunks_reembedded += 1

        source.updated_at = datetime.now(UTC)

    await session.flush()
    logger.info(
        "reindex_complete",
        sources_count=len(sources),
        chunks_count=total_chunks_reembedded,
        provider=embeddings_provider.provider_name,
    )
    return {
        "sources_reindexed": len(sources),
        "chunks_reembedded": total_chunks_reembedded,
    }
