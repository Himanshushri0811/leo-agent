"""
Semantic retrieval using pgvector.

Queries only APPROVED knowledge chunks. Pending, rejected, or archived
chunks are never returned to the AI layer.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.interfaces import EmbeddingRequest, EmbeddingsProvider
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.models.knowledge import SourceStatus
from app.db.session import get_session_factory

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: uuid.UUID
    source_name: str
    source_url: str | None
    content: str
    similarity_score: float
    chunk_index: int


async def retrieve_relevant_chunks(
    query: str,
    embeddings_provider: EmbeddingsProvider,
    settings: Settings,
    session: AsyncSession | None = None,
) -> list[RetrievalResult]:
    """
    Embed the query and perform cosine similarity search against approved chunks.

    Returns at most `max_retrieval_results` results above `min_similarity_score`.
    Falls back to an empty list if no database session is available (e.g., mock mode).
    """
    import time

    t_retrieval_start = time.perf_counter()

    # 1. Embed query
    t_embed_start = time.perf_counter()
    embed_resp = await embeddings_provider.embed(EmbeddingRequest(texts=[query]))
    embed_duration_ms = (time.perf_counter() - t_embed_start) * 1000.0

    if not embed_resp.embeddings:
        logger.info(
            "timing_knowledge_retrieval",
            embed_duration_ms=round(embed_duration_ms, 2),
            query_duration_ms=0.0,
            total_duration_ms=round((time.perf_counter() - t_retrieval_start) * 1000.0, 2),
            chunks_found=0,
        )
        return []
    query_vector = embed_resp.embeddings[0]

    # 2. Build vector literal — float-only, safe to embed directly in SQL
    vector_literal = f"[{','.join(str(v) for v in query_vector)}]"

    async def _query(sess: AsyncSession) -> list[RetrievalResult]:
        # Detect if query has Hindi script or Hinglish markers for cross-lingual threshold adjustment
        import re
        from app.voice.language import _HINGLISH_MARKERS
        has_indic_script = bool(re.search(r"[\u0900-\u0D7F]", query))
        has_hinglish = any(w.strip(".,!?:;\"'").lower() in _HINGLISH_MARKERS for w in query.split())
        is_cross_lingual = has_indic_script or has_hinglish

        if settings.embeddings_provider == "mock":
            effective_min_score = min(settings.min_similarity_score, 0.10)
        elif is_cross_lingual:
            # Cross-lingual embedding similarity (e.g. Hindi query vs English doc) has lower baseline cosine similarity
            effective_min_score = min(settings.min_similarity_score, 0.45)
        else:
            # Voice queries often have conversational phrasing; ensure threshold allows relevant matches
            effective_min_score = min(settings.min_similarity_score, 0.55)

        # asyncpg converts :param -> $N positional args, which breaks the ::vector
        # type-cast syntax. We embed the float vector literal directly into SQL
        # (it contains only numbers/brackets, no user input) and use CAST() for
        # the filter comparison to avoid the named-param + cast conflict.
        sql = f"""
            SELECT
                kc.id,
                kc.content,
                kc.chunk_index,
                ks.name        AS source_name,
                ks.source_url,
                1 - (kc.embedding <=> '{vector_literal}'::vector) AS similarity
            FROM knowledge_chunks kc
            JOIN knowledge_sources ks ON ks.id = kc.source_id
            WHERE
                ks.status = :approved_status
                AND kc.embedding IS NOT NULL
                AND 1 - (kc.embedding <=> '{vector_literal}'::vector) >= :min_score
            ORDER BY kc.embedding <=> '{vector_literal}'::vector
            LIMIT :limit
        """
        result = await sess.execute(
            text(sql),
            {
                "approved_status": SourceStatus.APPROVED,
                "min_score": effective_min_score,
                "limit": settings.max_retrieval_results,
            },
        )
        rows = result.mappings().all()
        if not rows:
            return []

        # Relevance optimization: prune lower-relevance chunks when a clear top match is found
        top_score = float(rows[0]["similarity"])
        margin = 0.018 if settings.embeddings_provider == "mock" else 0.15
        dynamic_cutoff = max(effective_min_score, top_score - margin)

        filtered_rows = [
            row for row in rows
            if float(row["similarity"]) >= dynamic_cutoff
        ]

        return [
            RetrievalResult(
                chunk_id=row["id"],
                source_name=row["source_name"],
                source_url=row["source_url"],
                content=row["content"],
                similarity_score=float(row["similarity"]),
                chunk_index=row["chunk_index"],
            )
            for row in filtered_rows
        ]

    t_db_start = time.perf_counter()
    try:
        async with asyncio.timeout(3.5):
            if session is not None:
                results = await _query(session)
            else:
                factory = get_session_factory()
                async with factory() as sess:
                    results = await _query(sess)
        query_duration_ms = (time.perf_counter() - t_db_start) * 1000.0
        total_retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000.0
        source_identifiers = list({f"{r.source_name} (chunk {r.chunk_index})" for r in results})
        logger.info(
            "timing_knowledge_retrieval",
            embed_duration_ms=round(embed_duration_ms, 2),
            query_duration_ms=round(query_duration_ms, 2),
            total_duration_ms=round(total_retrieval_ms, 2),
            chunks_found=len(results),
            source_identifiers=source_identifiers,
            top_similarity=round(results[0].similarity_score, 4) if results else None,
        )
        return results
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "timing_knowledge_retrieval_timeout",
            embed_duration_ms=round(embed_duration_ms, 2),
            query_duration_ms=round((time.perf_counter() - t_db_start) * 1000.0, 2),
        )
        return []
    except RuntimeError:
        # Database not initialised (mock / test mode without DB)
        logger.warning("retrieval_skipped_no_db")
        return []
    except Exception as exc:
        logger.error(
            "retrieval_error",
            error=str(exc),
            embed_duration_ms=round(embed_duration_ms, 2),
            query_duration_ms=round((time.perf_counter() - t_db_start) * 1000.0, 2),
        )
        return []

