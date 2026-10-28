"""
Embedding generation for knowledge chunks.

Wraps the embeddings provider interface and handles batching.
"""

from __future__ import annotations

from app.ai.interfaces import EmbeddingRequest, EmbeddingsProvider
from app.core.logging import get_logger
from app.knowledge.chunking import TextChunk

logger = get_logger(__name__)

EMBED_BATCH_SIZE = 50  # Number of chunks per embedding API call


async def embed_chunks(
    chunks: list[TextChunk],
    provider: EmbeddingsProvider,
) -> list[list[float]]:
    """
    Embed a list of text chunks in batches.

    Returns a list of embedding vectors, one per chunk, in the same order.
    """
    if not chunks:
        return []

    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        texts = [c.content for c in batch]

        logger.debug(
            "embedding_batch",
            batch_start=batch_start,
            batch_size=len(batch),
        )

        response = await provider.embed(EmbeddingRequest(texts=texts))
        all_embeddings.extend(response.embeddings)

    return all_embeddings
