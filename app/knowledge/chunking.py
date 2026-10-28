"""
Configurable text chunking strategies.

Chunks are produced from normalised text. The chunker is stateless
and testable independently of any database or AI provider.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    token_estimate: int


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[TextChunk]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text: Normalised plain text to chunk.
        chunk_size: Target chunk size in words.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        Ordered list of TextChunk objects.
    """
    if not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    step = max(chunk_size - overlap, 1)
    chunks: list[TextChunk] = []
    idx = 0
    char_pos = 0

    while idx < len(words):
        end_idx = min(idx + chunk_size, len(words))
        chunk_words = words[idx:end_idx]
        content = " ".join(chunk_words)

        # Approximate character positions (not exact but useful for metadata)
        start_char = text.find(chunk_words[0], char_pos)
        end_char = start_char + len(content)

        chunks.append(
            TextChunk(
                content=content,
                chunk_index=len(chunks),
                start_char=max(start_char, 0),
                end_char=end_char,
                token_estimate=len(chunk_words),  # ~1 token per word approximation
            )
        )

        if end_idx == len(words):
            break
        char_pos = start_char
        idx += step

    return chunks
