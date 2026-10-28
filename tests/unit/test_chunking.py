"""Unit tests for text chunking algorithms."""

from app.knowledge.chunking import chunk_text


def test_chunking_empty_string() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunking_short_text() -> None:
    text = "Bitmax Group provides cloud solutions."
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == text
    assert chunks[0].token_estimate == 5


def test_chunking_long_text_creates_overlap() -> None:
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=30, overlap=10)

    assert len(chunks) > 1
    # Check that sequential chunks have overlapping content
    chunk0_words = set(chunks[0].content.split())
    chunk1_words = set(chunks[1].content.split())
    overlap = chunk0_words.intersection(chunk1_words)
    assert len(overlap) >= 10
