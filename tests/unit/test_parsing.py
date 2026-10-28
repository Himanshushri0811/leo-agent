"""Unit tests for document parsing and text normalisation."""

from app.knowledge.parsing import parse_text


def test_parse_text_normalisation() -> None:
    raw = "  Bitmax \x00 Group \n\n\n\n Cloud   Solutions.  "
    normalized, content_hash = parse_text(raw)

    assert "\x00" not in normalized
    assert "\n\n\n" not in normalized
    assert "   " not in normalized
    assert normalized == "Bitmax Group \n\n Cloud Solutions."
    assert len(content_hash) == 64  # SHA-256


def test_parse_text_hash_is_deterministic() -> None:
    text1, hash1 = parse_text("Identical content test")
    text2, hash2 = parse_text("Identical content test")
    assert hash1 == hash2
