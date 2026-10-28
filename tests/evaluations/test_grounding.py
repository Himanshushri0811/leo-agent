"""Evaluation tests: Grounding, evidence boundary enforcement, and source traceability."""

import uuid

from app.ai.interfaces import Message
from app.ai.orchestrator import _format_passages, _trim_history
from app.knowledge.retrieval import RetrievalResult


def test_format_passages_includes_citations() -> None:
    results = [
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            source_name="Company Overview",
            source_url="https://bitmaxgroup.com/about",
            content="Bitmax Group provides cloud solutions.",
            similarity_score=0.92,
            chunk_index=0,
        ),
        RetrievalResult(
            chunk_id=uuid.uuid4(),
            source_name="Services Guide",
            source_url=None,
            content="We offer managed Kubernetes.",
            similarity_score=0.85,
            chunk_index=1,
        ),
    ]

    formatted = _format_passages(results)
    assert "[1] (Source: https://bitmaxgroup.com/about)" in formatted
    assert "Bitmax Group provides cloud solutions." in formatted
    assert "[2] (Source: Services Guide)" in formatted
    assert "We offer managed Kubernetes." in formatted


def test_trim_history_bounds_context_size() -> None:
    # Generate 20 turns
    history = [
        Message(role="user" if i % 2 == 0 else "assistant", content=f"Turn {i}") for i in range(20)
    ]
    trimmed = _trim_history(history)
    # MAX_HISTORY_TURNS is 6 -> 12 messages maximum
    assert len(trimmed) == 12
    assert trimmed[-1].content == "Turn 19"
