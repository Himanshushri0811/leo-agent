"""Unit tests for system prompts and prompt assembly."""

import pytest
from app.ai.prompts.system import (
    GLOBAL_PROJECTS_CORE_PROFILE,
    SYSTEM_PROMPT_VERSION,
    build_system_prompt,
)


def test_system_prompt_version_exists() -> None:
    assert isinstance(SYSTEM_PROMPT_VERSION, str)
    assert len(SYSTEM_PROMPT_VERSION) > 0


def test_build_system_prompt_injects_passages() -> None:
    passages = "[1] Global Projects verified service."
    prompt = build_system_prompt(passages)
    assert passages in prompt
    assert "You are LEO" in prompt
    assert "<<<BEGIN_REFERENCE_DATA>>>" in prompt


def test_build_system_prompt_handles_empty_passages() -> None:
    prompt = build_system_prompt("")
    assert "(none)" in prompt


def test_global_projects_identity_and_scope() -> None:
    prompt = build_system_prompt("")
    assert "Global Projects" in prompt
    assert "dedicated exclusively to Global Projects" in prompt
    assert GLOBAL_PROJECTS_CORE_PROFILE in prompt


def test_bitmax_strict_rejection_rules() -> None:
    prompt = build_system_prompt("")
    assert "I'm sorry, but I can assist only with Global Projects-related queries." in prompt
    assert "Maaf kijiye, main sirf Global Projects se related jaankari de sakta hoon." in prompt
    assert "FINAL NON-NEGOTIABLE RULE:" in prompt
    assert "Never provide Bitmax-related information or answers." in prompt


def test_multilingual_scope() -> None:
    prompt = build_system_prompt("")
    expected_langs = [
        "English", "Hindi", "Hinglish", "Spanish", "French", "German",
        "Arabic", "Portuguese", "Bengali", "Tamil", "Telugu", "Marathi",
        "Kannada", "Malayalam"
    ]
    for lang in expected_langs:
        assert lang in prompt


def test_voice_and_length_guidelines() -> None:
    prompt = build_system_prompt("")
    assert "Simple questions: 1–3 sentences." in prompt
    assert "Normal questions: 3–5 sentences." in prompt
    assert "Write for the ear: plain text only." in prompt
