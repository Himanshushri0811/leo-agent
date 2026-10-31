"""Unit tests for system prompts and prompt assembly."""

from app.ai.prompts.system import SYSTEM_PROMPT_VERSION, build_system_prompt


def test_system_prompt_version_exists() -> None:
    assert isinstance(SYSTEM_PROMPT_VERSION, str)
    assert len(SYSTEM_PROMPT_VERSION) > 0


def test_build_system_prompt_injects_passages() -> None:
    passages = "[1] Bitmax Group founded in 2020."
    prompt = build_system_prompt(passages)
    assert passages in prompt
    assert "You are LEO" in prompt
    assert "<<<BEGIN_REFERENCE_DATA>>>" in prompt


def test_build_system_prompt_handles_empty_passages() -> None:
    prompt = build_system_prompt("")
    assert "(none)" in prompt
