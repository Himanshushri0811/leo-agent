"""One chat turn must make exactly ONE LLM provider call."""
import asyncio
import types

import pytest

from app.ai import orchestrator
from app.ai.interfaces import GenerationResponse
from app.core.exceptions import ProviderError


class FakeLLM:
    provider_name = "fake"
    _model = "fake-model"

    def __init__(self, fail=None):
        self.generate_calls = 0
        self.stream_calls = 0
        self.fail = fail

    @property
    def total_calls(self):
        return self.generate_calls + self.stream_calls

    async def generate(self, req):
        self.generate_calls += 1
        if self.fail:
            raise self.fail
        return GenerationResponse(content="hello", model="fake-model", prompt_tokens=11,
                                  completion_tokens=2, finish_reason="stop", provider="fake")

    async def generate_stream(self, req):
        self.stream_calls += 1
        if self.fail:
            raise self.fail
        yield "hel"
        yield "lo"


async def _no_chunks(**kwargs):
    return []


def _run(llm, monkeypatch):
    monkeypatch.setattr(orchestrator, "retrieve_relevant_chunks", _no_chunks)
    req = orchestrator.ChatRequest(session_id="s", user_message="hi")
    settings = types.SimpleNamespace(llm_max_tokens=100, llm_temperature=0.2)
    return asyncio.run(orchestrator.run_chat(req, llm, object(), settings))


def test_successful_turn_makes_exactly_one_provider_call(monkeypatch):
    llm = FakeLLM()
    resp = _run(llm, monkeypatch)
    assert llm.total_calls == 1, f"expected 1 provider call, got {llm.total_calls} (stream={llm.stream_calls}, generate={llm.generate_calls})"
    assert resp.response_text == "hello"
    assert resp.prompt_tokens == 11, "real token usage must be preserved"


def test_provider_failure_is_not_retried_by_orchestrator(monkeypatch):
    llm = FakeLLM(fail=ProviderError("Gemini quota/rate limit exceeded (HTTP 429)"))
    with pytest.raises(ProviderError):
        _run(llm, monkeypatch)
    assert llm.total_calls == 1, f"failed turn made {llm.total_calls} provider calls; retries belong to the provider, under one deadline"