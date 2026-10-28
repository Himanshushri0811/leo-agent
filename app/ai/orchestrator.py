"""
Conversation orchestrator.

Coordinates: retrieval → prompt assembly → LLM generation → response.

The orchestrator is deliberately independent of HTTP request objects.
It accepts plain Python data and returns plain Python data.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Union

from app.ai.interfaces import EmbeddingsProvider, GenerationRequest, LLMProvider, Message
from app.ai.prompts.system import build_system_prompt
from app.core.config import Settings
from app.core.logging import get_logger
from app.knowledge.retrieval import RetrievalResult, retrieve_relevant_chunks

logger = get_logger(__name__)

MAX_HISTORY_TURNS = 6  # Keep last N user+assistant pairs to bound context size


@dataclass
class ChatRequest:
    session_id: str
    user_message: str
    conversation_history: list[Message] = field(default_factory=list)


@dataclass
class SourceReference:
    chunk_id: str
    source_name: str
    source_url: str | None
    similarity_score: float
    content_excerpt: str


import time

@dataclass
class ChatResponse:
    session_id: str
    response_text: str
    sources: list[SourceReference]
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    prompt_version: str
    retrieval_duration_ms: float | None = None
    llm_duration_ms: float | None = None
    ttft_ms: float | None = None
    total_duration_ms: float | None = None


async def run_chat(
    request: ChatRequest,
    llm: LLMProvider,
    embeddings: EmbeddingsProvider,
    settings: Settings,
) -> ChatResponse:
    """
    End-to-end RAG chat pipeline:

    1. Embed the user query.
    2. Retrieve relevant approved knowledge chunks.
    3. Assemble system prompt with retrieved passages.
    4. Build message history (bounded).
    5. Call LLM provider (measures TTFT via streaming if supported).
    6. Return structured response with source attribution.
    """
    t_chat_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    logger.info(
        "chat_request_received",
        session_id=request.session_id,
        request_id=request_id,
        # Message content is not logged to protect user privacy
        message_length=len(request.user_message),
    )

    # 1. Retrieve relevant knowledge
    t_retrieval_start = time.perf_counter()
    retrieval_results: list[RetrievalResult] = await retrieve_relevant_chunks(
        query=request.user_message,
        embeddings_provider=embeddings,
        settings=settings,
    )
    retrieval_duration_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

    # 2. Format passages for the system prompt
    passages = _format_passages(retrieval_results)

    # 3. Build messages
    system_prompt = build_system_prompt(passages)
    history = _trim_history(request.conversation_history)

    messages: list[Message] = [
        Message(role="system", content=system_prompt),
        *history,
        Message(role="user", content=request.user_message),
    ]

    # 4. Generate
    gen_request = GenerationRequest(
        messages=messages,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )

    t_gen_start = time.perf_counter()
    ttft_ms: float | None = None  # not measured: generation is one non-streaming call

    # Exactly ONE provider call per turn.
    # Previously this block streamed the reply, hit a NameError (GenerationResponse was
    # never imported), swallowed it with `except Exception`, and silently called
    # generate() a second time: double latency, double quota, and a second full
    # deadline on failures. The provider owns retry/backoff under a single total
    # deadline (llm_timeout_seconds); errors now propagate to the caller untouched.
    gen_response = await llm.generate(gen_request)

    llm_duration_ms = (time.perf_counter() - t_gen_start) * 1000.0
    total_duration_ms = (time.perf_counter() - t_chat_start) * 1000.0

    logger.info(
        "timing_rag_pipeline",
        session_id=request.session_id,
        request_id=request_id,
        provider=gen_response.provider,
        model=gen_response.model,
        retrieval_duration_ms=round(retrieval_duration_ms, 2),
        ttft_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
        llm_duration_ms=round(llm_duration_ms, 2),
        total_duration_ms=round(total_duration_ms, 2),
        sources_count=len(retrieval_results),
    )

    sources = [
        SourceReference(
            chunk_id=str(r.chunk_id),
            source_name=r.source_name,
            source_url=r.source_url,
            similarity_score=r.similarity_score,
            content_excerpt=r.content[:200],
        )
        for r in retrieval_results
    ]

    from app.ai.prompts.system import SYSTEM_PROMPT_VERSION

    return ChatResponse(
        session_id=request.session_id,
        response_text=gen_response.content,
        sources=sources,
        model=gen_response.model,
        provider=gen_response.provider,
        prompt_tokens=gen_response.prompt_tokens,
        completion_tokens=gen_response.completion_tokens,
        prompt_version=SYSTEM_PROMPT_VERSION,
        retrieval_duration_ms=round(retrieval_duration_ms, 2),
        llm_duration_ms=round(llm_duration_ms, 2),
        ttft_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
        total_duration_ms=round(total_duration_ms, 2),
    )


async def run_chat_streaming(
    request: ChatRequest,
    llm: LLMProvider,
    embeddings: EmbeddingsProvider,
    settings: Settings,
) -> AsyncGenerator[Union[str, ChatResponse], None]:
    """
    Sentence-streaming RAG chat pipeline for low-latency LLM→TTS pipelining.

    Yields:
      - str: Each complete sentence/phrase as soon as it is produced by the LLM.
             The caller should immediately start TTS synthesis on each sentence.
      - ChatResponse: Exactly one final ChatResponse object (always the last item).
             Contains the full response text, sources, and timing metadata.

    This cuts time-to-first-audio from ~2,800 ms (full LLM wait) to ~600–900 ms
    (time to first sentence completion from LLM token stream).
    """
    from app.voice.streaming import SentenceAccumulator

    t_chat_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    logger.info(
        "chat_streaming_request_received",
        session_id=request.session_id,
        request_id=request_id,
        message_length=len(request.user_message),
    )

    # 1. Retrieve knowledge in parallel with prompt assembly
    t_retrieval_start = time.perf_counter()
    retrieval_results: list[RetrievalResult] = await retrieve_relevant_chunks(
        query=request.user_message,
        embeddings_provider=embeddings,
        settings=settings,
    )
    retrieval_duration_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

    # 2. Build messages
    passages = _format_passages(retrieval_results)
    system_prompt = build_system_prompt(passages)
    history = _trim_history(request.conversation_history)

    messages: list[Message] = [
        Message(role="system", content=system_prompt),
        *history,
        Message(role="user", content=request.user_message),
    ]

    gen_request = GenerationRequest(
        messages=messages,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )

    # 3. Stream tokens through sentence accumulator
    accumulator = SentenceAccumulator(max_phrase_chars=120)
    full_text_parts: list[str] = []
    ttft_ms: float | None = None
    t_gen_start = time.perf_counter()

    # Check if the LLM provider supports streaming
    stream_fn = getattr(llm, "generate_stream", None)
    has_streaming = stream_fn is not None
    tokens_yielded = 0

    if has_streaming:
        try:
            async for token in llm.generate_stream(gen_request):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_gen_start) * 1000.0
                full_text_parts.append(token)
                tokens_yielded += 1
                for sentence in accumulator.add(token):
                    if sentence:
                        yield sentence

            # Flush any remaining text
            remainder = accumulator.flush()
            if remainder:
                yield remainder

        except Exception as exc:
            category = "timeout" if "timeout" in str(exc).lower() or "deadline" in str(exc).lower() else "provider_error"
            # Fall back to non-streaming ONLY if no response content has been committed to the client
            if tokens_yielded == 0:
                logger.warning(
                    "llm_stream_fallback",
                    provider=getattr(llm, "provider_name", "gemini"),
                    model=getattr(llm, "_model", "gemini"),
                    failure_category=category,
                    fallback_outcome="non_streaming_attempted",
                    error=str(exc)[:150],
                )
                try:
                    gen_response = await llm.generate(gen_request)
                    full_text_parts = [gen_response.content]
                    yield gen_response.content
                except Exception as fb_exc:
                    logger.error(
                        "llm_stream_fallback_failed",
                        provider=getattr(llm, "provider_name", "gemini"),
                        model=getattr(llm, "_model", "gemini"),
                        failure_category=category,
                        fallback_outcome="recovery_message_returned",
                        error=str(fb_exc)[:150],
                    )
                    recovery_text = get_recovery_message(request.user_message)
                    full_text_parts = [recovery_text]
                    yield recovery_text
            else:
                # Part of the response was already sent; do not replay entire response from start
                logger.warning(
                    "llm_stream_interrupted_after_commit",
                    tokens_yielded=tokens_yielded,
                    failure_category=category,
                    fallback_outcome="aborted_without_duplicate",
                    error=str(exc)[:150],
                )
    else:
        # Provider does not support streaming; return full response as single sentence
        try:
            gen_response = await llm.generate(gen_request)
            full_text_parts = [gen_response.content]
            yield gen_response.content
        except Exception as exc:
            logger.error(
                "llm_generate_non_streaming_failed",
                provider=getattr(llm, "provider_name", "gemini"),
                model=getattr(llm, "_model", "gemini"),
                error=str(exc)[:150],
            )
            recovery_text = get_recovery_message(request.user_message)
            full_text_parts = [recovery_text]
            yield recovery_text

    llm_duration_ms = (time.perf_counter() - t_gen_start) * 1000.0
    total_duration_ms = (time.perf_counter() - t_chat_start) * 1000.0
    full_text = "".join(full_text_parts)

    logger.info(
        "timing_rag_streaming_pipeline",
        session_id=request.session_id,
        request_id=request_id,
        retrieval_duration_ms=round(retrieval_duration_ms, 2),
        ttft_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
        llm_duration_ms=round(llm_duration_ms, 2),
        total_duration_ms=round(total_duration_ms, 2),
        sources_count=len(retrieval_results),
    )

    sources = [
        SourceReference(
            chunk_id=str(r.chunk_id),
            source_name=r.source_name,
            source_url=r.source_url,
            similarity_score=r.similarity_score,
            content_excerpt=r.content[:200],
        )
        for r in retrieval_results
    ]

    from app.ai.prompts.system import SYSTEM_PROMPT_VERSION

    # Build a synthetic GenerationResponse for metadata (streaming doesn't expose token counts)
    # Token count is estimated; callers should not rely on it for billing.
    estimated_tokens = max(1, len(full_text.split()))

    yield ChatResponse(
        session_id=request.session_id,
        response_text=full_text,
        sources=sources,
        model=getattr(llm, "_model", "gemini"),
        provider=getattr(llm, "provider_name", "gemini"),
        prompt_tokens=0,  # Not available from streaming path
        completion_tokens=estimated_tokens,
        prompt_version=SYSTEM_PROMPT_VERSION,
        retrieval_duration_ms=round(retrieval_duration_ms, 2),
        llm_duration_ms=round(llm_duration_ms, 2),
        ttft_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
        total_duration_ms=round(total_duration_ms, 2),
    )


def _format_passages(results: list[RetrievalResult]) -> str:
    if not results:
        return ""
    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        ref = r.source_url or r.source_name
        parts.append(f"[{i}] (Source: {ref})\n{r.content}")
    return "\n\n".join(parts)


def _trim_history(history: list[Message]) -> list[Message]:
    """Keep only the most recent N turns to bound context size."""
    # Each turn = 1 user + 1 assistant message
    max_messages = MAX_HISTORY_TURNS * 2
    return history[-max_messages:] if len(history) > max_messages else history


def get_recovery_message(user_message: str) -> str:
    """Localized recovery message when LLM provider times out or fails."""
    try:
        from app.voice.language import LanguageDetector, SupportedLanguage
        det = LanguageDetector.detect_utterance(user_message)
        if det.detected_language == SupportedLanguage.HINDI:
            return "माफ़ कीजिये, मुझे एक तकनीकी समस्या आ रही है। क्या आप कृपया अपना सवाल दोहरा सकते हैं?"
        elif det.detected_language == SupportedLanguage.HINGLISH:
            return "Sorry, mujhe ek brief technical issue aa raha hai. Kya aap please apna question repeat kar sakte hain?"
    except Exception:
        pass
    return "I'm sorry, I'm having a brief technical issue. Could you please repeat your question?"
