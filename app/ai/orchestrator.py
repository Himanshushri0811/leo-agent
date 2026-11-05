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
from typing import Any, Union
from app.ai.interfaces import EmbeddingsProvider, GenerationRequest, GenerationResponse, LLMProvider, Message
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
    cancellation_token: Any = None,
    request_id: str | None = None,
    call_id: str | None = None,
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
    req_id = request_id or str(uuid.uuid4())
    cid = call_id or "unknown"

    logger.info(
        "chat_request_received",
        session_id=request.session_id,
        request_id=req_id,
        call_id=cid,
        message_length=len(request.user_message),
    )

    # 1. Retrieve relevant knowledge
    t_retrieval_start = time.perf_counter()
    try:
        retrieval_results: list[RetrievalResult] = await retrieve_relevant_chunks(
            query=request.user_message,
            embeddings_provider=embeddings,
            settings=settings,
            request_id=req_id,
            call_id=cid,
        )
    except Exception as exc:
        logger.warning(
            "rag_retrieval_failed",
            session_id=request.session_id,
            request_id=req_id,
            call_id=cid,
            error=str(exc)[:150],
        )
        retrieval_results = []

    retrieval_duration_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

    # 2. Format passages for the system prompt
    t_proc_start = time.perf_counter()
    passages = _format_passages(retrieval_results)
    proc_duration_ms = (time.perf_counter() - t_proc_start) * 1000.0
    logger.info(
        "stage_timing",
        request_id=req_id,
        call_id=cid,
        stage="retrieved_chunks_processing",
        stage_name="Retrieved chunks processing",
        duration_ms=round(proc_duration_ms, 2),
    )

    # 3. Build messages
    t_prompt_start = time.perf_counter()
    system_prompt = build_system_prompt(passages)
    lang_directive = _build_turn_language_directive(request.user_message, request.conversation_history)
    history = _trim_history(request.conversation_history)

    if (
        history
        and history[-1].role == "user"
        and history[-1].content.strip() == request.user_message.strip()
    ):
        history = history[:-1]

    messages: list[Message] = [
        Message(role="system", content=system_prompt + lang_directive),
        *history,
        Message(role="user", content=request.user_message),
    ]

    gen_request = GenerationRequest(
        messages=messages,
        max_tokens=min(settings.llm_max_tokens, 256),
        temperature=settings.llm_temperature,
    )
    prompt_duration_ms = (time.perf_counter() - t_prompt_start) * 1000.0
    logger.info(
        "stage_timing",
        request_id=req_id,
        call_id=cid,
        stage="system_prompt_construction",
        stage_name="System prompt construction",
        duration_ms=round(prompt_duration_ms, 2),
    )

    # 4. Generate with streaming TTFT detection
    t_gen_start = time.perf_counter()
    logger.info(
        "stage_timing",
        request_id=req_id,
        call_id=cid,
        stage="gemini_api_request_started",
        stage_name="Gemini API request started",
        duration_ms=0.0,
    )

    ttft_ms: float | None = None
    gen_response: GenerationResponse | None = None
    stream_fn = getattr(llm, "generate_stream", None)

    if stream_fn is not None:
        try:
            tokens: list[str] = []
            async for chunk in stream_fn(gen_request):
                if cancellation_token and getattr(cancellation_token, "is_cancelled", False):
                    raise asyncio.CancelledError()
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_gen_start) * 1000.0
                    logger.info(
                        "stage_timing",
                        request_id=req_id,
                        call_id=cid,
                        stage="gemini_first_token_received",
                        stage_name="Gemini first token received",
                        duration_ms=round(ttft_ms, 2),
                    )
                tokens.append(chunk)

            full_text = "".join(tokens)
            llm_duration_ms = (time.perf_counter() - t_gen_start) * 1000.0
            logger.info(
                "stage_timing",
                request_id=req_id,
                call_id=cid,
                stage="gemini_response_completed",
                stage_name="Gemini response completed",
                duration_ms=round(llm_duration_ms, 2),
            )
            gen_response = GenerationResponse(
                content=full_text,
                model=getattr(llm, "_model", "gemini"),
                prompt_tokens=len(" ".join(m.content for m in messages).split()),
                completion_tokens=len(full_text.split()),
                finish_reason="stop",
                provider=getattr(llm, "provider_name", "gemini"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("streaming_failed_fallback_to_generate", error=str(exc))
            gen_response = None

    if gen_response is None:
        gen_response = await llm.generate(gen_request)
        llm_duration_ms = (time.perf_counter() - t_gen_start) * 1000.0
        logger.info(
            "stage_timing",
            request_id=req_id,
            call_id=cid,
            stage="gemini_response_completed",
            stage_name="Gemini response completed",
            duration_ms=round(llm_duration_ms, 2),
        )

    total_duration_ms = (time.perf_counter() - t_chat_start) * 1000.0

    logger.info(
        "timing_rag_pipeline",
        session_id=request.session_id,
        request_id=req_id,
        call_id=cid,
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
    cancellation_token: Any = None,
) -> AsyncGenerator[Union[str, ChatResponse], None]:
    """

    Sentence-streaming RAG chat pipeline for low-latency LLM→TTS pipelining.



    Yields:

         - str: Each complete sentence/phrase as soon as it is produced by the LLM.

                            The caller should immediately start TTS synthesis on each sentence.

         - ChatResponse: Exactly one final ChatResponse object (always the last item).

                            Contains the full response text, sources, and timing metadata.



    This cuts time-to-first-audio from \~2,800 ms (full LLM wait) to \~600–900 ms

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

    try:
        retrieval_results: list[RetrievalResult] = await retrieve_relevant_chunks(
            query=request.user_message,
            embeddings_provider=embeddings,
            settings=settings,
        )

    except Exception as exc:
        logger.warning(
            "rag_retrieval_failed",
            session_id=request.session_id,
            error=str(exc)[:150],
        )

        retrieval_results = []

    retrieval_duration_ms = (time.perf_counter() - t_retrieval_start) * 1000.0

    if cancellation_token is not None and cancellation_token.is_cancelled:
        raise asyncio.CancelledError()

    # 2. Build messages
    passages = _format_passages(retrieval_results)
    system_prompt = build_system_prompt(passages)
    lang_directive = _build_turn_language_directive(request.user_message, request.conversation_history)
    history = _trim_history(request.conversation_history)

    # The caller may already include the current utterance in history.
    # Keep it only once in the provider request.
    if (
        history
        and history[-1].role == "user"
        and history[-1].content.strip() == request.user_message.strip()
    ):
        history = history[:-1]

    messages: list[Message] = [
        Message(role="system", content=system_prompt + lang_directive),
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
                if cancellation_token is not None and cancellation_token.is_cancelled:
                    raise asyncio.CancelledError()

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
            err_msg = str(exc).lower()

            is_gateway_timeout = "504" in err_msg or "gateway" in err_msg

            is_deadline_timeout = (
                "deadline" in err_msg
                or "timed out" in err_msg
                or isinstance(exc, (TimeoutError, asyncio.TimeoutError))
            ) and not is_gateway_timeout

            category = "timeout" if is_deadline_timeout else "provider_error"

            logger.warning(
                "llm_stream_failed",
                session_id=request.session_id,
                provider=getattr(llm, "provider_name", "unknown"),
                model=getattr(llm, "_model", "unknown"),
                failure_category=category,
                tokens_yielded=tokens_yielded,
                error=str(exc)[:200],
                exc_type=type(exc).__name__,
            )

            if tokens_yielded == 0:
                if category == "timeout":
                    # Avoid spending a second full LLM deadline after a genuine timeout.
                    recovery_text = get_recovery_message(request.user_message)
                    full_text_parts = [recovery_text]
                    yield recovery_text
                else:
                    # Attempt non-streaming fallback for non-timeout provider errors.
                    try:
                        gen_response = await llm.generate(gen_request)
                        response_text = (gen_response.content or "").strip()
                        if response_text:
                            full_text_parts = [response_text]
                            yield response_text
                        else:
                            recovery_text = get_recovery_message(request.user_message)

                            full_text_parts = [recovery_text]

                            yield recovery_text

                    except Exception as fallback_exc:
                        logger.warning(
                            "llm_non_streaming_fallback_failed",
                            error=str(fallback_exc)[:150],
                            failure_category=category,
                        )

                        recovery_text = get_recovery_message(request.user_message)

                        full_text_parts = [recovery_text]

                        yield recovery_text

            else:
                logger.warning(
                    "llm_stream_partial_failure",
                    error=str(exc)[:150],
                    failure_category=category,
                    fallback_outcome="partial_response_preserved",
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

    # Never leave the user without a response if the stream is empty.

    if tokens_yielded == 0 and not "".join(full_text_parts).strip():
        recovery_text = get_recovery_message(request.user_message)

        full_text_parts = [recovery_text]

        yield recovery_text

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

    if cancellation_token is not None and cancellation_token.is_cancelled:
        raise asyncio.CancelledError()
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


def _build_turn_language_directive(
    user_message: str,
    conversation_history: list[Message] | None = None,
) -> str:
    """Explicit turn-level language enforcement so LLM formulates response in caller's language."""

    from app.voice.language import LanguageDetector, SupportedLanguage

    try:
        session_lang: SupportedLanguage | None = None
        if conversation_history:
            # Check recent history for non-English session context (e.g. Tamil, Hindi, Bengali)
            for msg in reversed(conversation_history[-4:]):
                txt = (msg.content or "").strip()
                if txt:
                    prev_det = LanguageDetector.detect_utterance(txt)
                    if prev_det.detected_language != SupportedLanguage.ENGLISH:
                        session_lang = prev_det.detected_language
                        break

        det = LanguageDetector.detect_utterance(user_message, session_language=session_lang)

        lang = det.detected_language

        directives = {
            SupportedLanguage.HINDI: "\n\n[Turn Language Directive: Caller spoke Hindi. You MUST reply in natural Hindi using Devanagari script. Explain retrieved facts in Hindi. Do NOT respond in English.]",
            SupportedLanguage.HINGLISH: "\n\n[Turn Language Directive: Caller spoke Hinglish. You MUST reply in natural conversational Hinglish using Roman script. Explain retrieved facts in Hinglish. Do NOT respond in pure English or pure Devanagari.]",
            SupportedLanguage.TAMIL: "\n\n[Turn Language Directive: Caller spoke Tamil. You MUST reply in natural Tamil using native Tamil script (தமிழ்). Explain retrieved facts in Tamil. Do NOT respond in English.]",
            SupportedLanguage.TELUGU: "\n\n[Turn Language Directive: Caller spoke Telugu. You MUST reply in natural Telugu using native Telugu script (తెలుగు). Explain retrieved facts in Telugu. Do NOT respond in English.]",
            SupportedLanguage.KANNADA: "\n\n[Turn Language Directive: Caller spoke Kannada. You MUST reply in natural Kannada using native Kannada script (ಕನ್ನಡ). Explain retrieved facts in Kannada. Do NOT respond in English.]",
            SupportedLanguage.MALAYALAM: "\n\n[Turn Language Directive: Caller spoke Malayalam. You MUST reply in natural Malayalam using native Malayalam script (മലയാളം). Explain retrieved facts in Malayalam. Do NOT respond in English.]",
            SupportedLanguage.BENGALI: "\n\n[Turn Language Directive: Caller spoke Bengali. You MUST reply in natural Bengali using native Bengali script (বাংলা). Explain retrieved facts in Bengali. Do NOT respond in English.]",
        }

        return directives.get(lang, "")

    except Exception:
        return ""


def get_recovery_message(user_message: str) -> str:
    """Localized recovery message when LLM provider times out or fails across all 7 languages."""

    try:
        from app.voice.language import LanguageDetector, SupportedLanguage

        det = LanguageDetector.detect_utterance(user_message)

        lang = det.detected_language

        if lang == SupportedLanguage.HINDI:
            return "माफ़ कीजिये, मुझे एक तकनीकी समस्या आ रही है। क्या आप कृपया अपना सवाल दोहरा सकते हैं?"

        elif lang == SupportedLanguage.HINGLISH:
            return "Sorry, mujhe ek brief technical issue aa raha hai. Kya aap please apna question repeat kar sakte hain?"

        elif lang == SupportedLanguage.TAMIL:
            return "மன்னிக்கவும், எனக்கு ஒரு சிறிய தொழில்நுட்ப சிக்கல் ஏற்பட்டுள்ளது. உங்கள் கேள்வியை மீண்டும் கேட்க முடியுமா?"

        elif lang == SupportedLanguage.TELUGU:
            return "క్షమించండి, నాకు ఒక చిన్న సాంకేతిక సమస్య ఎదురవుతోంది. దయచేసి మీ ప్రశ్నను మళ్లీ అడగగలరా?"

        elif lang == SupportedLanguage.KANNADA:
            return "ಕ್ಷಮಿಸಿ, ನನಗೆ ಒಂದು ಸಣ್ಣ ತಾಂತ್ರಿಕ ಸಮಸ್ಯೆ ಎದುರಾಗಿದೆ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಪುನರಾವರ್ತಿಸಬಹುದೇ?"

        elif lang == SupportedLanguage.MALAYALAM:
            return "ക്ഷമിക്കണം, എനിക്ക് ഒരു ചെറിയ സാങ്കേതിക പ്രശ്നം നേരിടുന്നു. ദയവായി നിങ്ങളുടെ ചോദ്യം ആവർത്തിക്കാമോ?"

        elif lang == SupportedLanguage.BENGALI:
            return "দুঃখিত, একটি প্রযুক্তিগত সমস্যা হয়েছে। অনুগ্রহ করে আপনার প্রশ্নটি আবার বলুন।"

    except Exception:
        pass

    return "I'm sorry, I'm having a brief technical issue. Could you please repeat your question?"
