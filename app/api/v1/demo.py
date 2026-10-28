"""
FastAPI router for LEO Call Center Operations Dashboard Demo.

Endpoints:
- POST /api/v1/demo/calls/simulate: Initiate incoming call simulation
- GET  /api/v1/demo/calls: Query call history and active sessions
- GET  /api/v1/demo/calls/{call_id}: Retrieve single call detail & transcript
- POST /api/v1/demo/calls/{call_id}/action: Execute Answer/Reject/Mute/End actions
- POST /api/v1/demo/calls/{call_id}/message: Send customer question to real LEO brain
- POST /api/v1/demo/calls/{call_id}/escalate: Trigger human handoff
- GET  /api/v1/demo/agents: View agent availability queue
- POST /api/v1/demo/calls/{call_id}/assign: Assign human agent to escalated call
- GET  /api/v1/demo/metrics: Retrieve call center metrics
- GET  /api/v1/demo/events: Server-Sent Events (SSE) stream for live updates
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response, StreamingResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.demo import (
    AssignAgentRequest,
    CallActionRequest,
    CallMetrics,
    DemoCall,
    EscalateCallRequest,
    HumanAgent,
    SendCustomerMessageRequest,
    SimulateCallRequest,
    SimulateOutboundCallRequest,
    SpeakerRole,
    TranscriptTurn,
)
from app.services.demo_call_service import DemoCallService, get_demo_service

router = APIRouter(prefix="/api/v1/demo", tags=["demo"])
logger = get_logger(__name__)


@router.post(
    "/calls/simulate",
    response_model=DemoCall,
    status_code=status.HTTP_201_CREATED,
    summary="Simulate incoming call",
    description="Creates an incoming call event with simulated caller info and optional auto-answer delay.",
)
async def simulate_call(
    body: SimulateCallRequest,
    demo_service: DemoCallService = Depends(get_demo_service),
) -> DemoCall:
    try:
        call = await demo_service.simulate_incoming_call(
            caller_name=body.caller_name,
            caller_phone=body.caller_phone,
            auto_answer_delay_sec=body.auto_answer_delay_sec,
            initial_message=body.initial_message,
        )
        return call
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc))


@router.post(
    "/calls/simulate-outbound",
    response_model=DemoCall,
    status_code=status.HTTP_201_CREATED,
    summary="Simulate outbound call",
    description="Creates an outbound call event with customer details and stated call purpose.",
)
async def simulate_outbound_call(
    body: SimulateOutboundCallRequest,
    demo_service: DemoCallService = Depends(get_demo_service),
) -> DemoCall:
    try:
        call = await demo_service.simulate_outbound_call(
            customer_name=body.customer_name,
            customer_phone=body.customer_phone,
            call_purpose=body.call_purpose,
            auto_answer_delay_sec=body.auto_answer_delay_sec,
        )
        return call
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc))


@router.get(
    "/calls",
    response_model=list[DemoCall],
    summary="List calls",
    description="Retrieve list of calls with status filter, search query, and pagination.",
)
async def list_calls(
    status: str | None = Query(None, description="Filter by status (ringing, active, completed, etc.)"),
    search: str | None = Query(None, description="Search query across caller, ID, phone, or transcript"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    demo_service: DemoCallService = Depends(get_demo_service),
) -> list[DemoCall]:
    return demo_service.list_calls(
        status_filter=status,
        search_query=search,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/calls/{call_id}",
    response_model=DemoCall,
    summary="Get call detail",
)
async def get_call(
    call_id: str,
    demo_service: DemoCallService = Depends(get_demo_service),
) -> DemoCall:
    call = await demo_service.get_or_load_call(call_id)
    if not call:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Call '{call_id}' not found.")
    return call


@router.post(
    "/calls/{call_id}/action",
    response_model=DemoCall,
    summary="Execute call action",
    description="Perform answer, reject, mute, unmute, or end on a call.",
)
async def execute_action(
    call_id: str,
    body: CallActionRequest,
    demo_service: DemoCallService = Depends(get_demo_service),
) -> DemoCall:
    try:
        return await demo_service.execute_call_action(
            call_id=call_id,
            action=body.action,
            reason=body.reason,
        )
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Call '{call_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post(
    "/calls/{call_id}/message",
    response_model=TranscriptTurn,
    summary="Send customer turn",
    description="Sends customer message turn into live call, invoking LEO AI brain for response.",
)
async def send_message(
    call_id: str,
    body: SendCustomerMessageRequest,
    demo_service: DemoCallService = Depends(get_demo_service),
) -> TranscriptTurn:
    try:
        return await demo_service.send_customer_message(
            call_id=call_id,
            message_text=body.message,
        )
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Call '{call_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post(
    "/calls/{call_id}/escalate",
    response_model=DemoCall,
    summary="Escalate call",
    description="Route call to human specialist queue.",
)
async def escalate_call(
    call_id: str,
    body: EscalateCallRequest,
    demo_service: DemoCallService = Depends(get_demo_service),
) -> DemoCall:
    try:
        return await demo_service.trigger_escalation(
            call_id=call_id,
            reason=body.reason,
            notes=body.notes,
        )
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Call '{call_id}' not found.")


@router.get(
    "/agents",
    response_model=list[HumanAgent],
    summary="List human agents",
    description="Retrieve agent queue status.",
)
async def list_agents(
    demo_service: DemoCallService = Depends(get_demo_service),
) -> list[HumanAgent]:
    return demo_service.list_agents()


@router.post(
    "/calls/{call_id}/assign",
    response_model=DemoCall,
    summary="Assign human agent",
    description="Assign a human agent to an escalated call.",
)
async def assign_agent(
    call_id: str,
    body: AssignAgentRequest,
    demo_service: DemoCallService = Depends(get_demo_service),
) -> DemoCall:
    try:
        return await demo_service.assign_agent(
            call_id=call_id,
            agent_id=body.agent_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get(
    "/metrics",
    response_model=CallMetrics,
    summary="Call center metrics",
)
async def get_metrics(
    demo_service: DemoCallService = Depends(get_demo_service),
) -> CallMetrics:
    return demo_service.get_metrics()


@router.api_route(
    "/calls/{call_id}/turns/{turn_id}/audio",
    methods=["GET", "POST"],
    summary="Synthesize LEO audio for a transcript turn",
    description=(
        "Synthesizes TTS audio for a specific LEO transcript turn and returns "
        "a playable WAV audio response. The audio is generated on demand using "
        "the configured TTS provider (default: Gemini with Aoede voice)."
    ),
    response_class=Response,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "Playable WAV audio"},
        404: {"description": "Call or turn not found"},
        400: {"description": "Turn is not a LEO turn or call not active"},
        503: {"description": "TTS provider unavailable or synthesis failed"},
    },
)
async def synthesize_turn_audio(
    call_id: str,
    turn_id: str,
    demo_service: DemoCallService = Depends(get_demo_service),
) -> Response:
    """
    Synthesize TTS audio for a LEO transcript turn and return WAV bytes.

    The frontend calls this after receiving a `transcript_turn` SSE event
    with role='leo'. Audio is returned as audio/wav so the browser can play
    it directly via Web Audio API or an <audio> element.
    """
    call = await demo_service.get_or_load_call(call_id)
    if not call:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Call '{call_id}' not found.")

    # Find the requested turn in the transcript
    turn = next((t for t in call.transcript if t.id == turn_id), None)
    if not turn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Turn '{turn_id}' not found in call '{call_id}'.")

    if turn.role != SpeakerRole.LEO:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Turn '{turn_id}' is role '{turn.role}', not 'leo'. Only LEO turns can be synthesized.",
        )

    if not turn.text or not turn.text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Turn text is empty.")

    try:
        lang_code = turn.metadata.get("language_locale") or call.metadata.get("language_locale")
        wav_bytes = await demo_service.synthesize_turn_audio(turn.text, language_code=lang_code)
    except Exception as exc:
        logger.error("tts_synthesis_endpoint_failed", call_id=call_id, turn_id=turn_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"TTS synthesis failed: {str(exc)}",
        )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="leo_turn_{turn_id}.wav"',
            "Cache-Control": "no-cache",
            "X-LEO-Turn-Id": turn_id,
            "X-LEO-TTS-Provider": "gemini",
        },
    )


@router.api_route(
    "/calls/{call_id}/turns/{turn_id}/audio-stream",
    methods=["GET", "POST"],
    summary="Stream LEO audio progressively",
    description=(
        "Streams TTS audio as framed raw PCM chunks so the browser can begin "
        "playback before the entire answer is generated. Falls back to the "
        "non-streaming endpoint if streaming is unavailable."
    ),
    response_class=StreamingResponse,
    responses={
        200: {"content": {"application/octet-stream": {}}, "description": "Framed PCM audio stream"},
        404: {"description": "Call or turn not found"},
        400: {"description": "Turn is not a LEO turn or call not active"},
        503: {"description": "TTS provider unavailable or synthesis failed"},
    },
)
async def stream_turn_audio(
    call_id: str,
    turn_id: str,
    demo_service: DemoCallService = Depends(get_demo_service),
) -> StreamingResponse:
    """Stream TTS audio as framed PCM chunks for progressive playback."""
    call = await demo_service.get_or_load_call(call_id)
    if not call:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Call '{call_id}' not found.")

    turn = next((t for t in call.transcript if t.id == turn_id), None)
    if not turn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Turn '{turn_id}' not found in call '{call_id}'.")

    if turn.role != SpeakerRole.LEO:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Turn '{turn_id}' is role '{turn.role}', not 'leo'. Only LEO turns can be synthesized.",
        )

    if not turn.text or not turn.text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Turn text is empty.")

    cancellation_token = await demo_service.turn_manager.get_token(call.session_id)
    lang_code = turn.metadata.get("language_locale") or call.metadata.get("language_locale")

    async def audio_stream():
        try:
            async for chunk in demo_service.stream_turn_audio_framed(
                turn.text,
                turn_id=turn_id,
                cancellation_token=cancellation_token,
                language_code=lang_code,
            ):
                yield chunk
        except Exception as exc:
            logger.error("tts_stream_endpoint_failed", call_id=call_id, turn_id=turn_id, error=str(exc))

    return StreamingResponse(
        audio_stream(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'inline; filename="leo_turn_{turn_id}.pcm"',
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-LEO-Turn-Id": turn_id,
            "X-LEO-TTS-Provider": "gemini",
            "X-LEO-Streaming": "true",
        },
    )


@router.post(
    "/calls/{call_id}/barge-in",
    response_model=DemoCall,
    summary="Signal user speech onset / barge-in",
    description="Immediately aborts active audio synthesis and server-side generation when customer speaks.",
)
async def barge_in(
    call_id: str,
    demo_service: DemoCallService = Depends(get_demo_service),
) -> DemoCall:
    try:
        return await demo_service.handle_barge_in(call_id, reason="user_barge_in")
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Call '{call_id}' not found.")


@router.get(
    "/tts/config",
    summary="TTS provider configuration",
    description="Returns the active TTS provider name, model, and voice for the frontend.",
)
async def get_tts_config() -> dict:
    settings = get_settings()
    return {
        "provider": settings.tts_provider,
        "model": settings.tts_model,
        "voice": settings.tts_voice,
        "language": settings.tts_language_code,
        "enabled": settings.tts_provider != "mock",
    }


@router.post(
    "/calls/{call_id}/live-stream",
    summary="Ultra-low-latency LLM→TTS pipeline stream",
    description=(
        "Accepts the customer message and immediately begins LLM inference with TTS synthesis "
        "sentence-by-sentence. Returns a framed binary stream with TEXT (0x03), HEADER (0x01), "
        "PCM (0x02), and DONE (0x04) frames interleaved. First audio arrives in ~700–1200ms. "
        "Also broadcasts transcript_turn SSE event with the full response once LLM completes."
    ),
    response_class=StreamingResponse,
    responses={
        200: {"content": {"application/octet-stream": {}}, "description": "Framed LLM+PCM stream"},
        404: {"description": "Call not found"},
        400: {"description": "Call not active or message empty"},
        503: {"description": "TTS/LLM provider unavailable"},
    },
)
async def live_stream_response(
    call_id: str,
    body: SendCustomerMessageRequest,
    demo_service: DemoCallService = Depends(get_demo_service),
) -> StreamingResponse:
    """
    Pipeline: customer message → LLM token stream → SentenceAccumulator → TTS framed PCM.

    The client connects immediately after sending the message; there is no need to wait
    for the SSE transcript_turn event before requesting audio.
    """
    call = await demo_service.get_or_load_call(call_id)
    if not call:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Call '{call_id}' not found.")

    message_text = (body.message or "").strip()
    if not message_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message text is empty.")

    from app.schemas.demo import CallStatus, SpeakerRole
    if call.status in (CallStatus.COMPLETED, CallStatus.FAILED, CallStatus.CALL_ENDING):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot stream to finished call ({call.status}).",
        )

    # Resolve language for TTS
    from app.voice.language import resolve_response_language
    _det_lang, lang_locale = resolve_response_language(message_text, call.metadata)

    # Get the current cancellation token for barge-in support
    cancellation_token = await demo_service.turn_manager.get_token(call.session_id)

    async def pipeline_stream():
        try:
            async for frame in demo_service.stream_llm_then_tts_framed(
                call_id=call_id,
                message_text=message_text,
                cancellation_token=cancellation_token,
                language_code=lang_locale,
            ):
                yield frame
        except KeyError:
            pass
        except Exception as exc:
            logger.error("live_stream_pipeline_failed", call_id=call_id, error=str(exc))

    return StreamingResponse(
        pipeline_stream(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'inline; filename="leo_live_{call_id}.pcm"',
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-LEO-Call-Id": call_id,
            "X-LEO-Streaming": "live",
            "X-LEO-TTS-Provider": "gemini",
        },
    )


@router.get(
    "/events",
    summary="SSE Real-time Events Stream",
    description="Stream live call events, state transitions, and transcript turns to dashboard clients.",
)
async def event_stream(
    demo_service: DemoCallService = Depends(get_demo_service),
) -> StreamingResponse:
    async def sse_generator():
        queue = await demo_service.subscribe()
        try:
            # Send initial keepalive
            yield f"event: ping\ndata: {json.dumps({'status': 'connected'})}\n\n"

            # Send current calls snapshot on connect/reconnect
            calls = await asyncio.to_thread(
                demo_service.list_calls,
                limit=25,
                offset=0,
            )

            snapshot = [
                call.model_dump(mode="json") if hasattr(call, "model_dump") else call
                for call in calls
            ]

            yield f"event: sessions_snapshot\ndata: {json.dumps(snapshot, default=str)}\n\n"

            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=25.0)
                    yield f"event: {payload['event']}\ndata: {json.dumps(payload['data'])}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat to keep connection alive
                    yield "event: ping\ndata: {}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await demo_service.unsubscribe(queue)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
