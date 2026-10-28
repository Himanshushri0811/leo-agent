"""
Conversational Turn Manager & Barge-in Handling.

Provides:
- Turn-taking lifecycle tracking (LISTENING -> THINKING -> SPEAKING -> INTERRUPTED).
- Instant cancellation of stale speech chunks upon user interruption / barge-in.
- Per-session cancellation tokens.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
import uuid

from app.core.logging import get_logger

logger = get_logger(__name__)


class TurnState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class UtteranceCancellationToken:
    """Cooperative cancellation token passed into speech synthesis and streaming."""

    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_cancelled: bool = False
    cancelled_reason: str | None = None

    def cancel(self, reason: str = "user_barge_in") -> None:
        self.is_cancelled = True
        self.cancelled_reason = reason


class TurnManager:
    """Coordinates voice turn-taking and handles interruption / barge-in per session."""

    def __init__(self) -> None:
        self._tokens: dict[str, UtteranceCancellationToken] = {}
        self._states: dict[str, TurnState] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    async def get_state(self, session_id: str) -> TurnState:
        async with self._lock:
            return self._states.get(session_id, TurnState.IDLE)

    async def set_state(self, session_id: str, state: TurnState) -> None:
        """Update conversation turn state for a session."""
        async with self._lock:
            self._states[session_id] = state

    async def get_token(self, session_id: str) -> UtteranceCancellationToken | None:
        """Retrieve current active utterance cancellation token for a session."""
        async with self._lock:
            return self._tokens.get(session_id)

    async def register_turn_task(self, session_id: str, task: asyncio.Task[Any]) -> None:
        """Associate an in-flight background task (e.g. LLM generation) with the session."""
        async with self._lock:
            self._tasks[session_id] = task

    async def unregister_turn_task(self, session_id: str, task: asyncio.Task[Any] | None = None) -> None:
        """Disassociate a completed or cancelled task from the session."""
        async with self._lock:
            current = self._tasks.get(session_id)
            if task is None or current is task:
                self._tasks.pop(session_id, None)

    async def cleanup_session(self, session_id: str) -> None:
        """Release session tokens, tasks, and states upon call completion."""
        async with self._lock:
            task = self._tasks.pop(session_id, None)
            if task and not task.done():
                task.cancel()
            self._tokens.pop(session_id, None)
            self._states.pop(session_id, None)

    async def on_user_speech_start(self, session_id: str, reason: str = "user_barge_in") -> UtteranceCancellationToken | None:
        """
        Invoked when Voice Activity Detection (VAD) or new customer speech detects barge-in.
        Instantly cancels any in-flight agent speech token and background task for this session.
        """
        async with self._lock:
            active_token = self._tokens.get(session_id)
            if active_token and not active_token.is_cancelled:
                active_token.cancel(reason=reason)
                logger.info(
                    "barge_in_triggered_stale_speech_cancelled",
                    session_id=session_id,
                    token_id=active_token.token_id,
                    reason=reason,
                )

            # Cancel active generation task if currently awaiting in background
            active_task = self._tasks.pop(session_id, None)
            if active_task and not active_task.done():
                active_task.cancel()
                logger.info(
                    "barge_in_cancelled_active_generation_task",
                    session_id=session_id,
                    reason=reason,
                )

            self._states[session_id] = TurnState.LISTENING
            return active_token

    async def create_agent_turn(
        self, session_id: str, task: asyncio.Task[Any] | None = None
    ) -> UtteranceCancellationToken:
        """Create a fresh cancellation token when the agent begins generating an utterance."""
        async with self._lock:
            # Cancel any previous stale turn task before creating a new one
            prev_token = self._tokens.get(session_id)
            if prev_token and not prev_token.is_cancelled:
                prev_token.cancel(reason="superseded_by_new_turn")
            prev_task = self._tasks.pop(session_id, None)
            if prev_task and not prev_task.done():
                prev_task.cancel()

            token = UtteranceCancellationToken()
            self._tokens[session_id] = token
            if task:
                self._tasks[session_id] = task
            self._states[session_id] = TurnState.SPEAKING
            return token

    async def complete_agent_turn(self, session_id: str, token: UtteranceCancellationToken) -> None:
        """Mark agent turn completed normally unless it was interrupted."""
        async with self._lock:
            if self._tokens.get(session_id) is not token:
                return  # a newer turn owns this session; do not touch its task or state
            self._tasks.pop(session_id, None)
            if not token.is_cancelled:
                self._states[session_id] = TurnState.IDLE

    async def filter_speech_stream(
        self,
        stream: AsyncIterator[bytes],
        token: UtteranceCancellationToken,
    ) -> AsyncIterator[bytes]:
        """
        Yields audio chunks from the provider only as long as the token is not cancelled.
        Aborts immediately when interrupted, protecting against stale speech playback.
        """
        async for chunk in stream:
            if token.is_cancelled:
                logger.info(
                    "speech_stream_aborted_due_to_interruption",
                    token_id=token.token_id,
                    reason=token.cancelled_reason,
                )
                break
            yield chunk
