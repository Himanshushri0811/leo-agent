"""Per-session turn ownership and cooperative barge-in cancellation."""

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
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_cancelled: bool = False
    cancelled_reason: str | None = None

    def cancel(self, reason: str = "user_barge_in") -> None:
        self.is_cancelled = True
        self.cancelled_reason = reason


class TurnManager:
    """Owns exactly one active generation token/task per voice session."""

    def __init__(self) -> None:
        self._tokens: dict[str, UtteranceCancellationToken] = {}
        self._states: dict[str, TurnState] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    async def get_state(self, session_id: str) -> TurnState:
        async with self._lock:
            return self._states.get(session_id, TurnState.IDLE)

    async def set_state(self, session_id: str, state: TurnState) -> None:
        async with self._lock:
            self._states[session_id] = state

    async def get_token(self, session_id: str) -> UtteranceCancellationToken | None:
        async with self._lock:
            return self._tokens.get(session_id)

    async def is_current(self, session_id: str, token: UtteranceCancellationToken) -> bool:
        async with self._lock:
            return self._tokens.get(session_id) is token and not token.is_cancelled

    async def register_turn_task(self, session_id: str, task: asyncio.Task[Any]) -> None:
        async with self._lock:
            if self._tokens.get(session_id) is not None:
                self._tasks[session_id] = task

    async def unregister_turn_task(
        self, session_id: str, task: asyncio.Task[Any] | None = None
    ) -> None:
        async with self._lock:
            current = self._tasks.get(session_id)
            if task is None or current is task:
                self._tasks.pop(session_id, None)

    async def cleanup_session(self, session_id: str) -> None:
        async with self._lock:
            task = self._tasks.pop(session_id, None)
            token = self._tokens.pop(session_id, None)
            if token:
                token.cancel("session_cleanup")
            self._states.pop(session_id, None)
            if task and task is not asyncio.current_task() and not task.done():
                task.cancel()

    async def on_user_speech_start(
        self, session_id: str, reason: str = "user_barge_in"
    ) -> UtteranceCancellationToken | None:
        async with self._lock:
            token = self._tokens.get(session_id)
            if token and not token.is_cancelled:
                token.cancel(reason)
                logger.info(
                    "barge_in_triggered_stale_speech_cancelled",
                    session_id=session_id,
                    token_id=token.token_id,
                    reason=reason,
                )
            task = self._tasks.pop(session_id, None)
            if task and task is not asyncio.current_task() and not task.done():
                task.cancel()
                logger.info(
                    "barge_in_cancelled_active_generation_task",
                    session_id=session_id,
                    reason=reason,
                )
            self._states[session_id] = TurnState.LISTENING
            return token

    async def create_agent_turn(
        self, session_id: str, task: asyncio.Task[Any] | None = None
    ) -> UtteranceCancellationToken:
        async with self._lock:
            previous = self._tokens.get(session_id)
            if previous:
                previous.cancel("superseded_by_new_turn")
            old_task = self._tasks.pop(session_id, None)
            if old_task and old_task is not asyncio.current_task() and not old_task.done():
                old_task.cancel()
            token = UtteranceCancellationToken()
            self._tokens[session_id] = token
            if task is not None:
                self._tasks[session_id] = task
            self._states[session_id] = TurnState.THINKING
            return token

    async def mark_speaking(self, session_id: str, token: UtteranceCancellationToken) -> bool:
        async with self._lock:
            if self._tokens.get(session_id) is not token or token.is_cancelled:
                return False
            self._states[session_id] = TurnState.SPEAKING
            return True

    async def complete_agent_turn(self, session_id: str, token: UtteranceCancellationToken) -> None:
        async with self._lock:
            if self._tokens.get(session_id) is not token:
                return
            task = self._tasks.get(session_id)
            if task is asyncio.current_task() or task is None or task.done():
                self._tasks.pop(session_id, None)
            if not token.is_cancelled:
                self._states[session_id] = TurnState.IDLE
                self._tokens.pop(session_id, None)

    async def filter_speech_stream(
        self, stream: AsyncIterator[bytes], token: UtteranceCancellationToken
    ) -> AsyncIterator[bytes]:
        try:
            async for chunk in stream:
                if token.is_cancelled:
                    logger.info(
                        "speech_stream_aborted_due_to_interruption",
                        token_id=token.token_id,
                        reason=token.cancelled_reason,
                    )
                    break
                yield chunk
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
