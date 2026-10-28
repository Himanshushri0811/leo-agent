"""
Focused tests verifying demo calls API polling behavior and frontend client contract.
Validates:
1. GET /api/v1/demo/calls endpoint contract for session sync (?limit=25) and history (?limit=50).
2. Elimination of duplicate unmanaged polling timers in frontend client logic.
3. Overlap guards and lifecycle cleanup in static client script.
"""

from __future__ import annotations

from pathlib import Path
import re
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_demo_calls_endpoint_for_session_sync(async_client: AsyncClient) -> None:
    """Verify session sync caller receives valid call records with limit=25."""
    # Simulate a call to ensure data exists
    sim_resp = await async_client.post(
        "/api/v1/demo/calls/simulate",
        json={"caller_name": "Polling Verification Call", "auto_answer_delay_sec": 0},
    )
    assert sim_resp.status_code == 201
    call_id = sim_resp.json()["call_id"]

    # Test loadActiveSessions query (?limit=25)
    resp = await async_client.get("/api/v1/demo/calls?limit=25")
    assert resp.status_code == 200
    calls = resp.json()
    assert isinstance(calls, list)
    assert any(c["call_id"] == call_id for c in calls)


@pytest.mark.asyncio
async def test_demo_calls_endpoint_for_history_query(async_client: AsyncClient) -> None:
    """Verify loadHistory caller with filters returns up-to-date data."""
    resp = await async_client.get("/api/v1/demo/calls?limit=50&status=active")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    for call in data:
        assert call["status"] == "active"


def test_frontend_polling_contract_and_cleanup() -> None:
    """Verify app.js does not contain uncontrolled polling loops and implements proper guards."""
    static_file = Path("app/static/app.js")
    assert static_file.exists(), "app.js must exist in app/static"

    content = static_file.read_text(encoding="utf-8")

    # 1. Ensure blind 2.5s setInterval(loadActiveSessions, ...) is removed
    assert not re.search(r"setInterval\(\s*loadActiveSessions\s*,\s*2500\s*\)", content), (
        "Uncontrolled setInterval(loadActiveSessions, 2500) must be eliminated"
    )

    # 2. Ensure controlled scheduleNextSessionPoll exists
    assert "scheduleNextSessionPoll" in content, (
        "scheduleNextSessionPoll controlled polling function must be present"
    )

    # 3. Ensure overlap guard is present in loadActiveSessions
    assert "isFetchingSessions" in content, (
        "Overlap guard state.isFetchingSessions must be present"
    )

    # 4. Ensure sessions_snapshot is not nested inside ping listener
    ping_block = re.search(r"evtSource\.addEventListener\('ping',\s*\(\)\s*=>\s*\{([^}]*)\}\);", content)
    assert ping_block, "ping event listener must exist"
    assert "sessions_snapshot" not in ping_block.group(1), (
        "sessions_snapshot must not be nested inside ping listener"
    )

    # 5. Ensure cleanup handlers exist for timers
    assert "cleanupTimersAndConnections" in content, (
        "Timer and connection cleanup handler must exist"
    )
    assert "beforeunload" in content, "beforeunload listener must be registered"
    assert "visibilitychange" in content, "visibilitychange listener must be registered"
