"""
API integration tests for LEO Call Center Operations Demo Endpoints.
Verifies REST contract for call simulation, actions, messaging, escalation, agent queue, and metrics.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_simulate_call_endpoint(async_client: AsyncClient) -> None:
    resp = await async_client.post(
        "/api/v1/demo/calls/simulate",
        json={
            "caller_name": "Horizon Health",
            "caller_phone": "+1 (555) 341-9876",
            "auto_answer_delay_sec": 0,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["caller_name"] == "Horizon Health"
    assert data["status"] == "ringing"
    assert data["is_simulated"] is True
    assert "call_id" in data


@pytest.mark.asyncio
async def test_list_and_get_calls_endpoints(async_client: AsyncClient) -> None:
    # 1. Create a call
    sim_resp = await async_client.post(
        "/api/v1/demo/calls/simulate",
        json={"auto_answer_delay_sec": 0},
    )
    call_id = sim_resp.json()["call_id"]

    # 2. List calls
    list_resp = await async_client.get("/api/v1/demo/calls")
    assert list_resp.status_code == 200
    calls = list_resp.json()
    assert any(c["call_id"] == call_id for c in calls)

    # 3. Get specific call
    get_resp = await async_client.get(f"/api/v1/demo/calls/{call_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["call_id"] == call_id


@pytest.mark.asyncio
async def test_call_action_endpoint(async_client: AsyncClient) -> None:
    sim_resp = await async_client.post(
        "/api/v1/demo/calls/simulate",
        json={"auto_answer_delay_sec": 0},
    )
    call_id = sim_resp.json()["call_id"]

    # Answer call
    ans_resp = await async_client.post(
        f"/api/v1/demo/calls/{call_id}/action",
        json={"action": "answer"},
    )
    assert ans_resp.status_code == 200
    assert ans_resp.json()["status"] == "active"

    # Mute call
    mute_resp = await async_client.post(
        f"/api/v1/demo/calls/{call_id}/action",
        json={"action": "mute"},
    )
    assert mute_resp.status_code == 200
    assert mute_resp.json()["is_muted"] is True

    # End call
    end_resp = await async_client.post(
        f"/api/v1/demo/calls/{call_id}/action",
        json={"action": "end"},
    )
    assert end_resp.status_code == 200
    assert end_resp.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_customer_message_endpoint(async_client: AsyncClient) -> None:
    sim_resp = await async_client.post(
        "/api/v1/demo/calls/simulate",
        json={"auto_answer_delay_sec": 0},
    )
    call_id = sim_resp.json()["call_id"]
    await async_client.post(
        f"/api/v1/demo/calls/{call_id}/action",
        json={"action": "answer"},
    )

    msg_resp = await async_client.post(
        f"/api/v1/demo/calls/{call_id}/message",
        json={"message": "What cloud architecture consulting do you offer?"},
    )
    assert msg_resp.status_code == 200
    turn = msg_resp.json()
    assert turn["role"] == "customer"
    assert "What cloud architecture" in turn["text"]


@pytest.mark.asyncio
async def test_escalation_and_agent_assignment_endpoints(async_client: AsyncClient) -> None:
    sim_resp = await async_client.post(
        "/api/v1/demo/calls/simulate",
        json={"auto_answer_delay_sec": 0},
    )
    call_id = sim_resp.json()["call_id"]
    await async_client.post(
        f"/api/v1/demo/calls/{call_id}/action",
        json={"action": "answer"},
    )

    # Escalate
    esc_resp = await async_client.post(
        f"/api/v1/demo/calls/{call_id}/escalate",
        json={"reason": "customer_request", "notes": "Customer requested specialist."},
    )
    assert esc_resp.status_code == 200
    assert esc_resp.json()["escalated"] is True

    # List agents
    agent_resp = await async_client.get("/api/v1/demo/agents")
    assert agent_resp.status_code == 200
    agents = agent_resp.json()
    assert len(agents) > 0
    available_agent = next(a for a in agents if a["status"] == "available")

    # Assign agent
    assign_resp = await async_client.post(
        f"/api/v1/demo/calls/{call_id}/assign",
        json={"agent_id": available_agent["agent_id"]},
    )
    assert assign_resp.status_code == 200
    assert assign_resp.json()["assigned_agent_id"] == available_agent["agent_id"]


@pytest.mark.asyncio
async def test_metrics_endpoint(async_client: AsyncClient) -> None:
    resp = await async_client.get("/api/v1/demo/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_calls" in data
    assert "active_calls" in data
    assert data["concurrency_limit"] == 25
    assert "Unavailable" in data["telephony_bridge_status"]


@pytest.mark.asyncio
async def test_simulate_outbound_call_endpoint(async_client: AsyncClient) -> None:
    resp = await async_client.post(
        "/api/v1/demo/calls/simulate-outbound",
        json={
            "customer_name": "Acme Corp CTO",
            "customer_phone": "+1 (555) 777-1234",
            "call_purpose": "Security compliance audit",
            "auto_answer_delay_sec": 0,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["caller_name"] == "Acme Corp CTO"
    assert data["direction"] == "outbound"
    assert data["status"] == "ringing"
    assert "call_id" in data
