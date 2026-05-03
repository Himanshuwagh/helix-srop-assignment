"""
Integration tests — exercise the full SROP pipeline.
LLM mocked at the ADK boundary (not at the HTTP layer).
"""
import pytest
from sqlalchemy import select

from app.db.models import Session


@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_001"})
    assert resp.status_code == 200
    assert "session_id" in resp.json()


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_adk):
    """
    Core integration test.

    Sends a knowledge question, asserts:
    1. Response contains a reply
    2. routed_to == "knowledge"
    3. trace exists with retrieved chunk IDs
    4. Turn 2 in the same session has access to context from turn 1
       (state persistence — at minimum, plan_tier available without re-asking)

    Implement after pipeline.run() and state persistence are working.
    The mock_adk fixture must patch at the ADK boundary, not at the HTTP layer.
    """
    # Create session
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_002", "plan_tier": "pro"})
    session_id = sess.json()["session_id"]

    # Turn 1 — knowledge query
    r1 = await client.post(f"/v1/chat/{session_id}", json={"content": "How do I rotate a deploy key?"})
    assert r1.status_code == 200
    assert r1.json()["routed_to"] == "knowledge"
    trace_id = r1.json()["trace_id"]

    # Trace must have chunk IDs
    trace = await client.get(f"/v1/traces/{trace_id}")
    assert trace.status_code == 200
    assert len(trace.json()["retrieved_chunk_ids"]) > 0

    # Turn 2 — follow-up in same session
    r2 = await client.post(f"/v1/chat/{session_id}", json={"content": "What is my plan tier?"})
    assert r2.status_code == 200
    # Agent should know plan_tier from state — not re-ask
    assert "pro" in r2.json()["reply"].lower()


@pytest.mark.asyncio
async def test_ambiguous_followup_uses_persisted_recent_turns(client, db, mock_adk):
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_002", "plan_tier": "pro"})
    session_id = sess.json()["session_id"]

    r1 = await client.post(f"/v1/chat/{session_id}", json={"content": "Show me my last 3 builds"})
    assert r1.status_code == 200
    assert r1.json()["routed_to"] == "account"

    r2 = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "What was the most recent one's status?"},
    )
    assert r2.status_code == 200
    assert r2.json()["routed_to"] == "account"
    assert "failed" in r2.json()["reply"].lower()

    session = await db.scalar(select(Session).where(Session.session_id == session_id))
    assert session is not None
    recent_turns = session.state["recent_turns"]
    assert len(recent_turns) == 2
    assert recent_turns[0]["user_message"] == "Show me my last 3 builds"
    assert "build_u_test_002_1: failed" in recent_turns[0]["assistant_response"]
    assert recent_turns[1]["user_message"] == "What was the most recent one's status?"


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client):
    resp = await client.post("/v1/chat/nonexistent-id", json={"content": "hello"})
    assert resp.status_code == 404
