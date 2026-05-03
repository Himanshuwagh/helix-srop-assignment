"""
SROP entrypoint — called by the message route.

This is the core of the assignment. It ties together:
  - Loading session state from DB
  - Running the ADK orchestrator with that state as context
  - Extracting routing decision and tool calls from ADK events
  - Recording the trace
  - Persisting updated session state to DB

The route calls: result = await pipeline.run(session_id, user_message, db)
It receives: PipelineResult(content, routed_to, trace_id)

Design questions you need to answer:
  1. How do you inject SessionState into the ADK agent so it knows the user's context?
     (system prompt injection vs ADK session state vs re-hydrating from message history)
  2. How do you determine WHICH sub-agent handled the turn from ADK's event stream?
  3. How do you capture tool calls (name, args, result) for the trace?
  4. What is your timeout strategy? (see settings.llm_timeout_seconds)
  5. If the DB write for state fails after the LLM responds, what do you do?

See docs/google-adk-guide.md for ADK event stream patterns.
"""
import asyncio
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import OrchestratorResult, run_orchestrator
from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, Message, Session as SessionModel
from app.settings import settings
from app.srop.state import SessionState


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


def _try_answer_from_context(state: SessionState, message: str) -> str | None:
    """Deterministic guard: answer directly from session state when possible.

    Only triggers for short, single-intent messages to avoid cutting off
    compound questions that need LLM reasoning.
    """
    msg = message.lower().strip(" ?!.")

    # If the message is long or contains "and", it's likely a compound question.
    # Let the LLM handle it so we don't miss context.
    if len(msg.split()) > 7 or " and " in msg or " also " in msg:
        return None

    if any(p in msg for p in ("plan tier", "tier plan", "what tier", "my tier", "which plan")):
        return f"Your plan tier is {state.plan_tier}."
    if any(p in msg for p in ("user id", "my id", "who am i", "my user id")):
        return f"Your user ID is {state.user_id}."
    if any(p in msg for p in ("how many turns", "turn count", "how long", "conversation length")):
        return f"We have had {state.turn_count} turn(s) so far in this session."
    if any(p in msg for p in ("last agent", "who helped", "previous agent")):
        return f"The last agent you spoke with was {state.last_agent}." if state.last_agent else "This is the first turn of the session."
    return None


async def run(session_id: str, user_message: str, db: AsyncSession, idempotency_key: str | None = None) -> PipelineResult:
    # E1: Idempotency check — if we've seen this key for this session, return the cached reply.
    if idempotency_key:
        stmt = select(Message).where(
            Message.session_id == session_id,
            Message.idempotency_key == idempotency_key,
            Message.role == "assistant"
        )
        existing_msg = await db.scalar(stmt)
        if existing_msg:
            # We found a previous response for this key.
            # Map it back to the trace so the caller gets the full PipelineResult.
            trace_stmt = select(AgentTrace).where(AgentTrace.trace_id == existing_msg.trace_id)
            trace = await db.scalar(trace_stmt)
            return PipelineResult(
                content=existing_msg.content,
                routed_to=trace.routed_to if trace else "unknown",
                trace_id=existing_msg.trace_id or "unknown"
            )

    trace_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    session = await db.scalar(select(SessionModel).where(SessionModel.session_id == session_id))
    if session is None:
        raise SessionNotFoundError(f"Session {session_id} does not exist")

    state = SessionState.from_db_dict(session.state)
    session_context = (
        f"- user_id: {state.user_id}\n"
        f"- plan_tier: {state.plan_tier}\n"
        f"- last_agent: {state.last_agent}\n"
        f"- turn_count: {state.turn_count}\n"
        f"- ticket_ids: {', '.join(state.ticket_ids) if state.ticket_ids else 'None'}\n"
        f"- recent_conversation:\n{_format_recent_turns(state)}"
    )

    db.add(
        Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=user_message,
            trace_id=trace_id,
            idempotency_key=idempotency_key,
        )
    )

    # Deterministic guard: if the answer is in session state, don't waste an LLM call.
    direct_reply = _try_answer_from_context(state, user_message)
    if direct_reply:
        orchestrator_result = OrchestratorResult(
            reply=direct_reply,
            routed_to="account" if any(p in user_message.lower() for p in ("plan", "tier")) else "smalltalk",
            tool_calls=[],
            retrieved_chunk_ids=[],
        )
    else:
        try:
            orchestrator_result = await asyncio.wait_for(
                run_orchestrator(
                    session_id=session_id,
                    user_message=user_message,
                    session_context=session_context,
                ),
                timeout=settings.llm_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise UpstreamTimeoutError(
                f"LLM did not respond within {settings.llm_timeout_seconds}s"
            ) from exc

    state.turn_count += 1
    if orchestrator_result.routed_to in {"knowledge", "account", "smalltalk", "escalation"}:
        state.last_agent = orchestrator_result.routed_to

    # E2: Capture any newly created ticket IDs into session state
    for call in orchestrator_result.tool_calls:
        if call["tool_name"] == "create_ticket" and call["result"]:
            state.ticket_ids.append(call["result"])

    state.append_turn(user_message=user_message, assistant_response=orchestrator_result.reply)
    session.state = state.to_db_dict()

    db.add(
        Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=orchestrator_result.reply,
            trace_id=trace_id,
            idempotency_key=idempotency_key,
        )
    )
    db.add(
        AgentTrace(
            trace_id=trace_id,
            session_id=session_id,
            routed_to=orchestrator_result.routed_to,
            tool_calls=orchestrator_result.tool_calls,
            retrieved_chunk_ids=orchestrator_result.retrieved_chunk_ids,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
    )
    await db.commit()
    return PipelineResult(
        content=orchestrator_result.reply,
        routed_to=orchestrator_result.routed_to,
        trace_id=trace_id,
    )


def _format_recent_turns(state: SessionState) -> str:
    if not state.recent_turns:
        return "None"

    lines: list[str] = []
    for index, turn in enumerate(state.recent_turns, start=1):
        lines.append(f"  Turn {index} user: {turn.user_message}")
        lines.append(f"  Turn {index} assistant: {turn.assistant_response}")
    return "\n".join(lines)

