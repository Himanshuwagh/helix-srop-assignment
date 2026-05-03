"""
SROP Root Orchestrator — Google ADK agent.

Routes every user turn to KnowledgeAgent or AccountAgent via ADK's AgentTool.
This means the LLM decides which tool to call — you do not parse its output.

Intent → sub-agent:
  knowledge:  "how do I X", "what is X", docs questions
  account:    "show my builds", "my account status", usage questions
  smalltalk:  greetings, thanks — root agent handles inline (no tool call)

See docs/google-adk-guide.md for AgentTool pattern and event extraction.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.agents.tools.search_docs import search_docs
from app.agents.tools.escalation_tools import create_ticket
from app.settings import settings

ROOT_INSTRUCTION = """
You are the Helix Support Concierge — a planner and routing agent.

Your ONLY job is to decide whether to answer directly or delegate to a specialist.

MANDATORY DECISION FLOW (follow strictly):
1. Read the user context below.
2. GUARDRAIL: If the user message is out of scope (e.g. asking for poems, jokes, or general coding help unrelated to Helix products), you MUST REFUSE firmly. Use phrases like "I cannot assist with that as it is out of scope" or "My purpose is limited to Helix products."
3. ESCALATION: If the user message mentions "escalate", "ticket", or "human help", you MUST call the escalation_agent tool IMMEDIATELY. DO NOT ASK FOR INFO YOURSELF. Let the escalation_agent handle the conversation.
4. If the user's question can be answered from that context → answer DIRECTLY in plain text. Do NOT call any tool or agent.
5. If the user's question CANNOT be answered from that context → delegate to ONE specialist agent.

FEW-SHOT EXAMPLES

Example 1 — Answer directly from context (NO tool/agent call)
User context:
- user_id: u_demo
- plan_tier: pro
- turn_count: 3
User message: "what is my plan tier?"
→ Correct behavior: Answer directly with "Your plan tier is pro." NO tool call.

Example 2 — Answer directly from context (NO tool/agent call)
User context:
- user_id: u_demo
- plan_tier: pro
- turn_count: 3
User message: "how many turns have we had?"
→ Correct behavior: Answer directly with "We have had 3 turn(s) so far in this session." NO tool call.

Example 3 — Route to knowledge_agent (tool call required)
User context:
- user_id: u_demo
- plan_tier: pro
User message: "How do I rotate a deploy key?"
→ Correct behavior: The context does NOT contain deploy key instructions. Call knowledge_agent tool.

Example 4 — Route to knowledge_agent (tool call required)
User context:
- user_id: u_demo
- plan_tier: pro
User message: "what is log retention for pro plans?"
→ Correct behavior: The context only says "plan_tier: pro" but does NOT explain log retention. Call knowledge_agent tool.

Example 5 — Route to account_agent (tool call required)
User context:
- user_id: u_demo
- plan_tier: pro
User message: "show my recent builds"
→ Correct behavior: The context does NOT contain build history. Call account_agent tool.

Example 6 — Direct smalltalk (NO tool/agent call)
User context:
- user_id: u_demo
- plan_tier: pro
User message: "hi there"
→ Correct behavior: Answer directly with a greeting. NO tool call.

Example 7 — Route to escalation_agent (tool call required)
User context:
- user_id: u_demo
User message: "I need to escalate this"
→ Correct behavior: Call escalation_agent tool IMMEDIATELY.

Example 8 — Route to escalation_agent (tool call required)
User context:
- user_id: u_demo
User message: "Create a support ticket for me"
→ Correct behavior: Call escalation_agent tool IMMEDIATELY.

Routing rules (only when context does NOT contain the answer):
- HOW to do something, WHAT something is, docs/feature questions → knowledge_agent
- Their builds, detailed account status, usage requiring DB lookup → account_agent
- The user is frustrated, wants human help, or explicitly asks to open a ticket/escalate → escalation_agent
- Greetings or off-topic → respond directly, no tool call

ABSOLUTE RULES — DO NOT BREAK:
- ALWAYS check context first. If the answer is in the context, answer immediately. No exceptions.
- NEVER call an agent if the answer is already in the session context.
- NEVER call more than one agent for a single user message.
- When a specialist agent returns an answer, repeat it VERBATIM to the user. Do NOT summarize, shorten, or drop any part of it.
- Do NOT explain routing or mention tools. Return only the answer to the user.
- If the user message is a follow-up and ambiguous, prefer the previous `last_agent` from context.

Current user context is provided below. Use it.
"""


@dataclass
class OrchestratorResult:
    reply: str
    routed_to: str
    tool_calls: list[dict[str, Any]]
    retrieved_chunk_ids: list[str]


_tool_calls_var: contextvars.ContextVar[list[dict[str, Any]]] = contextvars.ContextVar(
    "tool_calls", default=[]
)
_retrieved_chunk_ids_var: contextvars.ContextVar[list[str]] = contextvars.ContextVar(
    "retrieved_chunk_ids", default=[]
)


async def run_orchestrator(session_id: str, user_message: str, session_context: str) -> OrchestratorResult:
    # Reset per invocation so cross-request state doesn't leak.
    _tool_calls_var.set([])
    _retrieved_chunk_ids_var.set([])

    runner, root_agent = _build_adk_runner(session_context)
    from google.genai.types import Content, Part

    existing_session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id="helix_user",
        session_id=session_id,
    )
    if existing_session is None:
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id="helix_user",
            session_id=session_id,
        )

    response = runner.run_async(
        user_id="helix_user",
        session_id=session_id,
        new_message=Content(parts=[Part.from_text(text=user_message)], role="user"),
    )

    final_text = ""
    specialist_text = ""
    routed_to = "smalltalk"
    routed_to_locked: str | None = None

    async for event in response:
        # ADK v1.x: prefer function call/response helpers when available.
        get_calls = getattr(event, "get_function_calls", None)
        if callable(get_calls):
            for call in get_calls() or []:
                name = getattr(call, "name", "") or ""
                args = getattr(call, "args", None) or {}
                _tool_calls_var.get().append({"tool_name": name, "args": args, "result": None})
                if name == "knowledge_agent":
                    routed_to_locked = "knowledge"
                elif name == "account_agent":
                    routed_to_locked = "account"
                elif name == "escalation_agent":
                    routed_to_locked = "escalation"

        get_responses = getattr(event, "get_function_responses", None)
        if callable(get_responses):
            for resp in get_responses() or []:
                name = getattr(resp, "name", "") or ""
                response_value = getattr(resp, "response", None)
                # Attach to the most recent matching call (best-effort).
                for record in reversed(_tool_calls_var.get()):
                    if record["tool_name"] == name and record["result"] is None:
                        record["result"] = response_value
                        break

        if hasattr(event, "is_final_response") and event.is_final_response():
            parts = getattr(getattr(event, "content", None), "parts", [])
            if parts:
                text = getattr(parts[0], "text", "")
                author = getattr(event, "author", "")
                if text:
                    # If the event is from a specialist agent, prioritize its text.
                    if "knowledge" in author or "account" in author or "escalation" in author:
                        specialist_text = text
                    else:
                        final_text = text

            # Update routing based on author if not already locked by a tool call.
            author = getattr(event, "author", "")
            if routed_to_locked is None:
                if author == getattr(root_agent, "name", "srop_root"):
                    # E5: Simple refusal detection to map to 'guardrail' category
                    refusal_keywords = [
                        "out of scope", "cannot assist", "cannot help", "firmly but politely", 
                        "not related", "cannot write", "cannot tell", "purpose is to", 
                        "unable to provide", "limited to helix", "not a poet", "not a comedian"
                    ]
                    if any(k in final_text.lower() for k in refusal_keywords):
                        routed_to = "guardrail"
                    else:
                        routed_to = "smalltalk"
                elif "knowledge" in author:
                    routed_to = "knowledge"
                elif "account" in author:
                    routed_to = "account"
                elif "escalation" in author:
                    routed_to = "escalation"

    if routed_to_locked is not None:
        routed_to = routed_to_locked

    # Final reply priority:
    # 1. Specialist agent's direct text response.
    # 2. Root agent's text response (which should be a verbatim repeat).
    # 3. Fallback: extract from the tool call result if a specialist was invoked.
    reply = specialist_text or final_text

    if not reply and routed_to in {"knowledge", "account", "escalation"}:
        for call in reversed(_tool_calls_var.get()):
            if call["tool_name"] in {"knowledge_agent", "account_agent", "escalation_agent"} and call["result"]:
                if isinstance(call["result"], dict) and "result" in call["result"]:
                    reply = call["result"]["result"]
                    break
                elif isinstance(call["result"], str):
                    reply = call["result"]
                    break

    # E5: PII Redaction in logs
    _redact_pii_in_logs(user_message, reply)

    return OrchestratorResult(
        reply=reply,
        routed_to=routed_to,
        tool_calls=_tool_calls_var.get(),
        retrieved_chunk_ids=_retrieved_chunk_ids_var.get(),
    )


def _build_adk_runner(session_context: str):
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.adk.tools.agent_tool import AgentTool

    async def traced_search_docs(query: str, k: int = 5, product_area: str | None = None):
        record: dict[str, Any] = {
            "tool_name": "search_docs",
            "args": {"query": query, "k": k, "product_area": product_area},
            "result": None,
        }
        _tool_calls_var.get().append(record)
        result = await search_docs(query=query, k=k, product_area=product_area)
        record["result"] = [
            {"chunk_id": c.chunk_id, "score": c.score, "metadata": c.metadata} for c in result
        ]
        _retrieved_chunk_ids_var.get().extend([c.chunk_id for c in result])
        return result

    traced_search_docs.__name__ = "search_docs"

    async def traced_get_recent_builds(user_id: str, limit: int = 5):
        record: dict[str, Any] = {
            "tool_name": "get_recent_builds",
            "args": {"user_id": user_id, "limit": limit},
            "result": None,
        }
        _tool_calls_var.get().append(record)
        result = await get_recent_builds(user_id=user_id, limit=limit)
        record["result"] = [
            {
                "build_id": b.build_id,
                "pipeline": b.pipeline,
                "status": b.status,
                "branch": b.branch,
                "started_at": b.started_at.isoformat() if hasattr(b.started_at, "isoformat") else str(b.started_at),
                "duration_seconds": b.duration_seconds,
            }
            for b in result
        ]
        return result

    traced_get_recent_builds.__name__ = "get_recent_builds"

    async def traced_create_ticket(user_id: str, summary: str, priority: str = "medium"):
        record: dict[str, Any] = {
            "tool_name": "create_ticket",
            "args": {"user_id": user_id, "summary": summary, "priority": priority},
            "result": None,
        }
        _tool_calls_var.get().append(record)
        result = await create_ticket(user_id=user_id, summary=summary, priority=priority)
        record["result"] = result
        return result

    traced_create_ticket.__name__ = "create_ticket"

    async def traced_get_account_status(user_id: str):
        record: dict[str, Any] = {
            "tool_name": "get_account_status",
            "args": {"user_id": user_id},
            "result": None,
        }
        _tool_calls_var.get().append(record)
        result = await get_account_status(user_id=user_id)
        record["result"] = result.__dict__
        return result

    traced_get_account_status.__name__ = "get_account_status"

    knowledge_agent = LlmAgent(
        name="knowledge_agent",
        model=settings.adk_model,
        instruction=(
            "You answer Helix product questions using the search_docs tool. "
            "Always call search_docs before answering and always cite chunk IDs like [chunk_abc123]."
        ),
        tools=[traced_search_docs],
    )
    account_agent = LlmAgent(
        name="account_agent",
        model=settings.adk_model,
        instruction=(
            "You answer account, plan, usage, and builds questions. "
            f"FIRST check if the answer is already in the session context:\n{session_context}\n"
            "If the answer is in the context, use it directly without calling tools. "
            "Only use tools for data not available in context (like detailed build history). "
            "When calling tools, extract the user_id from the session context."
        ),
        tools=[traced_get_recent_builds, traced_get_account_status],
    )
    escalation_agent = LlmAgent(
        name="escalation_agent",
        model=settings.adk_model,
        instruction=(
            "You help users escalate issues by creating support tickets. "
            f"REQUIRED CONTEXT:\n{session_context}\n"
            "DIRECTIONS:\n"
            "1. Extract the user_id from the 'REQUIRED CONTEXT' above. Even if it is 'string' or 'u_demo', use it exactly as provided.\n"
            "2. Use the user's current message as the summary of the issue.\n"
            "3. CALL create_ticket IMMEDIATELY using that user_id and summary. Do NOT ask the user for this information; it is already in the context.\n"
            "4. After calling the tool, confirm the ticket ID to the user."
        ),
        tools=[traced_create_ticket],
    )
    root_agent = LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        instruction=f"{ROOT_INSTRUCTION}\n\nCurrent user context:\n{session_context}",
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=account_agent),
            AgentTool(agent=escalation_agent),
        ],
    )
    return InMemoryRunner(agent=root_agent), root_agent


def _redact_pii_in_logs(user_msg: str, assistant_reply: str) -> None:
    """Helper to redact potential PII (emails, phone numbers) from log output."""
    import re
    import logging

    logger = logging.getLogger("app.pii_guard")
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"

    def redact(text: str) -> str:
        text = re.sub(email_pattern, "[EMAIL_REDACTED]", text)
        text = re.sub(phone_pattern, "[PHONE_REDACTED]", text)
        return text

    logger.info(f"User (redacted): {redact(user_msg)}")
    logger.info(f"Assistant (redacted): {redact(assistant_reply)}")
