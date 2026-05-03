"""
Test fixtures.

Key fixtures:
- `client`: async test client with in-memory SQLite DB
- `mock_adk`: patches the ADK root agent so tests don't hit the real LLM
- `seeded_db`: DB with a test user and session pre-created
"""
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.orchestrator import OrchestratorResult
from app.db.models import Base
from app.db.session import get_db
from app.main import app


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db):
    """Async test client with DB overridden to in-memory SQLite."""

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch):
    """
    Patch the ADK pipeline so tests don't call the real LLM.

    TODO for candidate: patch at the ADK boundary (not at the HTTP layer).
    The mock should:
    1. Accept a user message
    2. Return a canned response with a specified routed_to value
    3. Allow tests to assert which sub-agent was called

    Example:
        def mock_run(session_id, message, db):
            if "rotate" in message.lower():
                return PipelineResult(
                    content="To rotate a deploy key...",
                    routed_to="knowledge",
                    trace_id="test-trace-001",
                )
            ...

        monkeypatch.setattr("app.srop.pipeline.run", mock_run)
    """
    async def fake_run_orchestrator(
        session_id: str,
        user_message: str,
        session_context: str,
    ) -> OrchestratorResult:
        if "rotate" in user_message.lower():
            return OrchestratorResult(
                reply="Rotate the deploy key from Settings > Deploy Keys [chunk_test_001].",
                routed_to="knowledge",
                tool_calls=[
                    {
                        "tool_name": "search_docs",
                        "args": {"query": user_message, "k": 5},
                        "result": [
                            {
                                "chunk_id": "chunk_test_001",
                                "score": 0.95,
                                "content": "Rotate deploy keys in the Deploy Keys settings page.",
                                "metadata": {"product_area": "security", "title": "Deploy Keys"},
                            }
                        ],
                    }
                ],
                retrieved_chunk_ids=["chunk_test_001"],
            )
        if "plan tier" in user_message.lower():
            return OrchestratorResult(
                reply=_reply_from_context(session_context),
                routed_to="account",
                tool_calls=[
                    {
                        "tool_name": "get_account_status",
                        "args": {"user_id": "u_test_002"},
                        "result": {"plan_tier": "pro"},
                    }
                ],
                retrieved_chunk_ids=[],
            )
        if "last 3 builds" in user_message.lower():
            return OrchestratorResult(
                reply="Here are your last 3 builds:\n- build_u_test_002_1: failed\n- build_u_test_002_2: passed\n- build_u_test_002_3: cancelled",
                routed_to="account",
                tool_calls=[
                    {
                        "tool_name": "get_recent_builds",
                        "args": {"user_id": "u_test_002", "limit": 3},
                        "result": [
                            {"build_id": "build_u_test_002_1", "status": "failed"},
                            {"build_id": "build_u_test_002_2", "status": "passed"},
                            {"build_id": "build_u_test_002_3", "status": "cancelled"},
                        ],
                    }
                ],
                retrieved_chunk_ids=[],
            )
        if "most recent one" in user_message.lower():
            return OrchestratorResult(
                reply=_reply_from_recent_conversation(session_context),
                routed_to="account",
                tool_calls=[
                    {
                        "tool_name": "get_recent_builds",
                        "args": {"user_id": "u_test_002", "limit": 3},
                        "result": [{"build_id": "build_u_test_002_1", "status": "failed"}],
                    }
                ],
                retrieved_chunk_ids=[],
            )
        return OrchestratorResult(
            reply="Hello from the mock ADK boundary.",
            routed_to="smalltalk",
            tool_calls=[],
            retrieved_chunk_ids=[],
        )

    monkeypatch.setattr("app.srop.pipeline.run_orchestrator", fake_run_orchestrator)


def _reply_from_context(session_context: str) -> str:
    for line in session_context.splitlines():
        if line.startswith("- plan_tier:"):
            return f"Your plan tier is {line.split(':', 1)[1].strip()}."
    return "Your plan tier is unknown."


def _reply_from_recent_conversation(session_context: str) -> str:
    if "build_u_test_002_1: failed" in session_context:
        return "The most recent build was build_u_test_002_1, and its status was failed."
    return "I need more context about which build you mean."
