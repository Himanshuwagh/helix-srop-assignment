"""
Session state schema — persisted in sessions.state (JSON column).

Stores stable session metadata plus a bounded recent-turn memory so follow-up
questions can be answered after process restarts.
"""
from typing import Literal
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    user_message: str
    assistant_response: str


class SessionState(BaseModel):
    user_id: str
    plan_tier: Literal["free", "pro", "enterprise"] = "free"
    last_agent: Literal["knowledge", "account", "smalltalk", "escalation", "guardrail"] | None = None
    turn_count: int = 0
    recent_turns: list[ConversationTurn] = Field(default_factory=list)
    ticket_ids: list[str] = Field(default_factory=list)

    def append_turn(self, user_message: str, assistant_response: str) -> None:
        self.recent_turns.append(
            ConversationTurn(
                user_message=user_message,
                assistant_response=assistant_response,
            )
        )
        self.recent_turns = self.recent_turns[-5:]

    def to_db_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_db_dict(cls, data: dict) -> "SessionState":
        return cls.model_validate(data)
