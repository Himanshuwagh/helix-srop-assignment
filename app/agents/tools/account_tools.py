"""
Account tools — used by AccountAgent.

These tools query the DB for user-specific data.
Mock data is acceptable for the take-home; the integration matters.

TODO for candidate: implement these tools.
"""
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str  # passed | failed | cancelled
    branch: str
    started_at: datetime
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


async def get_recent_builds(user_id: str, limit: int = 5) -> list[BuildSummary]:
    """
    Return the most recent builds for a user, newest first.

    For the take-home: returning mock/seeded data is fine.
    The key evaluation point is that this is wired as an ADK tool
    and the agent correctly invokes it when the user asks about builds.
    """
    now = datetime.now(UTC)
    statuses = ["failed", "passed", "cancelled", "failed", "passed"]
    builds: list[BuildSummary] = []
    for index in range(max(limit, 1)):
        builds.append(
            BuildSummary(
                build_id=f"build_{user_id}_{index + 1}",
                pipeline="helix-ci",
                status=statuses[index % len(statuses)],
                branch="main" if index % 2 == 0 else "release",
                started_at=now - timedelta(hours=index + 1),
                duration_seconds=180 + (index * 25),
            )
        )
    return builds[:limit]


async def get_account_status(user_id: str) -> AccountStatus:
    """
    Return current account status (plan, usage limits).

    For the take-home: mock data is fine.
    """
    plan_tier = "enterprise" if "ent" in user_id else "pro" if "pro" in user_id else "free"
    concurrent_limit = 20 if plan_tier == "enterprise" else 5 if plan_tier == "pro" else 2
    storage_limit = 500.0 if plan_tier == "enterprise" else 100.0 if plan_tier == "pro" else 10.0
    return AccountStatus(
        user_id=user_id,
        plan_tier=plan_tier,
        concurrent_builds_used=2 if plan_tier != "free" else 1,
        concurrent_builds_limit=concurrent_limit,
        storage_used_gb=12.5 if plan_tier != "free" else 4.2,
        storage_limit_gb=storage_limit,
    )
