"""
Escalation tools — used by EscalationAgent.
"""
import uuid
from datetime import datetime
from sqlalchemy import insert
from app.db.models import Ticket
from app.db.session import get_db_context

async def create_ticket(user_id: str, summary: str, priority: str = "medium") -> str:
    """
    Create a support ticket in the database.
    
    Args:
        user_id: user who is reporting the issue
        summary: brief description of the problem
        priority: low | medium | high
        
    Returns:
        The newly created ticket_id.
    """
    ticket_id = f"TICK-{str(uuid.uuid4())[:8].upper()}"
    async with get_db_context() as db:
        await db.execute(
            insert(Ticket).values(
                ticket_id=ticket_id,
                user_id=user_id,
                summary=summary,
                priority=priority,
                created_at=datetime.utcnow()
            )
        )
        await db.commit()
    return ticket_id
