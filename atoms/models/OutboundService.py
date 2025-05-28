from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class OutboundCallRequest(BaseModel):
    """Outbound call request model.

    Attributes:
        from_phone: The phone number of the caller.
        to_phone: The phone number of the callee.
    """

    from_phone: Optional[str] = Field(default=None)
    to_phone: str
    agent_id: str
    provider: Optional[str] = Field(default="plivo")
