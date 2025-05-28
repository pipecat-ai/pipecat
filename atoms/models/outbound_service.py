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
    tts_service: Optional[str] = Field(default="smallest")
    stt_service: Optional[str] = Field(default="deepgram")
    krisp_enabled: Optional[bool] = Field(default=True)
    voice_id: Optional[str] = Field(default="deepika")
