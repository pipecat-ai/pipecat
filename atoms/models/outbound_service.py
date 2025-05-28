from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class OutboundCallRequest(BaseModel):
    """Outbound call request model.

    Attributes:
        from_phone: The phone number of the caller.
        to_phone: The phone number of the callee.
        agent_id: The agent ID to use for the call.
        provider: The telephony provider to use (plivo or twilio).
        tts_service: The TTS service to use (waves or cartesia).
        stt_service: The STT service to use (deepgram, groq_whisper, or openai).
        krisp_enabled: Whether to enable Krisp noise suppression.
        voice_id: The voice ID to use for TTS.
    """

    from_phone: Optional[str] = Field(default=None, description="The phone number of the caller")
    to_phone: str = Field(description="The phone number of the callee")
    agent_id: str = Field(description="The agent ID to use for the call")
    provider: Optional[str] = Field(
        default="plivo", description="The telephony provider (plivo or twilio)"
    )
    tts_service: Optional[str] = Field(
        default="waves", description="The TTS service (waves or cartesia)"
    )
    stt_service: Optional[str] = Field(
        default="deepgram", description="The STT service (deepgram, groq_whisper, or openai)"
    )
    krisp_enabled: Optional[bool] = Field(
        default=True, description="Whether to enable Krisp noise suppression"
    )
    voice_id: Optional[str] = Field(default="deepika", description="The voice ID to use for TTS")
