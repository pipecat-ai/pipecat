"""Session models for voice pipeline management"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class SessionStatus(str, Enum):
    """Session lifecycle status"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class SessionType(str, Enum):
    """Type of session connection"""
    WEBSOCKET = "websocket"
    TWILIO = "twilio"
    WEBRTC = "webrtc"


class SessionBase(BaseModel):
    """Base session model"""
    user_id: Optional[str] = None
    session_type: SessionType = SessionType.WEBSOCKET
    voice_config: str = "conversational"  # References VOICE_CONFIGS
    system_prompt: str = "default"  # References SYSTEM_PROMPTS
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionCreate(SessionBase):
    """Session creation model"""
    pass


class Session(SessionBase):
    """Session model returned to clients"""
    id: str
    status: SessionStatus
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Metrics
    messages_sent: int = 0
    messages_received: int = 0
    total_audio_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)

    # Connection details
    remote_addr: Optional[str] = None
    user_agent: Optional[str] = None

    # Twilio specific
    call_sid: Optional[str] = None
    stream_sid: Optional[str] = None

    class Config:
        from_attributes = True


class SessionMetrics(BaseModel):
    """Real-time session metrics"""
    session_id: str
    status: SessionStatus
    uptime_seconds: float
    messages_sent: int
    messages_received: int
    latency_ms: Optional[float] = None
    audio_quality: Optional[float] = None  # 0-1 score


class SessionListResponse(BaseModel):
    """Paginated session list response"""
    sessions: List[Session]
    total: int
    page: int
    page_size: int
