"""Enumeration types for pipecat utilities."""

from enum import Enum


class EndTaskReason(Enum):
    """Reasons for ending a task."""

    CALL_DURATION_EXCEEDED = "call_duration_exceeded"
    VOICEMAIL_DETECTED = "voicemail_detected"
    USER_IDLE_MAX_DURATION_EXCEEDED = "user_idle_max_duration_exceeded"
    USER_HANGUP = "user_hangup"
    USER_QUALIFIED = "user_qualified"
    USER_DISQUALIFIED = "user_disqualified"
    SYSTEM_CANCELLED = "system_cancelled"
    SYSTEM_CONNECT_ERROR = "system_connect_error"
    UNKNOWN = "unknown"
