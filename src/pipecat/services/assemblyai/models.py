from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Word(BaseModel):
    """Represents a single word in a transcription with timing and confidence."""

    start: int
    end: int
    text: str
    confidence: float
    word_is_final: bool = Field(..., alias="word_is_final")


class BaseMessage(BaseModel):
    """Base class for all AssemblyAI WebSocket messages."""

    type: str


class BeginMessage(BaseMessage):
    """Message sent when a new session begins."""

    type: Literal["Begin"] = "Begin"
    id: str
    expires_at: int


class TurnMessage(BaseMessage):
    """Message containing transcription data for a turn of speech."""

    type: Literal["Turn"] = "Turn"
    turn_order: int
    turn_is_formatted: bool
    end_of_turn: bool
    transcript: str
    end_of_turn_confidence: float
    words: List[Word]


class TerminationMessage(BaseMessage):
    """Message sent when the session is terminated."""

    type: Literal["Termination"] = "Termination"
    audio_duration_seconds: float
    session_duration_seconds: float


# Union type for all possible message types
AnyMessage = BeginMessage | TurnMessage | TerminationMessage


class AssemblyAIConnectionParams(BaseModel):
    sample_rate: int = 16000
    encoding: Literal["pcm_s16le", "pcm_mulaw"] = "pcm_s16le"
    formatted_finals: bool = True
    word_finalization_max_wait_time: Optional[int] = None
    end_of_turn_confidence_threshold: Optional[float] = None
    min_end_of_turn_silence_when_confident: Optional[int] = None
    max_turn_silence: Optional[int] = None
