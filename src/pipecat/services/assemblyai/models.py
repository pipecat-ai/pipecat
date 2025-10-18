#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AssemblyAI WebSocket API message models and connection parameters.

This module defines Pydantic models for handling AssemblyAI's real-time
transcription WebSocket messages and connection configuration.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Word(BaseModel):
    """Represents a single word in a transcription with timing and confidence.

    Parameters:
        start: Start time of the word in milliseconds.
        end: End time of the word in milliseconds.
        text: The transcribed word text.
        confidence: Confidence score for the word (0.0 to 1.0).
        word_is_final: Whether this word is finalized and won't change.
    """

    start: int
    end: int
    text: str
    confidence: float
    word_is_final: bool = Field(..., alias="word_is_final")


class BaseMessage(BaseModel):
    """Base class for all AssemblyAI WebSocket messages.

    Parameters:
        type: The message type identifier.
    """

    type: str


class BeginMessage(BaseMessage):
    """Message sent when a new session begins.

    Parameters:
        type: Always "Begin" for this message type.
        id: Unique session identifier.
        expires_at: Unix timestamp when the session expires.
    """

    type: Literal["Begin"] = "Begin"
    id: str
    expires_at: int


class TurnMessage(BaseMessage):
    """Message containing transcription data for a turn of speech.

    Parameters:
        type: Always "Turn" for this message type.
        turn_order: Sequential number of this turn in the session.
        turn_is_formatted: Whether the transcript has been formatted.
        end_of_turn: Whether this marks the end of a speaking turn.
        transcript: The transcribed text for this turn.
        end_of_turn_confidence: Confidence score for end-of-turn detection.
        words: List of individual words with timing and confidence data.
    """

    type: Literal["Turn"] = "Turn"
    turn_order: int
    turn_is_formatted: bool
    end_of_turn: bool
    transcript: str
    end_of_turn_confidence: float
    words: List[Word]


class TerminationMessage(BaseMessage):
    """Message sent when the session is terminated.

    Parameters:
        type: Always "Termination" for this message type.
        audio_duration_seconds: Total duration of audio processed.
        session_duration_seconds: Total duration of the session.
    """

    type: Literal["Termination"] = "Termination"
    audio_duration_seconds: float
    session_duration_seconds: float


# Union type for all possible message types
AnyMessage = BeginMessage | TurnMessage | TerminationMessage


class AssemblyAIConnectionParams(BaseModel):
    """Configuration parameters for AssemblyAI WebSocket connection.

    Parameters:
        sample_rate: Audio sample rate in Hz. Defaults to 16000.
        encoding: Audio encoding format. Defaults to "pcm_s16le".
        formatted_finals: Whether to enable transcript formatting. Defaults to True.
        word_finalization_max_wait_time: Maximum time to wait for word finalization in milliseconds.
        end_of_turn_confidence_threshold: Confidence threshold for end-of-turn detection.
        min_end_of_turn_silence_when_confident: Minimum silence duration when confident about end-of-turn.
        max_turn_silence: Maximum silence duration before forcing end-of-turn.
        keyterms_prompt: List of key terms to guide transcription. Will be JSON serialized before sending.
        speech_model: Select between English and multilingual models. Defaults to "universal-streaming-english".
    """

    sample_rate: int = 16000
    encoding: Literal["pcm_s16le", "pcm_mulaw"] = "pcm_s16le"
    formatted_finals: bool = True
    word_finalization_max_wait_time: Optional[int] = None
    end_of_turn_confidence_threshold: Optional[float] = None
    min_end_of_turn_silence_when_confident: Optional[int] = None
    max_turn_silence: Optional[int] = None
    keyterms_prompt: Optional[List[str]] = None
    speech_model: Literal["universal-streaming-english", "universal-streaming-multilingual"] = (
        "universal-streaming-english"
    )
