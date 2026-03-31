#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AssemblyAI WebSocket API message models and connection parameters.

This module defines Pydantic models for handling AssemblyAI's real-time
transcription WebSocket messages and connection configuration.
"""

from typing import List, Literal, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
        language_code: Detected language code (e.g., "es", "fr"). Only present with
            complete utterances or when end_of_turn is True.
        language_confidence: Confidence score (0-1) for language detection. Only present
            with complete utterances or when end_of_turn is True.
        speaker: Speaker label (e.g., "A", "B"). Only present when speaker_labels is
            enabled and end_of_turn is True. Maps to 'speaker_label' in JSON response.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["Turn"] = "Turn"
    turn_order: int
    turn_is_formatted: bool
    end_of_turn: bool
    transcript: str
    end_of_turn_confidence: float
    words: List[Word]
    language_code: Optional[str] = None
    language_confidence: Optional[float] = None
    speaker: Optional[str] = Field(default=None, alias="speaker_label")


class SpeechStartedMessage(BaseMessage):
    """Message sent when speech is first detected in the audio stream.

    Parameters:
        type: Always "SpeechStarted" for this message type.
        timestamp: Audio timestamp in milliseconds when speech was detected.
    """

    type: Literal["SpeechStarted"] = "SpeechStarted"
    timestamp: int


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
AnyMessage = BeginMessage | TurnMessage | SpeechStartedMessage | TerminationMessage


class AssemblyAIConnectionParams(BaseModel):
    """Configuration parameters for AssemblyAI WebSocket connection.

    .. deprecated:: 0.0.105
        Use ``settings=AssemblyAISTTService.Settings(foo=...)`` instead.

    Parameters:
        sample_rate: Audio sample rate in Hz. Defaults to 16000.
        encoding: Audio encoding format. Defaults to "pcm_s16le".
        end_of_turn_confidence_threshold: Confidence threshold for end-of-turn detection.
        min_turn_silence: Minimum silence duration when confident about end-of-turn.
        min_end_of_turn_silence_when_confident: DEPRECATED. Use min_turn_silence instead.
        max_turn_silence: Maximum silence duration before forcing end-of-turn.
        keyterms_prompt: List of key terms to guide transcription. Will be JSON serialized before sending.
        prompt: Optional text prompt to guide the transcription. Only used when speech_model is "u3-rt-pro".
        speech_model: Select between English, multilingual, and u3-rt-pro models. Defaults to "u3-rt-pro".
        language_detection: Enable automatic language detection. Only applicable to
            universal-streaming-multilingual. When enabled, Turn messages include
            language_code and language_confidence fields. Defaults to None (not sent).
        format_turns: Whether to format transcript turns. Only applicable to
            universal-streaming-english and universal-streaming-multilingual models.
            For u3-rt-pro, formatting is automatic and built-in. Defaults to True.
        speaker_labels: Enable speaker diarization. When enabled, final transcripts
            (end_of_turn=True) include a speaker field identifying the speaker
            (e.g., "Speaker A", "Speaker B"). Defaults to None (not sent).
        vad_threshold: Voice activity detection confidence threshold. Only applicable to
            u3-rt-pro. The confidence threshold (0.0 to 1.0) for classifying audio frames
            as silence. Frames with VAD confidence below this value are considered silent.
            Increase for noisy environments to reduce false speech detection. Defaults to
            0.3 (API default). For best performance when using with external VAD (e.g., Silero),
            align this value with your VAD's activation threshold to avoid the "dead zone"
            where AssemblyAI transcribes speech that your VAD hasn't detected yet.
            Defaults to None (not sent).
    """

    sample_rate: int = 16000
    encoding: Literal["pcm_s16le", "pcm_mulaw"] = "pcm_s16le"
    end_of_turn_confidence_threshold: Optional[float] = None
    min_turn_silence: Optional[int] = None
    min_end_of_turn_silence_when_confident: Optional[int] = None  # Deprecated
    max_turn_silence: Optional[int] = None
    keyterms_prompt: Optional[List[str]] = None
    prompt: Optional[str] = None
    speech_model: Literal[
        "universal-streaming-english", "universal-streaming-multilingual", "u3-rt-pro"
    ] = "u3-rt-pro"
    language_detection: Optional[bool] = None
    format_turns: bool = True
    speaker_labels: Optional[bool] = None
    vad_threshold: Optional[float] = None

    @model_validator(mode="after")
    def handle_deprecated_param(self):
        """Handle deprecated min_end_of_turn_silence_when_confident parameter."""
        if self.min_end_of_turn_silence_when_confident is not None:
            logger.warning(
                "The 'min_end_of_turn_silence_when_confident' parameter is deprecated and will be "
                "removed in a future version. Please use 'min_turn_silence' instead."
            )
            # If min_turn_silence is not set, use the deprecated value
            if self.min_turn_silence is None:
                self.min_turn_silence = self.min_end_of_turn_silence_when_confident
        return self
