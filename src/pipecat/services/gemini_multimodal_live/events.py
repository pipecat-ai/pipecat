#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Event models and utilities for Google Gemini Multimodal Live API."""

import base64
import io
import json
from enum import Enum
from typing import List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field

from pipecat.frames.frames import ImageRawFrame

#
# Client events
#


class MediaChunk(BaseModel):
    """Represents a chunk of media data for transmission.

    Parameters:
        mimeType: MIME type of the media content.
        data: Base64-encoded media data.
    """

    mimeType: str
    data: str


class ContentPart(BaseModel):
    """Represents a part of content that can contain text or media.

    Parameters:
        text: Text content. Defaults to None.
        inlineData: Inline media data. Defaults to None.
    """

    text: Optional[str] = Field(default=None, validate_default=False)
    inlineData: Optional[MediaChunk] = Field(default=None, validate_default=False)
    fileData: Optional["FileData"] = Field(default=None, validate_default=False)


class FileData(BaseModel):
    """Represents a file reference in the Gemini File API."""

    mimeType: str
    fileUri: str


ContentPart.model_rebuild()  # Rebuild model to resolve forward reference


class Turn(BaseModel):
    """Represents a conversational turn in the dialogue.

    Parameters:
        role: The role of the speaker, either "user" or "model". Defaults to "user".
        parts: List of content parts that make up the turn.
    """

    role: Literal["user", "model"] = "user"
    parts: List[ContentPart]


class StartSensitivity(str, Enum):
    """Determines how start of speech is detected."""

    UNSPECIFIED = "START_SENSITIVITY_UNSPECIFIED"  # Default is HIGH
    HIGH = "START_SENSITIVITY_HIGH"  # Detect start of speech more often
    LOW = "START_SENSITIVITY_LOW"  # Detect start of speech less often


class EndSensitivity(str, Enum):
    """Determines how end of speech is detected."""

    UNSPECIFIED = "END_SENSITIVITY_UNSPECIFIED"  # Default is HIGH
    HIGH = "END_SENSITIVITY_HIGH"  # End speech more often
    LOW = "END_SENSITIVITY_LOW"  # End speech less often


class AutomaticActivityDetection(BaseModel):
    """Configures automatic detection of voice activity.

    Parameters:
        disabled: Whether automatic activity detection is disabled. Defaults to None.
        start_of_speech_sensitivity: Sensitivity for detecting speech start. Defaults to None.
        prefix_padding_ms: Padding before speech start in milliseconds. Defaults to None.
        end_of_speech_sensitivity: Sensitivity for detecting speech end. Defaults to None.
        silence_duration_ms: Duration of silence to detect speech end. Defaults to None.
    """

    disabled: Optional[bool] = None
    start_of_speech_sensitivity: Optional[StartSensitivity] = None
    prefix_padding_ms: Optional[int] = None
    end_of_speech_sensitivity: Optional[EndSensitivity] = None
    silence_duration_ms: Optional[int] = None


class RealtimeInputConfig(BaseModel):
    """Configures the realtime input behavior.

    Parameters:
        automatic_activity_detection: Voice activity detection configuration. Defaults to None.
    """

    automatic_activity_detection: Optional[AutomaticActivityDetection] = None


class RealtimeInput(BaseModel):
    """Contains realtime input media chunks and text.

    Parameters:
        mediaChunks: List of media chunks for realtime processing.
        text: Text for realtime processing.
    """

    mediaChunks: Optional[List[MediaChunk]] = None
    text: Optional[str] = None


class ClientContent(BaseModel):
    """Content sent from client to the Gemini Live API.

    Parameters:
        turns: List of conversation turns. Defaults to None.
        turnComplete: Whether the client's turn is complete. Defaults to False.
    """

    turns: Optional[List[Turn]] = None
    turnComplete: bool = False


class AudioInputMessage(BaseModel):
    """Message containing audio input data.

    Parameters:
        realtimeInput: Realtime input containing audio chunks.
    """

    realtimeInput: RealtimeInput

    @classmethod
    def from_raw_audio(cls, raw_audio: bytes, sample_rate: int) -> "AudioInputMessage":
        """Create an audio input message from raw audio data.

        Args:
            raw_audio: Raw audio bytes.
            sample_rate: Audio sample rate in Hz.

        Returns:
            AudioInputMessage instance with encoded audio data.
        """
        data = base64.b64encode(raw_audio).decode("utf-8")
        return cls(
            realtimeInput=RealtimeInput(
                mediaChunks=[MediaChunk(mimeType=f"audio/pcm;rate={sample_rate}", data=data)]
            )
        )


class VideoInputMessage(BaseModel):
    """Message containing video/image input data.

    Parameters:
        realtimeInput: Realtime input containing video/image chunks.
    """

    realtimeInput: RealtimeInput

    @classmethod
    def from_image_frame(cls, frame: ImageRawFrame) -> "VideoInputMessage":
        """Create a video input message from an image frame.

        Args:
            frame: Image frame to encode.

        Returns:
            VideoInputMessage instance with encoded image data.
        """
        buffer = io.BytesIO()
        Image.frombytes(frame.format, frame.size, frame.image).save(buffer, format="JPEG")
        data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return cls(
            realtimeInput=RealtimeInput(mediaChunks=[MediaChunk(mimeType=f"image/jpeg", data=data)])
        )


class TextInputMessage(BaseModel):
    """Message containing text input data."""

    realtimeInput: RealtimeInput

    @classmethod
    def from_text(cls, text: str) -> "TextInputMessage":
        """Create a text input message from a string.

        Args:
            text: The text to send.

        Returns:
            A TextInputMessage instance.
        """
        return cls(realtimeInput=RealtimeInput(text=text))


class ClientContentMessage(BaseModel):
    """Message containing client content for the API.

    Parameters:
        clientContent: The client content to send.
    """

    clientContent: ClientContent


class SystemInstruction(BaseModel):
    """System instruction for the model.

    Parameters:
        parts: List of content parts that make up the system instruction.
    """

    parts: List[ContentPart]


class AudioTranscriptionConfig(BaseModel):
    """Configuration for audio transcription."""

    pass


class Setup(BaseModel):
    """Setup configuration for the Gemini Live session.

    Parameters:
        model: Model identifier to use.
        system_instruction: System instruction for the model. Defaults to None.
        tools: List of available tools/functions. Defaults to None.
        generation_config: Generation configuration parameters. Defaults to None.
        input_audio_transcription: Input audio transcription config. Defaults to None.
        output_audio_transcription: Output audio transcription config. Defaults to None.
        realtime_input_config: Realtime input configuration. Defaults to None.
    """

    model: str
    system_instruction: Optional[SystemInstruction] = None
    tools: Optional[List[dict]] = None
    generation_config: Optional[dict] = None
    input_audio_transcription: Optional[AudioTranscriptionConfig] = None
    output_audio_transcription: Optional[AudioTranscriptionConfig] = None
    realtime_input_config: Optional[RealtimeInputConfig] = None


class Config(BaseModel):
    """Configuration message for session setup.

    Parameters:
        setup: Setup configuration for the session.
    """

    setup: Setup


#
# Grounding metadata models
#


class SearchEntryPoint(BaseModel):
    """Represents the search entry point with rendered content for search suggestions."""

    renderedContent: Optional[str] = None


class WebSource(BaseModel):
    """Represents a web source from grounding chunks."""

    uri: Optional[str] = None
    title: Optional[str] = None


class GroundingChunk(BaseModel):
    """Represents a grounding chunk containing web source information."""

    web: Optional[WebSource] = None


class GroundingSegment(BaseModel):
    """Represents a segment of text that is grounded."""

    startIndex: Optional[int] = None
    endIndex: Optional[int] = None
    text: Optional[str] = None


class GroundingSupport(BaseModel):
    """Represents support information for grounded text segments."""

    segment: Optional[GroundingSegment] = None
    groundingChunkIndices: Optional[List[int]] = None
    confidenceScores: Optional[List[float]] = None


class GroundingMetadata(BaseModel):
    """Represents grounding metadata from Google Search."""

    searchEntryPoint: Optional[SearchEntryPoint] = None
    groundingChunks: Optional[List[GroundingChunk]] = None
    groundingSupports: Optional[List[GroundingSupport]] = None
    webSearchQueries: Optional[List[str]] = None


#
# Server events
#


class SetupComplete(BaseModel):
    """Indicates that session setup is complete."""

    pass


class InlineData(BaseModel):
    """Inline data embedded in server responses.

    Parameters:
        mimeType: MIME type of the data.
        data: Base64-encoded data content.
    """

    mimeType: str
    data: str


class Part(BaseModel):
    """Part of a server response containing data or text.

    Parameters:
        inlineData: Inline binary data. Defaults to None.
        text: Text content. Defaults to None.
    """

    inlineData: Optional[InlineData] = None
    text: Optional[str] = None


class ModelTurn(BaseModel):
    """Represents a turn from the model in the conversation.

    Parameters:
        parts: List of content parts in the model's response.
    """

    parts: List[Part]


class ServerContentInterrupted(BaseModel):
    """Indicates server content was interrupted.

    Parameters:
        interrupted: Whether the content was interrupted.
    """

    interrupted: bool


class ServerContentTurnComplete(BaseModel):
    """Indicates the server's turn is complete.

    Parameters:
        turnComplete: Whether the turn is complete.
    """

    turnComplete: bool


class BidiGenerateContentTranscription(BaseModel):
    """Transcription data from bidirectional content generation.

    Parameters:
        text: The transcribed text content.
    """

    text: str


class ServerContent(BaseModel):
    """Content sent from server to client.

    Parameters:
        modelTurn: Model's conversational turn. Defaults to None.
        interrupted: Whether content was interrupted. Defaults to None.
        turnComplete: Whether the turn is complete. Defaults to None.
        inputTranscription: Transcription of input audio. Defaults to None.
        outputTranscription: Transcription of output audio. Defaults to None.
    """

    modelTurn: Optional[ModelTurn] = None
    interrupted: Optional[bool] = None
    turnComplete: Optional[bool] = None
    inputTranscription: Optional[BidiGenerateContentTranscription] = None
    outputTranscription: Optional[BidiGenerateContentTranscription] = None
    groundingMetadata: Optional[GroundingMetadata] = None


class FunctionCall(BaseModel):
    """Represents a function call from the model.

    Parameters:
        id: Unique identifier for the function call.
        name: Name of the function to call.
        args: Arguments to pass to the function.
    """

    id: str
    name: str
    args: dict


class ToolCall(BaseModel):
    """Contains one or more function calls.

    Parameters:
        functionCalls: List of function calls to execute.
    """

    functionCalls: List[FunctionCall]


class Modality(str, Enum):
    """Modality types in token counts."""

    UNSPECIFIED = "MODALITY_UNSPECIFIED"
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"


class ModalityTokenCount(BaseModel):
    """Token count for a specific modality.

    Parameters:
        modality: The modality type.
        tokenCount: Number of tokens for this modality.
    """

    modality: Modality
    tokenCount: int


class UsageMetadata(BaseModel):
    """Usage metadata about the API response.

    Parameters:
        promptTokenCount: Number of tokens in the prompt. Defaults to None.
        cachedContentTokenCount: Number of cached content tokens. Defaults to None.
        responseTokenCount: Number of tokens in the response. Defaults to None.
        toolUsePromptTokenCount: Number of tokens for tool use prompts. Defaults to None.
        thoughtsTokenCount: Number of tokens for model thoughts. Defaults to None.
        totalTokenCount: Total number of tokens used. Defaults to None.
        promptTokensDetails: Detailed breakdown of prompt tokens by modality. Defaults to None.
        cacheTokensDetails: Detailed breakdown of cache tokens by modality. Defaults to None.
        responseTokensDetails: Detailed breakdown of response tokens by modality. Defaults to None.
        toolUsePromptTokensDetails: Detailed breakdown of tool use tokens by modality. Defaults to None.
    """

    promptTokenCount: Optional[int] = None
    cachedContentTokenCount: Optional[int] = None
    responseTokenCount: Optional[int] = None
    toolUsePromptTokenCount: Optional[int] = None
    thoughtsTokenCount: Optional[int] = None
    totalTokenCount: Optional[int] = None
    promptTokensDetails: Optional[List[ModalityTokenCount]] = None
    cacheTokensDetails: Optional[List[ModalityTokenCount]] = None
    responseTokensDetails: Optional[List[ModalityTokenCount]] = None
    toolUsePromptTokensDetails: Optional[List[ModalityTokenCount]] = None


class ServerEvent(BaseModel):
    """Server event received from the Gemini Live API.

    Parameters:
        setupComplete: Setup completion notification. Defaults to None.
        serverContent: Content from the server. Defaults to None.
        toolCall: Tool/function call request. Defaults to None.
        usageMetadata: Token usage metadata. Defaults to None.
    """

    setupComplete: Optional[SetupComplete] = None
    serverContent: Optional[ServerContent] = None
    toolCall: Optional[ToolCall] = None
    usageMetadata: Optional[UsageMetadata] = None


def parse_server_event(str):
    """Parse a server event from JSON string.

    Args:
        str: JSON string containing the server event.

    Returns:
        ServerEvent instance if parsing succeeds, None otherwise.
    """
    try:
        evt = json.loads(str)
        return ServerEvent.model_validate(evt)
    except Exception as e:
        print(f"Error parsing server event: {e}")
        return None


class ContextWindowCompressionConfig(BaseModel):
    """Configuration for context window compression.

    Parameters:
        sliding_window: Whether to use sliding window compression. Defaults to True.
        trigger_tokens: Token count threshold to trigger compression. Defaults to None.
    """

    sliding_window: Optional[bool] = Field(default=True)
    trigger_tokens: Optional[int] = Field(default=None)
