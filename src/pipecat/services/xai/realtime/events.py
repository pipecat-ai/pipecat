#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Event models and data structures for Grok Voice Agent API communication.

Based on xAI's Grok Voice Agent API documentation:
https://docs.x.ai/docs/guides/voice/agent
"""

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipecat.adapters.schemas.direct_function import DirectFunction
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext

#
# Audio format configuration
#

# Grok supports configurable sample rates for PCM audio
SUPPORTED_SAMPLE_RATES = Literal[8000, 16000, 21050, 24000, 32000, 44100, 48000]


class AudioFormat(BaseModel):
    """Base class for audio format configuration."""

    type: str


class PCMAudioFormat(AudioFormat):
    """PCM audio format configuration with configurable sample rate.

    Grok supports: 8000, 16000, 21050, 24000, 32000, 44100, 48000 Hz

    Parameters:
        type: Audio format type, always "audio/pcm".
        rate: Sample rate in Hz. Defaults to 24000.
    """

    type: Literal["audio/pcm"] = "audio/pcm"
    rate: SUPPORTED_SAMPLE_RATES = 24000


class PCMUAudioFormat(AudioFormat):
    """PCMU (G.711 μ-law) audio format configuration.

    Fixed at 8000 Hz sample rate.

    Parameters:
        type: Audio format type, always "audio/pcmu".
    """

    type: Literal["audio/pcmu"] = "audio/pcmu"


class PCMAAudioFormat(AudioFormat):
    """PCMA (G.711 A-law) audio format configuration.

    Fixed at 8000 Hz sample rate.

    Parameters:
        type: Audio format type, always "audio/pcma".
    """

    type: Literal["audio/pcma"] = "audio/pcma"


#
# Turn detection configuration
#


class TurnDetection(BaseModel):
    """Server-side voice activity detection configuration.

    Parameters:
        type: Detection type, must be "server_vad" or None for manual.
        threshold: VAD activation threshold (0.1–0.9). Higher requires louder audio.
        silence_duration_ms: Silence before the server ends the user turn.
        prefix_padding_ms: Audio (ms) included before detected speech start.
        idle_timeout_ms: When set, the server re-engages after this many ms of
            silence following an assistant response (``timeout_triggered``).
    """

    type: Literal["server_vad"] | None = "server_vad"
    threshold: float | None = None
    silence_duration_ms: int | None = None
    prefix_padding_ms: int | None = None
    idle_timeout_ms: int | None = None


#
# Audio configuration
#


class InputAudioTranscription(BaseModel):
    """Input audio transcription settings for the voice agent.

    Parameters:
        model: Transcription model. Use ``grok-transcribe`` for streaming
            ``conversation.item.input_audio_transcription.updated`` events.
        language_hint: BCP-47 language code to bias ASR.
        keyterms: Domain terms to bias transcription (max 100, ≤50 chars each).
    """

    model: str | None = None
    language_hint: str | None = None
    keyterms: list[str] | None = None


class AudioInput(BaseModel):
    """Audio input configuration.

    Parameters:
        format: The format configuration for input audio.
        transcription: Optional input transcription settings.
        transport: Wire path for input audio (``json`` or ``binary``).
    """

    format: PCMAudioFormat | PCMUAudioFormat | PCMAAudioFormat | None = None
    transcription: InputAudioTranscription | None = None
    transport: Literal["json", "binary"] | None = None


class AudioOutput(BaseModel):
    """Audio output configuration.

    Parameters:
        format: The format configuration for output audio.
        speed: Playback speed multiplier (0.7–1.5).
        transport: Wire path for assistant audio (``json`` or ``binary``).
    """

    format: PCMAudioFormat | PCMUAudioFormat | PCMAAudioFormat | None = None
    speed: float | None = None
    transport: Literal["json", "binary"] | None = None


class AudioConfiguration(BaseModel):
    """Audio configuration for input and output.

    Parameters:
        input: Configuration for input audio.
        output: Configuration for output audio.
    """

    input: AudioInput | None = None
    output: AudioOutput | None = None


class Reasoning(BaseModel):
    """Reasoning controls for voice-agent models.

    Parameters:
        effort: ``high`` enables reasoning; ``none`` disables it.
    """

    effort: Literal["high", "none"] | None = None


class SessionResumption(BaseModel):
    """Session resumption opt-in for reconnecting with conversation history.

    Parameters:
        enabled: When true, the server caches turns for ``conversation_id`` replay.
    """

    enabled: bool | None = None


#
# Tool definitions - Grok-specific tools
#


class WebSearchTool(BaseModel):
    """Web search tool configuration.

    Enables the voice agent to search the web for current information.
    """

    type: Literal["web_search"] = "web_search"


class XSearchTool(BaseModel):
    """X (Twitter) search tool configuration.

    Enables the voice agent to search X for posts and information.

    Parameters:
        type: Tool type, always "x_search".
        allowed_x_handles: Optional list of X handles to filter search results.
    """

    type: Literal["x_search"] = "x_search"
    allowed_x_handles: list[str] | None = None


class FileSearchTool(BaseModel):
    """File/Collection search tool configuration.

    Enables the voice agent to search through uploaded document collections.

    Parameters:
        type: Tool type, always "file_search".
        vector_store_ids: List of collection IDs to search.
        max_num_results: Maximum number of results to return.
    """

    type: Literal["file_search"] = "file_search"
    vector_store_ids: list[str]
    max_num_results: int | None = 10


class FunctionTool(BaseModel):
    """Custom function tool configuration.

    Parameters:
        type: Tool type, always "function".
        name: Name of the function.
        description: Description of what the function does.
        parameters: JSON schema for function parameters.
    """

    type: Literal["function"] = "function"
    name: str
    description: str
    parameters: dict[str, Any]


class McpTool(BaseModel):
    """Remote MCP tool configuration managed by xAI.

    Parameters:
        type: Tool type, always "mcp".
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["mcp"] = "mcp"


# Union type for all Grok tools
GrokTool = WebSearchTool | XSearchTool | FileSearchTool | FunctionTool | McpTool | dict[str, Any]


#
# Voice options
#

# Voice IDs are plain strings: any built-in voice (xAI documents the catalogue at
# https://docs.x.ai/docs/guides/voice/agent, and ``GET /v1/tts/voices`` returns it)
# or a custom ID from the Custom Voices API. IDs are case-insensitive.
GrokVoice = str


#
# Session properties
#


class SessionProperties(BaseModel):
    """Configuration properties for a Grok Voice Agent session.

    Parameters:
        instructions: System instructions for the assistant.
        voice: The voice the model uses to respond — a built-in voice ID (see
            `xAI's docs <https://docs.x.ai/docs/guides/voice/agent>`_) or a
            custom one from the Custom Voices API. Defaults to "eve", which is
            xAI's own default.
        turn_detection: Configuration for turn detection. Defaults to server-side VAD.
            Set to None for manual turn detection.
        audio: Configuration for input and output audio.
        tools: Available tools for the assistant (web_search, x_search, file_search,
            function, mcp).
        reasoning: Optional reasoning effort controls.
        resumption: Optional session-resumption opt-in.
        replace: Optional pronunciation replacement map applied before TTS.
        id: Session id when present on server snapshots (``session.created``).
        object: Object type from server snapshots.
        model: Model id echoed on server snapshots.
    """

    # Needed to support ToolSchema in tools field. Ignore unknown server fields
    # so session.created / session.updated snapshots remain parseable.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    instructions: str | None = None
    voice: str | None = "eve"
    turn_detection: TurnDetection | None = Field(
        default_factory=lambda: TurnDetection(type="server_vad")
    )
    audio: AudioConfiguration | None = None
    # Tools provided by the user may be a ToolsSchema or a plain list of standard
    # tools (the validator below normalizes that to a ToolsSchema); a list of
    # provider-native GrokTool objects passes through.
    tools: ToolsSchema | list[FunctionSchema | DirectFunction] | list[GrokTool] | None = None
    reasoning: Reasoning | None = None
    resumption: SessionResumption | None = None
    replace: dict[str, str] | None = None
    id: str | None = None
    object: str | None = None
    model: str | None = None

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_tools(cls, v):
        """Wrap a plain list of standard tools in a ``ToolsSchema``.

        Provider-native tool lists pass through unchanged.
        """
        if isinstance(v, list):
            normalized = LLMContext._normalize_and_validate_tools(v, allow_provider_tools=True)
            return normalized if isinstance(normalized, (ToolsSchema, list)) else None
        return v


#
# Conversation items
#


class ItemContent(BaseModel):
    """Content within a conversation item.

    Parameters:
        type: Content type (input_text, input_audio, text, audio).
        text: Text content for text-based items.
        audio: Base64-encoded audio data for audio items.
        transcript: Transcribed text for audio items.
    """

    type: Literal["text", "audio", "input_text", "input_audio", "output_text", "output_audio"]
    text: str | None = None
    audio: str | None = None  # base64-encoded audio
    transcript: str | None = None


class ConversationItem(BaseModel):
    """A conversation item in the realtime session.

    Parameters:
        id: Unique identifier for the item, auto-generated if not provided.
        object: Object type identifier for the realtime API.
        type: Item type (message, function_call, function_call_output, or force_message).
        status: Current status of the item.
        role: Speaker role for message items (user, assistant, or system).
        content: Content list for message items.
        call_id: Function call identifier for function_call items.
        name: Function name for function_call items.
        arguments: Function arguments as JSON string for function_call items.
        output: Function output as JSON string for function_call_output items.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex))
    object: Literal["realtime.item"] | None = None
    type: Literal["message", "function_call", "function_call_output", "force_message"]
    status: Literal["completed", "in_progress", "incomplete"] | None = None
    role: Literal["user", "assistant", "system", "tool"] | None = None
    content: list[ItemContent] | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    output: str | None = None


class RealtimeConversation(BaseModel):
    """A realtime conversation session.

    Parameters:
        id: Unique identifier for the conversation.
        object: Object type identifier, always "realtime.conversation".
    """

    id: str
    object: Literal["realtime.conversation"]


class ResponseProperties(BaseModel):
    """Properties for configuring assistant responses.

    Parameters:
        modalities: Output modalities for the response (text, audio, or both).
        instructions: Per-response system prompt override (session instructions
            resume on the next response).
    """

    modalities: list[Literal["text", "audio"]] | None = ["text", "audio"]
    instructions: str | None = None


#
# Error class
#


class RealtimeError(BaseModel):
    """Error information from the realtime API.

    Parameters:
        type: Error type identifier.
        code: Specific error code.
        message: Human-readable error message.
        param: Parameter name that caused the error, if applicable.
        event_id: Event ID associated with the error, if applicable.
    """

    type: str | None = None
    code: str | None = ""
    message: str
    param: str | None = None
    event_id: str | None = None


#
# Client Events (sent to Grok)
#


class ClientEvent(BaseModel):
    """Base class for client events sent to the realtime API.

    Parameters:
        event_id: Unique identifier for the event, auto-generated if not provided.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SessionUpdateEvent(ClientEvent):
    """Event to update session properties.

    Parameters:
        type: Event type, always "session.update".
        session: Updated session properties.
    """

    type: Literal["session.update"] = "session.update"
    session: SessionProperties


class InputAudioBufferAppendEvent(ClientEvent):
    """Event to append audio data to the input buffer.

    Parameters:
        type: Event type, always "input_audio_buffer.append".
        audio: Base64-encoded audio data to append.
    """

    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str  # base64-encoded audio


class InputAudioBufferCommitEvent(ClientEvent):
    """Event to commit the current input audio buffer.

    Used when turn_detection is null (manual mode).

    Parameters:
        type: Event type, always "input_audio_buffer.commit".
    """

    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"


class InputAudioBufferClearEvent(ClientEvent):
    """Event to clear the input audio buffer.

    Parameters:
        type: Event type, always "input_audio_buffer.clear".
    """

    type: Literal["input_audio_buffer.clear"] = "input_audio_buffer.clear"


class ConversationItemCreateEvent(ClientEvent):
    """Event to create a new conversation item.

    Parameters:
        type: Event type, always "conversation.item.create".
        previous_item_id: ID of the item to insert after, if any.
        item: The conversation item to create.
    """

    type: Literal["conversation.item.create"] = "conversation.item.create"
    previous_item_id: str | None = None
    item: ConversationItem


class ResponseCreateEvent(ClientEvent):
    """Event to create a new assistant response.

    Parameters:
        type: Event type, always "response.create".
        response: Optional response configuration properties.
    """

    type: Literal["response.create"] = "response.create"
    response: ResponseProperties | None = None


class ResponseCancelEvent(ClientEvent):
    """Event to cancel the current assistant response.

    Parameters:
        type: Event type, always "response.cancel".
    """

    type: Literal["response.cancel"] = "response.cancel"


class ConversationItemTruncateEvent(ClientEvent):
    """Event to truncate a previous assistant audio item.

    Parameters:
        type: Event type, always "conversation.item.truncate".
        item_id: ID of the item to truncate.
        content_index: Index of the content to truncate within the item.
        audio_end_ms: End time in milliseconds for the truncated audio.
    """

    type: Literal["conversation.item.truncate"] = "conversation.item.truncate"
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemDeleteEvent(ClientEvent):
    """Event to delete a conversation item by ID.

    Parameters:
        type: Event type, always "conversation.item.delete".
        item_id: ID of the item to delete.
    """

    type: Literal["conversation.item.delete"] = "conversation.item.delete"
    item_id: str


#
# Server Events (received from Grok)
#


class ServerEvent(BaseModel):
    """Base class for server events received from the realtime API.

    Parameters:
        event_id: Unique identifier for the event.
        type: Type of the server event.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_id: str
    type: str


class SessionCreatedEvent(ServerEvent):
    """Event indicating a session has been created on connect.

    Parameters:
        type: Event type, always "session.created".
        session: The created session properties.
    """

    type: Literal["session.created"]
    session: SessionProperties


class SessionUpdatedEvent(ServerEvent):
    """Event indicating a session has been updated.

    Parameters:
        type: Event type, always "session.updated".
        session: The updated session properties.
    """

    type: Literal["session.updated"]
    session: SessionProperties


class ConversationCreated(ServerEvent):
    """Event indicating a conversation has been created.

    Sent after connect; Pipecat uses this to send the initial session update.

    Parameters:
        type: Event type, always "conversation.created".
        conversation: The created conversation.
    """

    type: Literal["conversation.created"]
    conversation: RealtimeConversation


class ConversationItemAdded(ServerEvent):
    """Event indicating a conversation item has been added.

    Parameters:
        type: Event type, always "conversation.item.added".
        previous_item_id: ID of the previous item, if any.
        item: The added conversation item.
    """

    type: Literal["conversation.item.added"]
    previous_item_id: str | None = None
    item: ConversationItem


class ConversationItemDeleted(ServerEvent):
    """Event confirming a conversation item was deleted.

    Parameters:
        type: Event type, always "conversation.item.deleted".
        item_id: ID of the deleted conversation item.
    """

    type: Literal["conversation.item.deleted"]
    item_id: str


class ConversationItemTruncated(ServerEvent):
    """Event confirming a conversation item was truncated.

    Parameters:
        type: Event type, always "conversation.item.truncated".
        item_id: ID of the truncated conversation item.
        content_index: Index of the content within the item.
        audio_end_ms: End time in milliseconds for the truncated audio.
    """

    type: Literal["conversation.item.truncated"]
    item_id: str
    content_index: int | None = None
    audio_end_ms: int | None = None


class ConversationItemInputAudioTranscriptionUpdated(ServerEvent):
    """Cumulative streaming update for user input audio transcription.

    Emitted when ``audio.input.transcription.model`` is ``grok-transcribe``.
    Unlike a delta, ``transcript`` is the full cumulative text so far and may
    correct earlier updates.

    Parameters:
        type: Event type, always "conversation.item.input_audio_transcription.updated".
        item_id: ID of the conversation item being transcribed.
        content_index: Index of the content within the item, if provided.
        transcript: Cumulative transcription text so far.
    """

    type: Literal["conversation.item.input_audio_transcription.updated"]
    item_id: str
    content_index: int | None = None
    transcript: str


class ConversationItemInputAudioTranscriptionCompleted(ServerEvent):
    """Event indicating input audio transcription is complete.

    Parameters:
        type: Event type, always "conversation.item.input_audio_transcription.completed".
        item_id: ID of the conversation item that was transcribed.
        content_index: Index of the content within the item, if provided.
        transcript: Complete transcription text.
    """

    type: Literal["conversation.item.input_audio_transcription.completed"]
    item_id: str
    content_index: int | None = None
    transcript: str


class InputAudioBufferSpeechStarted(ServerEvent):
    """Event indicating speech has started in the input audio buffer.

    Only sent when turn_detection is "server_vad".

    Parameters:
        type: Event type, always "input_audio_buffer.speech_started".
        item_id: ID of the associated conversation item.
    """

    type: Literal["input_audio_buffer.speech_started"]
    item_id: str


class InputAudioBufferSpeechStopped(ServerEvent):
    """Event indicating speech has stopped in the input audio buffer.

    Only sent when turn_detection is "server_vad".

    Parameters:
        type: Event type, always "input_audio_buffer.speech_stopped".
        item_id: ID of the associated conversation item.
    """

    type: Literal["input_audio_buffer.speech_stopped"]
    item_id: str


class InputAudioBufferCommitted(ServerEvent):
    """Event indicating the input audio buffer has been committed.

    Parameters:
        type: Event type, always "input_audio_buffer.committed".
        previous_item_id: ID of the previous item, if any.
        item_id: ID of the committed conversation item.
    """

    type: Literal["input_audio_buffer.committed"]
    previous_item_id: str | None = None
    item_id: str


class InputAudioBufferCleared(ServerEvent):
    """Event indicating the input audio buffer has been cleared.

    Parameters:
        type: Event type, always "input_audio_buffer.cleared".
    """

    type: Literal["input_audio_buffer.cleared"]


class InputAudioBufferTimeoutTriggered(ServerEvent):
    """Event indicating the idle timeout fired with no user speech.

    When ``turn_detection.idle_timeout_ms`` is set, the server commits a silent
    user turn and generates a proactive check-in.

    Parameters:
        type: Event type, always "input_audio_buffer.timeout_triggered".
        item_id: ID of the associated conversation item, if provided.
    """

    type: Literal["input_audio_buffer.timeout_triggered"]
    item_id: str | None = None


class InputAudioBufferDtmfEventReceived(ServerEvent):
    """DTMF tone detected on a SIP session.

    Parameters:
        type: Event type, always "input_audio_buffer.dtmf_event_received".
        digit: The DTMF digit received.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: Literal["input_audio_buffer.dtmf_event_received"]
    digit: str | None = None


class ResponseOutputItemAdded(ServerEvent):
    """Event indicating an output item has been added to a response.

    Parameters:
        type: Event type, always "response.output_item.added".
        response_id: ID of the response.
        output_index: Index of the output item.
        item: The added conversation item.
    """

    type: Literal["response.output_item.added"]
    response_id: str
    output_index: int
    item: ConversationItem


class ResponseAudioTranscriptDelta(ServerEvent):
    """Event containing incremental audio transcript from a response.

    Parameters:
        type: Event type, always "response.output_audio_transcript.delta".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        delta: Incremental transcript text.
    """

    type: Literal["response.output_audio_transcript.delta"]
    response_id: str
    item_id: str
    delta: str


class ResponseAudioTranscriptDone(ServerEvent):
    """Event indicating audio transcript is complete.

    Parameters:
        type: Event type, always "response.output_audio_transcript.done".
        response_id: ID of the response.
        item_id: ID of the conversation item.
    """

    type: Literal["response.output_audio_transcript.done"]
    response_id: str
    item_id: str


class ResponseAudioDelta(ServerEvent):
    """Event containing incremental audio data from a response.

    Parameters:
        type: Event type, always "response.output_audio.delta".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        content_index: Index of the content part.
        delta: Base64-encoded incremental audio data.
    """

    type: Literal["response.output_audio.delta"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str  # base64-encoded audio


class ResponseAudioDone(ServerEvent):
    """Event indicating audio content is complete.

    Parameters:
        type: Event type, always "response.output_audio.done".
        response_id: ID of the response.
        item_id: ID of the conversation item.
    """

    type: Literal["response.output_audio.done"]
    response_id: str
    item_id: str


class ResponseFunctionCallArgumentsDelta(ServerEvent):
    """Event containing incremental function call arguments.

    Parameters:
        type: Event type, always "response.function_call_arguments.delta".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        call_id: ID of the function call.
        delta: Incremental function arguments as JSON.
        previous_item_id: ID of the previous item, if any.
    """

    type: Literal["response.function_call_arguments.delta"]
    response_id: str | None = None
    item_id: str | None = None
    call_id: str
    delta: str
    previous_item_id: str | None = None


class ResponseFunctionCallArgumentsDone(ServerEvent):
    """Event indicating function call arguments are complete.

    Parameters:
        type: Event type, always "response.function_call_arguments.done".
        call_id: ID of the function call.
        name: Name of the function being called.
        arguments: Complete function arguments as JSON string.
    """

    type: Literal["response.function_call_arguments.done"]
    call_id: str
    name: str
    arguments: str


class Usage(BaseModel):
    """Token usage statistics for a response.

    All fields are optional because Grok sends empty usage in some events.

    Parameters:
        total_tokens: Total number of tokens used.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens used.
    """

    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class Response(BaseModel):
    """A complete assistant response.

    Parameters:
        id: Unique identifier for the response.
        object: Object type, always "realtime.response".
        status: Current status of the response.
        output: List of conversation items in the response.
        usage: Token usage statistics for the response.
    """

    id: str
    object: Literal["realtime.response"]
    status: Literal["completed", "in_progress", "incomplete", "cancelled", "failed"]
    status_details: Any | None = None
    output: list[ConversationItem]
    usage: Usage | None = None


class ResponseCreated(ServerEvent):
    """Event indicating an assistant response has been created.

    Parameters:
        type: Event type, always "response.created".
        response: The created response object.
    """

    type: Literal["response.created"]
    response: Response


class ResponseDone(ServerEvent):
    """Event indicating an assistant response is complete.

    Parameters:
        type: Event type, always "response.done".
        response: The completed response object.
        usage: Token usage (also available at top level in Grok).
    """

    type: Literal["response.done"]
    response: Response
    usage: Usage | None = None


class ResponseOutputItemDone(ServerEvent):
    """Event indicating an output item is complete.

    Parameters:
        type: Event type, always "response.output_item.done".
        response_id: ID of the response.
        output_index: Index of the output item.
        item: The completed conversation item.
    """

    type: Literal["response.output_item.done"]
    response_id: str
    output_index: int
    item: ConversationItem


class ContentPart(BaseModel):
    """A content part within a response.

    Parameters:
        type: Type of the content part (audio, text).
        transcript: Transcript text if applicable.
    """

    type: str
    transcript: str | None = None


class ResponseContentPartAdded(ServerEvent):
    """Event indicating a content part has been added to a response.

    Parameters:
        type: Event type, always "response.content_part.added".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        content_index: Index of the content part.
        output_index: Index of the output item.
        part: The added content part.
    """

    type: Literal["response.content_part.added"]
    response_id: str
    item_id: str
    content_index: int
    output_index: int
    part: ContentPart


class ResponseContentPartDone(ServerEvent):
    """Event indicating a content part is complete.

    Parameters:
        type: Event type, always "response.content_part.done".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        content_index: Index of the content part.
        output_index: Index of the output item.
    """

    type: Literal["response.content_part.done"]
    response_id: str
    item_id: str
    content_index: int
    output_index: int


class ResponseTextDelta(ServerEvent):
    """Event containing incremental text-mode output from a response.

    xAI emits both ``response.text.delta`` and ``response.output_text.delta``
    with the same payload; clients should accept either name.

    Parameters:
        type: ``response.output_text.delta`` or ``response.text.delta``.
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        content_index: Index of the content part.
        delta: Incremental text content.
    """

    type: Literal["response.output_text.delta", "response.text.delta"]
    response_id: str | None = None
    item_id: str | None = None
    output_index: int | None = None
    content_index: int | None = None
    delta: str


class McpListToolsEvent(ServerEvent):
    """MCP tool discovery lifecycle event.

    Parameters:
        type: One of ``mcp_list_tools.in_progress``, ``.completed``, or ``.failed``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: Literal[
        "mcp_list_tools.in_progress",
        "mcp_list_tools.completed",
        "mcp_list_tools.failed",
    ]


class ResponseMcpCallEvent(ServerEvent):
    """MCP tool call lifecycle or argument streaming event.

    Parameters:
        type: An MCP call event name under ``response.mcp_call*``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: Literal[
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.mcp_call.in_progress",
        "response.mcp_call.completed",
        "response.mcp_call.failed",
    ]


class PingEvent(ServerEvent):
    """Keep-alive ping event from the server.

    Parameters:
        type: Event type, always "ping".
        timestamp: Server timestamp in milliseconds.
    """

    type: Literal["ping"]
    timestamp: int


class ErrorEvent(ServerEvent):
    """Event indicating an error occurred.

    Parameters:
        type: Event type, always "error".
        error: Error details.
    """

    type: Literal["error"]
    error: RealtimeError


#
# Event parsing
#

_server_event_types = {
    "error": ErrorEvent,
    "ping": PingEvent,
    "session.created": SessionCreatedEvent,
    "session.updated": SessionUpdatedEvent,
    "conversation.created": ConversationCreated,
    "conversation.item.added": ConversationItemAdded,
    "conversation.item.deleted": ConversationItemDeleted,
    "conversation.item.truncated": ConversationItemTruncated,
    "conversation.item.input_audio_transcription.updated": (
        ConversationItemInputAudioTranscriptionUpdated
    ),
    "conversation.item.input_audio_transcription.completed": (
        ConversationItemInputAudioTranscriptionCompleted
    ),
    "input_audio_buffer.speech_started": InputAudioBufferSpeechStarted,
    "input_audio_buffer.speech_stopped": InputAudioBufferSpeechStopped,
    "input_audio_buffer.committed": InputAudioBufferCommitted,
    "input_audio_buffer.cleared": InputAudioBufferCleared,
    "input_audio_buffer.timeout_triggered": InputAudioBufferTimeoutTriggered,
    "input_audio_buffer.dtmf_event_received": InputAudioBufferDtmfEventReceived,
    "response.created": ResponseCreated,
    "response.output_item.added": ResponseOutputItemAdded,
    "response.output_item.done": ResponseOutputItemDone,
    "response.content_part.added": ResponseContentPartAdded,
    "response.content_part.done": ResponseContentPartDone,
    "response.output_audio_transcript.delta": ResponseAudioTranscriptDelta,
    "response.output_audio_transcript.done": ResponseAudioTranscriptDone,
    "response.output_audio.delta": ResponseAudioDelta,
    "response.output_audio.done": ResponseAudioDone,
    "response.output_text.delta": ResponseTextDelta,
    "response.text.delta": ResponseTextDelta,
    "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDelta,
    "response.function_call_arguments.done": ResponseFunctionCallArgumentsDone,
    "mcp_list_tools.in_progress": McpListToolsEvent,
    "mcp_list_tools.completed": McpListToolsEvent,
    "mcp_list_tools.failed": McpListToolsEvent,
    "response.mcp_call_arguments.delta": ResponseMcpCallEvent,
    "response.mcp_call_arguments.done": ResponseMcpCallEvent,
    "response.mcp_call.in_progress": ResponseMcpCallEvent,
    "response.mcp_call.completed": ResponseMcpCallEvent,
    "response.mcp_call.failed": ResponseMcpCallEvent,
    "response.done": ResponseDone,
}


def parse_server_event(data: str | bytes):
    """Parse a server event from JSON.

    Args:
        data: JSON text containing the server event, as delivered by the
            websocket.

    Returns:
        Parsed server event object of the appropriate type.

    Raises:
        Exception: If the event type is unimplemented or parsing fails.
    """
    try:
        event = json.loads(data)
        event_type = event["type"]
        if event_type not in _server_event_types:
            raise Exception(f"Unimplemented server event type: {event_type}")
        return _server_event_types[event_type].model_validate(event)
    except Exception as e:
        raise Exception(f"{e} \n\n{data}")
