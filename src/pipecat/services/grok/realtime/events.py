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
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from pipecat.adapters.schemas.tools_schema import ToolsSchema

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
    """

    type: Optional[Literal["server_vad"]] = "server_vad"


#
# Audio configuration
#


class AudioInput(BaseModel):
    """Audio input configuration.

    Parameters:
        format: The format configuration for input audio.
    """

    format: Optional[Union[PCMAudioFormat, PCMUAudioFormat, PCMAAudioFormat]] = None


class AudioOutput(BaseModel):
    """Audio output configuration.

    Parameters:
        format: The format configuration for output audio.
    """

    format: Optional[Union[PCMAudioFormat, PCMUAudioFormat, PCMAAudioFormat]] = None


class AudioConfiguration(BaseModel):
    """Audio configuration for input and output.

    Parameters:
        input: Configuration for input audio.
        output: Configuration for output audio.
    """

    input: Optional[AudioInput] = None
    output: Optional[AudioOutput] = None


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
    allowed_x_handles: Optional[List[str]] = None


class FileSearchTool(BaseModel):
    """File/Collection search tool configuration.

    Enables the voice agent to search through uploaded document collections.

    Parameters:
        type: Tool type, always "file_search".
        vector_store_ids: List of collection IDs to search.
        max_num_results: Maximum number of results to return.
    """

    type: Literal["file_search"] = "file_search"
    vector_store_ids: List[str]
    max_num_results: Optional[int] = 10


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
    parameters: Dict[str, Any]


# Union type for all Grok tools
GrokTool = Union[WebSearchTool, XSearchTool, FileSearchTool, FunctionTool, Dict[str, Any]]


#
# Voice options
#

# Grok voice options: Ara (default), Rex, Sal, Eve, Leo
GrokVoice = Literal["Ara", "Rex", "Sal", "Eve", "Leo"]


#
# Session properties
#


class SessionProperties(BaseModel):
    """Configuration properties for a Grok Voice Agent session.

    Parameters:
        instructions: System instructions for the assistant.
        voice: The voice the model uses to respond. Options: Ara, Rex, Sal, Eve, Leo.
            Defaults to "Ara".
        turn_detection: Configuration for turn detection. Defaults to server-side VAD.
            Set to None for manual turn detection.
        audio: Configuration for input and output audio.
        tools: Available tools for the assistant (web_search, x_search, file_search, function).
    """

    # Needed to support ToolSchema in tools field.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    instructions: Optional[str] = None
    voice: Optional[GrokVoice] = "Ara"
    turn_detection: Optional[TurnDetection] = Field(
        default_factory=lambda: TurnDetection(type="server_vad")
    )
    audio: Optional[AudioConfiguration] = None
    # Tools can be ToolsSchema when provided by user, or list of dicts for API
    tools: Optional[ToolsSchema | List[GrokTool]] = None


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
    text: Optional[str] = None
    audio: Optional[str] = None  # base64-encoded audio
    transcript: Optional[str] = None


class ConversationItem(BaseModel):
    """A conversation item in the realtime session.

    Parameters:
        id: Unique identifier for the item, auto-generated if not provided.
        object: Object type identifier for the realtime API.
        type: Item type (message, function_call, or function_call_output).
        status: Current status of the item.
        role: Speaker role for message items (user, assistant, or system).
        content: Content list for message items.
        call_id: Function call identifier for function_call items.
        name: Function name for function_call items.
        arguments: Function arguments as JSON string for function_call items.
        output: Function output as JSON string for function_call_output items.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex))
    object: Optional[Literal["realtime.item"]] = None
    type: Literal["message", "function_call", "function_call_output"]
    status: Optional[Literal["completed", "in_progress", "incomplete"]] = None
    role: Optional[Literal["user", "assistant", "system", "tool"]] = None
    content: Optional[List[ItemContent]] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None


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
    """

    modalities: Optional[List[Literal["text", "audio"]]] = ["text", "audio"]


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

    type: Optional[str] = None
    code: Optional[str] = ""
    message: str
    param: Optional[str] = None
    event_id: Optional[str] = None


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
    previous_item_id: Optional[str] = None
    item: ConversationItem


class ResponseCreateEvent(ClientEvent):
    """Event to create a new assistant response.

    Parameters:
        type: Event type, always "response.create".
        response: Optional response configuration properties.
    """

    type: Literal["response.create"] = "response.create"
    response: Optional[ResponseProperties] = None


class ResponseCancelEvent(ClientEvent):
    """Event to cancel the current assistant response.

    Parameters:
        type: Event type, always "response.cancel".
    """

    type: Literal["response.cancel"] = "response.cancel"


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

    This is the first message received after connecting.

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
    previous_item_id: Optional[str] = None
    item: ConversationItem


class ConversationItemInputAudioTranscriptionCompleted(ServerEvent):
    """Event indicating input audio transcription is complete.

    Parameters:
        type: Event type, always "conversation.item.input_audio_transcription.completed".
        item_id: ID of the conversation item that was transcribed.
        transcript: Complete transcription text.
    """

    type: Literal["conversation.item.input_audio_transcription.completed"]
    item_id: str
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
    previous_item_id: Optional[str] = None
    item_id: str


class InputAudioBufferCleared(ServerEvent):
    """Event indicating the input audio buffer has been cleared.

    Parameters:
        type: Event type, always "input_audio_buffer.cleared".
    """

    type: Literal["input_audio_buffer.cleared"]


class ResponseCreated(ServerEvent):
    """Event indicating an assistant response has been created.

    Parameters:
        type: Event type, always "response.created".
        response: The created response object.
    """

    type: Literal["response.created"]
    response: "Response"


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
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    call_id: str
    delta: str
    previous_item_id: Optional[str] = None


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

    total_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


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
    status_details: Optional[Any] = None
    output: List[ConversationItem]
    usage: Optional[Usage] = None


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
    usage: Optional[Usage] = None


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
    transcript: Optional[str] = None


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
    "session.updated": SessionUpdatedEvent,
    "conversation.created": ConversationCreated,
    "conversation.item.added": ConversationItemAdded,
    "conversation.item.input_audio_transcription.completed": ConversationItemInputAudioTranscriptionCompleted,
    "input_audio_buffer.speech_started": InputAudioBufferSpeechStarted,
    "input_audio_buffer.speech_stopped": InputAudioBufferSpeechStopped,
    "input_audio_buffer.committed": InputAudioBufferCommitted,
    "input_audio_buffer.cleared": InputAudioBufferCleared,
    "response.created": ResponseCreated,
    "response.output_item.added": ResponseOutputItemAdded,
    "response.output_item.done": ResponseOutputItemDone,
    "response.content_part.added": ResponseContentPartAdded,
    "response.content_part.done": ResponseContentPartDone,
    "response.output_audio_transcript.delta": ResponseAudioTranscriptDelta,
    "response.output_audio_transcript.done": ResponseAudioTranscriptDone,
    "response.output_audio.delta": ResponseAudioDelta,
    "response.output_audio.done": ResponseAudioDone,
    "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDelta,
    "response.function_call_arguments.done": ResponseFunctionCallArgumentsDone,
    "response.done": ResponseDone,
}


def parse_server_event(data: str):
    """Parse a server event from JSON string.

    Args:
        data: JSON string containing the server event.

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
