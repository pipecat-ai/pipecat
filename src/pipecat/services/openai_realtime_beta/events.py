#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Event models and data structures for OpenAI Realtime API communication."""

import json
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

#
# session properties
#


class InputAudioTranscription(BaseModel):
    """Configuration for audio transcription settings."""

    model: str = "gpt-4o-transcribe"
    language: Optional[str]
    prompt: Optional[str]

    def __init__(
        self,
        model: Optional[str] = "gpt-4o-transcribe",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """Initialize InputAudioTranscription.

        Args:
            model: Transcription model to use (e.g., "gpt-4o-transcribe", "whisper-1").
            language: Optional language code for transcription.
            prompt: Optional transcription hint text.
        """
        super().__init__(model=model, language=language, prompt=prompt)


class TurnDetection(BaseModel):
    """Server-side voice activity detection configuration.

    Parameters:
        type: Detection type, must be "server_vad".
        threshold: Voice activity detection threshold (0.0-1.0). Defaults to 0.5.
        prefix_padding_ms: Padding before speech starts in milliseconds. Defaults to 300.
        silence_duration_ms: Silence duration to detect speech end in milliseconds. Defaults to 800.
    """

    type: Optional[Literal["server_vad"]] = "server_vad"
    threshold: Optional[float] = 0.5
    prefix_padding_ms: Optional[int] = 300
    silence_duration_ms: Optional[int] = 800


class SemanticTurnDetection(BaseModel):
    """Semantic-based turn detection configuration.

    Parameters:
        type: Detection type, must be "semantic_vad".
        eagerness: Turn detection eagerness level. Can be "low", "medium", "high", or "auto".
        create_response: Whether to automatically create responses on turn detection.
        interrupt_response: Whether to interrupt ongoing responses on turn detection.
    """

    type: Optional[Literal["semantic_vad"]] = "semantic_vad"
    eagerness: Optional[Literal["low", "medium", "high", "auto"]] = None
    create_response: Optional[bool] = None
    interrupt_response: Optional[bool] = None


class InputAudioNoiseReduction(BaseModel):
    """Input audio noise reduction configuration.

    Parameters:
        type: Noise reduction type for different microphone scenarios.
    """

    type: Optional[Literal["near_field", "far_field"]]


class SessionProperties(BaseModel):
    """Configuration properties for an OpenAI Realtime session.

    Parameters:
        modalities: Communication modalities to enable (text, audio, or both).
        instructions: System instructions for the assistant.
        voice: Voice ID for text-to-speech output.
        input_audio_format: Format for input audio data.
        output_audio_format: Format for output audio data.
        input_audio_transcription: Configuration for input audio transcription.
        input_audio_noise_reduction: Configuration for input audio noise reduction.
        turn_detection: Turn detection configuration or False to disable.
        tools: Available function tools for the assistant.
        tool_choice: Tool usage strategy ("auto", "none", or "required").
        temperature: Sampling temperature for response generation.
        max_response_output_tokens: Maximum tokens in response or "inf" for unlimited.
    """

    modalities: Optional[List[Literal["text", "audio"]]] = None
    instructions: Optional[str] = None
    voice: Optional[str] = None
    input_audio_format: Optional[Literal["pcm16", "g711_ulaw", "g711_alaw"]] = None
    output_audio_format: Optional[Literal["pcm16", "g711_ulaw", "g711_alaw"]] = None
    input_audio_transcription: Optional[InputAudioTranscription] = None
    input_audio_noise_reduction: Optional[InputAudioNoiseReduction] = None
    # set turn_detection to False to disable turn detection
    turn_detection: Optional[Union[TurnDetection, SemanticTurnDetection, bool]] = Field(
        default=None
    )
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    temperature: Optional[float] = None
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = None


#
# context
#


class ItemContent(BaseModel):
    """Content within a conversation item.

    Parameters:
        type: Content type (text, audio, input_text, or input_audio).
        text: Text content for text-based items.
        audio: Base64-encoded audio data for audio items.
        transcript: Transcribed text for audio items.
    """

    type: Literal["text", "audio", "input_text", "input_audio"]
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
    # role and content are present for message items
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[List[ItemContent]] = None
    # these four fields are present for function_call items
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
        modalities: Output modalities for the response. Defaults to ["audio", "text"].
        instructions: Specific instructions for this response.
        voice: Voice ID for text-to-speech in this response.
        output_audio_format: Audio format for this response.
        tools: Available tools for this response.
        tool_choice: Tool usage strategy for this response.
        temperature: Sampling temperature for this response.
        max_response_output_tokens: Maximum tokens for this response.
    """

    modalities: Optional[List[Literal["text", "audio"]]] = ["audio", "text"]
    instructions: Optional[str] = None
    voice: Optional[str] = None
    output_audio_format: Optional[Literal["pcm16", "g711_ulaw", "g711_alaw"]] = None
    tools: Optional[List[Dict]] = Field(default_factory=list)
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    temperature: Optional[float] = None
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = None


#
# error class
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

    type: str
    code: Optional[str] = ""
    message: str
    param: Optional[str] = None
    event_id: Optional[str] = None


#
# client events
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

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Serialize the event to a dictionary.

        Handles special serialization for turn_detection where False becomes null.

        Args:
            *args: Positional arguments passed to parent model_dump.
            **kwargs: Keyword arguments passed to parent model_dump.

        Returns:
            Dictionary representation of the event.
        """
        dump = super().model_dump(*args, **kwargs)

        # Handle turn_detection so that False is serialized as null
        if "turn_detection" in dump["session"]:
            if dump["session"]["turn_detection"] is False:
                dump["session"]["turn_detection"] = None

        return dump


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


class ConversationItemTruncateEvent(ClientEvent):
    """Event to truncate a conversation item's audio content.

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
    """Event to delete a conversation item.

    Parameters:
        type: Event type, always "conversation.item.delete".
        item_id: ID of the item to delete.
    """

    type: Literal["conversation.item.delete"] = "conversation.item.delete"
    item_id: str


class ConversationItemRetrieveEvent(ClientEvent):
    """Event to retrieve a conversation item by ID.

    Parameters:
        type: Event type, always "conversation.item.retrieve".
        item_id: ID of the item to retrieve.
    """

    type: Literal["conversation.item.retrieve"] = "conversation.item.retrieve"
    item_id: str


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
# server events
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
    """Event indicating a session has been created.

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

    Parameters:
        type: Event type, always "conversation.created".
        conversation: The created conversation.
    """

    type: Literal["conversation.created"]
    conversation: RealtimeConversation


class ConversationItemCreated(ServerEvent):
    """Event indicating a conversation item has been created.

    Parameters:
        type: Event type, always "conversation.item.created".
        previous_item_id: ID of the previous item, if any.
        item: The created conversation item.
    """

    type: Literal["conversation.item.created"]
    previous_item_id: Optional[str] = None
    item: ConversationItem


class ConversationItemInputAudioTranscriptionDelta(ServerEvent):
    """Event containing incremental input audio transcription.

    Parameters:
        type: Event type, always "conversation.item.input_audio_transcription.delta".
        item_id: ID of the conversation item being transcribed.
        content_index: Index of the content within the item.
        delta: Incremental transcription text.
    """

    type: Literal["conversation.item.input_audio_transcription.delta"]
    item_id: str
    content_index: int
    delta: str


class ConversationItemInputAudioTranscriptionCompleted(ServerEvent):
    """Event indicating input audio transcription is complete.

    Parameters:
        type: Event type, always "conversation.item.input_audio_transcription.completed".
        item_id: ID of the conversation item that was transcribed.
        content_index: Index of the content within the item.
        transcript: Complete transcription text.
    """

    type: Literal["conversation.item.input_audio_transcription.completed"]
    item_id: str
    content_index: int
    transcript: str


class ConversationItemInputAudioTranscriptionFailed(ServerEvent):
    """Event indicating input audio transcription failed.

    Parameters:
        type: Event type, always "conversation.item.input_audio_transcription.failed".
        item_id: ID of the conversation item that failed transcription.
        content_index: Index of the content within the item.
        error: Error details for the transcription failure.
    """

    type: Literal["conversation.item.input_audio_transcription.failed"]
    item_id: str
    content_index: int
    error: RealtimeError


class ConversationItemTruncated(ServerEvent):
    """Event indicating a conversation item has been truncated.

    Parameters:
        type: Event type, always "conversation.item.truncated".
        item_id: ID of the truncated conversation item.
        content_index: Index of the content within the item.
        audio_end_ms: End time in milliseconds for the truncated audio.
    """

    type: Literal["conversation.item.truncated"]
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemDeleted(ServerEvent):
    """Event indicating a conversation item has been deleted.

    Parameters:
        type: Event type, always "conversation.item.deleted".
        item_id: ID of the deleted conversation item.
    """

    type: Literal["conversation.item.deleted"]
    item_id: str


class ConversationItemRetrieved(ServerEvent):
    """Event containing a retrieved conversation item.

    Parameters:
        type: Event type, always "conversation.item.retrieved".
        item: The retrieved conversation item.
    """

    type: Literal["conversation.item.retrieved"]
    item: ConversationItem


class ResponseCreated(ServerEvent):
    """Event indicating an assistant response has been created.

    Parameters:
        type: Event type, always "response.created".
        response: The created response object.
    """

    type: Literal["response.created"]
    response: "Response"


class ResponseDone(ServerEvent):
    """Event indicating an assistant response is complete.

    Parameters:
        type: Event type, always "response.done".
        response: The completed response object.
    """

    type: Literal["response.done"]
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


class ResponseContentPartAdded(ServerEvent):
    """Event indicating a content part has been added to a response.

    Parameters:
        type: Event type, always "response.content_part.added".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        content_index: Index of the content part.
        part: The added content part.
    """

    type: Literal["response.content_part.added"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: ItemContent


class ResponseContentPartDone(ServerEvent):
    """Event indicating a content part is complete.

    Parameters:
        type: Event type, always "response.content_part.done".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        content_index: Index of the content part.
        part: The completed content part.
    """

    type: Literal["response.content_part.done"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: ItemContent


class ResponseTextDelta(ServerEvent):
    """Event containing incremental text from a response.

    Parameters:
        type: Event type, always "response.text.delta".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        content_index: Index of the content part.
        delta: Incremental text content.
    """

    type: Literal["response.text.delta"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseTextDone(ServerEvent):
    """Event indicating text content is complete.

    Parameters:
        type: Event type, always "response.text.done".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        content_index: Index of the content part.
        text: Complete text content.
    """

    type: Literal["response.text.done"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseAudioTranscriptDelta(ServerEvent):
    """Event containing incremental audio transcript from a response.

    Parameters:
        type: Event type, always "response.audio_transcript.delta".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        content_index: Index of the content part.
        delta: Incremental transcript text.
    """

    type: Literal["response.audio_transcript.delta"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseAudioTranscriptDone(ServerEvent):
    """Event indicating audio transcript is complete.

    Parameters:
        type: Event type, always "response.audio_transcript.done".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        content_index: Index of the content part.
        transcript: Complete transcript text.
    """

    type: Literal["response.audio_transcript.done"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    transcript: str


class ResponseAudioDelta(ServerEvent):
    """Event containing incremental audio data from a response.

    Parameters:
        type: Event type, always "response.audio.delta".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        content_index: Index of the content part.
        delta: Base64-encoded incremental audio data.
    """

    type: Literal["response.audio.delta"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str  # base64-encoded audio


class ResponseAudioDone(ServerEvent):
    """Event indicating audio content is complete.

    Parameters:
        type: Event type, always "response.audio.done".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        content_index: Index of the content part.
    """

    type: Literal["response.audio.done"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int


class ResponseFunctionCallArgumentsDelta(ServerEvent):
    """Event containing incremental function call arguments.

    Parameters:
        type: Event type, always "response.function_call_arguments.delta".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        call_id: ID of the function call.
        delta: Incremental function arguments as JSON.
    """

    type: Literal["response.function_call_arguments.delta"]
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    delta: str


class ResponseFunctionCallArgumentsDone(ServerEvent):
    """Event indicating function call arguments are complete.

    Parameters:
        type: Event type, always "response.function_call_arguments.done".
        response_id: ID of the response.
        item_id: ID of the conversation item.
        output_index: Index of the output item.
        call_id: ID of the function call.
        arguments: Complete function arguments as JSON string.
    """

    type: Literal["response.function_call_arguments.done"]
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    arguments: str


class InputAudioBufferSpeechStarted(ServerEvent):
    """Event indicating speech has started in the input audio buffer.

    Parameters:
        type: Event type, always "input_audio_buffer.speech_started".
        audio_start_ms: Start time of speech in milliseconds.
        item_id: ID of the associated conversation item.
    """

    type: Literal["input_audio_buffer.speech_started"]
    audio_start_ms: int
    item_id: str


class InputAudioBufferSpeechStopped(ServerEvent):
    """Event indicating speech has stopped in the input audio buffer.

    Parameters:
        type: Event type, always "input_audio_buffer.speech_stopped".
        audio_end_ms: End time of speech in milliseconds.
        item_id: ID of the associated conversation item.
    """

    type: Literal["input_audio_buffer.speech_stopped"]
    audio_end_ms: int
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


class ErrorEvent(ServerEvent):
    """Event indicating an error occurred.

    Parameters:
        type: Event type, always "error".
        error: Error details.
    """

    type: Literal["error"]
    error: RealtimeError


class RateLimitsUpdated(ServerEvent):
    """Event indicating rate limits have been updated.

    Parameters:
        type: Event type, always "rate_limits.updated".
        rate_limits: List of rate limit information.
    """

    type: Literal["rate_limits.updated"]
    rate_limits: List[Dict[str, Any]]


class TokenDetails(BaseModel):
    """Detailed token usage information.

    Parameters:
        cached_tokens: Number of cached tokens used. Defaults to 0.
        text_tokens: Number of text tokens used. Defaults to 0.
        audio_tokens: Number of audio tokens used. Defaults to 0.
    """

    cached_tokens: Optional[int] = 0
    text_tokens: Optional[int] = 0
    audio_tokens: Optional[int] = 0

    class Config:
        """Pydantic configuration for TokenDetails."""

        extra = "allow"


class Usage(BaseModel):
    """Token usage statistics for a response.

    Parameters:
        total_tokens: Total number of tokens used.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens used.
        input_token_details: Detailed breakdown of input token usage.
        output_token_details: Detailed breakdown of output token usage.
    """

    total_tokens: int
    input_tokens: int
    output_tokens: int
    input_token_details: TokenDetails
    output_token_details: TokenDetails


class Response(BaseModel):
    """A complete assistant response.

    Parameters:
        id: Unique identifier for the response.
        object: Object type, always "realtime.response".
        status: Current status of the response.
        status_details: Additional status information.
        output: List of conversation items in the response.
        usage: Token usage statistics for the response.
    """

    id: str
    object: Literal["realtime.response"]
    status: Literal["completed", "in_progress", "incomplete", "cancelled", "failed"]
    status_details: Any
    output: List[ConversationItem]
    usage: Optional[Usage] = None


_server_event_types = {
    "error": ErrorEvent,
    "session.created": SessionCreatedEvent,
    "session.updated": SessionUpdatedEvent,
    "conversation.created": ConversationCreated,
    "input_audio_buffer.committed": InputAudioBufferCommitted,
    "input_audio_buffer.cleared": InputAudioBufferCleared,
    "input_audio_buffer.speech_started": InputAudioBufferSpeechStarted,
    "input_audio_buffer.speech_stopped": InputAudioBufferSpeechStopped,
    "conversation.item.created": ConversationItemCreated,
    "conversation.item.input_audio_transcription.delta": ConversationItemInputAudioTranscriptionDelta,
    "conversation.item.input_audio_transcription.completed": ConversationItemInputAudioTranscriptionCompleted,
    "conversation.item.input_audio_transcription.failed": ConversationItemInputAudioTranscriptionFailed,
    "conversation.item.truncated": ConversationItemTruncated,
    "conversation.item.deleted": ConversationItemDeleted,
    "conversation.item.retrieved": ConversationItemRetrieved,
    "response.created": ResponseCreated,
    "response.done": ResponseDone,
    "response.output_item.added": ResponseOutputItemAdded,
    "response.output_item.done": ResponseOutputItemDone,
    "response.content_part.added": ResponseContentPartAdded,
    "response.content_part.done": ResponseContentPartDone,
    "response.text.delta": ResponseTextDelta,
    "response.text.done": ResponseTextDone,
    "response.audio_transcript.delta": ResponseAudioTranscriptDelta,
    "response.audio_transcript.done": ResponseAudioTranscriptDone,
    "response.audio.delta": ResponseAudioDelta,
    "response.audio.done": ResponseAudioDone,
    "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDelta,
    "response.function_call_arguments.done": ResponseFunctionCallArgumentsDone,
    "rate_limits.updated": RateLimitsUpdated,
}


def parse_server_event(str):
    """Parse a server event from JSON string.

    Args:
        str: JSON string containing the server event.

    Returns:
        Parsed server event object of the appropriate type.

    Raises:
        Exception: If the event type is unimplemented or parsing fails.
    """
    try:
        event = json.loads(str)
        event_type = event["type"]
        if event_type not in _server_event_types:
            raise Exception(f"Unimplemented server event type: {event_type}")
        return _server_event_types[event_type].model_validate(event)
    except Exception as e:
        raise Exception(f"{e} \n\n{str}")
