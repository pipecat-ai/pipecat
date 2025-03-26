#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
#

import json
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

#
# session properties
#


class InputAudioTranscription(BaseModel):
    """Configuration for audio transcription settings.

    Attributes:
        model: Transcription model to use (e.g., "gpt-4o-transcribe", "whisper-1").
        language: Optional language code for transcription.
        prompt: Optional transcription hint text.
    """

    model: str = "gpt-4o-transcribe"
    language: Optional[str]
    prompt: Optional[str]

    def __init__(
        self,
        model: Optional[str] = "gpt-4o-transcribe",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__(model=model, language=language, prompt=prompt)
        if self.model != "gpt-4o-transcribe" and (self.language or self.prompt):
            raise ValueError(
                "Fields 'language' and 'prompt' are only supported when model is 'gpt-4o-transcribe'"
            )


class TurnDetection(BaseModel):
    type: Optional[Literal["server_vad"]] = "server_vad"
    threshold: Optional[float] = 0.5
    prefix_padding_ms: Optional[int] = 300
    silence_duration_ms: Optional[int] = 800


class SemanticTurnDetection(BaseModel):
    type: Optional[Literal["semantic_vad"]] = "semantic_vad"
    eagerness: Optional[Literal["low", "medium", "high", "auto"]] = None
    create_response: Optional[bool] = None
    interrupt_response: Optional[bool] = None


class InputAudioNoiseReduction(BaseModel):
    type: Optional[Literal["near_field", "far_field"]]


class SessionProperties(BaseModel):
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
    type: Literal["text", "audio", "input_text", "input_audio"]
    text: Optional[str] = None
    audio: Optional[str] = None  # base64-encoded audio
    transcript: Optional[str] = None


class ConversationItem(BaseModel):
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
    id: str
    object: Literal["realtime.conversation"]


class ResponseProperties(BaseModel):
    modalities: Optional[List[Literal["text", "audio"]]] = ["audio", "text"]
    instructions: Optional[str] = None
    voice: Optional[str] = None
    output_audio_format: Optional[Literal["pcm16", "g711_ulaw", "g711_alaw"]] = None
    tools: Optional[List[Dict]] = []
    tool_choice: Optional[Literal["auto", "none", "required"]] = None
    temperature: Optional[float] = None
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = None


#
# error class
#
class RealtimeError(BaseModel):
    type: str
    code: Optional[str] = ""
    message: str
    param: Optional[str] = None
    event_id: Optional[str] = None


#
# client events
#


class ClientEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SessionUpdateEvent(ClientEvent):
    type: Literal["session.update"] = "session.update"
    session: SessionProperties

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(*args, **kwargs)

        # Handle turn_detection so that False is serialized as null
        if "turn_detection" in dump["session"]:
            if dump["session"]["turn_detection"] is False:
                dump["session"]["turn_detection"] = None

        return dump


class InputAudioBufferAppendEvent(ClientEvent):
    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str  # base64-encoded audio


class InputAudioBufferCommitEvent(ClientEvent):
    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"


class InputAudioBufferClearEvent(ClientEvent):
    type: Literal["input_audio_buffer.clear"] = "input_audio_buffer.clear"


class ConversationItemCreateEvent(ClientEvent):
    type: Literal["conversation.item.create"] = "conversation.item.create"
    previous_item_id: Optional[str] = None
    item: ConversationItem


class ConversationItemTruncateEvent(ClientEvent):
    type: Literal["conversation.item.truncate"] = "conversation.item.truncate"
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemDeleteEvent(ClientEvent):
    type: Literal["conversation.item.delete"] = "conversation.item.delete"
    item_id: str


class ConversationItemRetrieveEvent(ClientEvent):
    type: Literal["conversation.item.retrieve"] = "conversation.item.retrieve"
    item_id: str


class ResponseCreateEvent(ClientEvent):
    type: Literal["response.create"] = "response.create"
    response: Optional[ResponseProperties] = None


class ResponseCancelEvent(ClientEvent):
    type: Literal["response.cancel"] = "response.cancel"


#
# server events
#


class ServerEvent(BaseModel):
    event_id: str
    type: str

    class Config:
        arbitrary_types_allowed = True


class SessionCreatedEvent(ServerEvent):
    type: Literal["session.created"]
    session: SessionProperties


class SessionUpdatedEvent(ServerEvent):
    type: Literal["session.updated"]
    session: SessionProperties


class ConversationCreated(ServerEvent):
    type: Literal["conversation.created"]
    conversation: RealtimeConversation


class ConversationItemCreated(ServerEvent):
    type: Literal["conversation.item.created"]
    previous_item_id: Optional[str] = None
    item: ConversationItem


class ConversationItemInputAudioTranscriptionDelta(ServerEvent):
    type: Literal["conversation.item.input_audio_transcription.delta"]
    item_id: str
    content_index: int
    delta: str


class ConversationItemInputAudioTranscriptionCompleted(ServerEvent):
    type: Literal["conversation.item.input_audio_transcription.completed"]
    item_id: str
    content_index: int
    transcript: str


class ConversationItemInputAudioTranscriptionFailed(ServerEvent):
    type: Literal["conversation.item.input_audio_transcription.failed"]
    item_id: str
    content_index: int
    error: RealtimeError


class ConversationItemTruncated(ServerEvent):
    type: Literal["conversation.item.truncated"]
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemDeleted(ServerEvent):
    type: Literal["conversation.item.deleted"]
    item_id: str


class ConversationItemRetrieved(ServerEvent):
    type: Literal["conversation.item.retrieved"]
    item: ConversationItem


class ResponseCreated(ServerEvent):
    type: Literal["response.created"]
    response: "Response"


class ResponseDone(ServerEvent):
    type: Literal["response.done"]
    response: "Response"


class ResponseOutputItemAdded(ServerEvent):
    type: Literal["response.output_item.added"]
    response_id: str
    output_index: int
    item: ConversationItem


class ResponseOutputItemDone(ServerEvent):
    type: Literal["response.output_item.done"]
    response_id: str
    output_index: int
    item: ConversationItem


class ResponseContentPartAdded(ServerEvent):
    type: Literal["response.content_part.added"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: ItemContent


class ResponseContentPartDone(ServerEvent):
    type: Literal["response.content_part.done"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: ItemContent


class ResponseTextDelta(ServerEvent):
    type: Literal["response.text.delta"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseTextDone(ServerEvent):
    type: Literal["response.text.done"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseAudioTranscriptDelta(ServerEvent):
    type: Literal["response.audio_transcript.delta"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseAudioTranscriptDone(ServerEvent):
    type: Literal["response.audio_transcript.done"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    transcript: str


class ResponseAudioDelta(ServerEvent):
    type: Literal["response.audio.delta"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str  # base64-encoded audio


class ResponseAudioDone(ServerEvent):
    type: Literal["response.audio.done"]
    response_id: str
    item_id: str
    output_index: int
    content_index: int


class ResponseFunctionCallArgumentsDelta(ServerEvent):
    type: Literal["response.function_call_arguments.delta"]
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    delta: str


class ResponseFunctionCallArgumentsDone(ServerEvent):
    type: Literal["response.function_call_arguments.done"]
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    arguments: str


class InputAudioBufferSpeechStarted(ServerEvent):
    type: Literal["input_audio_buffer.speech_started"]
    audio_start_ms: int
    item_id: str


class InputAudioBufferSpeechStopped(ServerEvent):
    type: Literal["input_audio_buffer.speech_stopped"]
    audio_end_ms: int
    item_id: str


class InputAudioBufferCommitted(ServerEvent):
    type: Literal["input_audio_buffer.committed"]
    previous_item_id: Optional[str] = None
    item_id: str


class InputAudioBufferCleared(ServerEvent):
    type: Literal["input_audio_buffer.cleared"]


class ErrorEvent(ServerEvent):
    type: Literal["error"]
    error: RealtimeError


class RateLimitsUpdated(ServerEvent):
    type: Literal["rate_limits.updated"]
    rate_limits: List[Dict[str, Any]]


class TokenDetails(BaseModel):
    cached_tokens: Optional[int] = 0
    text_tokens: Optional[int] = 0
    audio_tokens: Optional[int] = 0

    class Config:
        extra = "allow"


class Usage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int
    input_token_details: TokenDetails
    output_token_details: TokenDetails


class Response(BaseModel):
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
    try:
        event = json.loads(str)
        event_type = event["type"]
        if event_type not in _server_event_types:
            raise Exception(f"Unimplemented server event type: {event_type}")
        return _server_event_types[event_type].model_validate(event)
    except Exception as e:
        raise Exception(f"{e} \n\n{str}")
