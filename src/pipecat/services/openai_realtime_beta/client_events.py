import json
import uuid

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Union

#
# session properties
#


class InputAudioTranscription(BaseModel):
    model: Optional[str] = "whisper-1"


class TurnDetection(BaseModel):
    type: Optional[Literal["server_vad"]] = "server_vad"
    threshold: Optional[float] = 0.5
    prefix_padding_ms: Optional[int] = 300
    silence_duration_ms: Optional[int] = 800


class SessionProperties(BaseModel):
    modalities: Optional[List[Literal["text", "audio"]]] = ["text", "audio"]
    instructions: Optional[str] = None
    voice: Optional[str] = "alloy"
    input_audio_format: Optional[Literal["pcm16", "g711_ulaw", "g711_alaw"]] = "pcm16"
    output_audio_format: Optional[Literal["pcm16", "g711_ulaw", "g711_alaw"]] = "pcm16"
    input_audio_transcription: Optional[InputAudioTranscription] = InputAudioTranscription()
    turn_detection: Optional[TurnDetection] = TurnDetection()
    tools: Optional[List[Dict]] = []
    tool_choice: Optional[Literal["auto", "none", "required"]] = "auto"
    temperature: Optional[float] = 0.8
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = Field(default=4096)


#
# context
#


class ItemContent(BaseModel):
    type: Literal["text", "audio", "input_text", "input_audio"]
    text: Optional[str] = None
    audio: Optional[str] = None  # base64-encoded audio
    transcript: Optional[str] = None


class ConversationItem(BaseModel):
    id: str
    object: Literal["realtime.item"]
    type: Literal["message", "function_call", "function_call_output"]
    status: Optional[Literal["completed", "in_progress", "incomplete"]] = None
    # role and content are present for message items
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[List[ItemContent]] = None
    # these four fields are present for function_call items
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    Output: Optional[str] = None


class RealtimeConversation(BaseModel):
    id: str
    object: Literal["realtime.conversation"]


#
# error class
#
class RealtimeError(BaseModel):
    type: str
    code: str
    message: str
    param: Optional[str] = None


#
# client events
#


class ClientEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SessionUpdateEvent(ClientEvent):
    session_properties: Optional[SessionProperties] = None


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
    status: Literal["completed", "in_progress", "incomplete"]
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
    "conversation.item.input_audio_transcription.completed": ConversationItemInputAudioTranscriptionCompleted,
    "conversation.item.input_audio_transcription.failed": ConversationItemInputAudioTranscriptionFailed,
    "conversation.item.truncated": ConversationItemTruncated,
    "conversation.item.deleted": ConversationItemDeleted,
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
