#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Event and session models for the OpenAI Live API (WebSocket transport).

The Live API is a full-duplex speech-to-speech API: the model listens and
speaks at the same time and decides on its own when to talk. Clients configure
a session, stream audio in, receive audio and transcript events out, and
answer the model's *delegations* (units of work it hands to a backend model).

Server events are parsed leniently: fields the alpha adds are kept, and event
types this module doesn't model yet become :class:`UnknownServerEvent` rather
than errors, since the API is expected to change during the alpha.
"""

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Value of the ``OpenAI-Alpha`` header that selects the Live API alpha.
OPENAI_LIVE_ALPHA_HEADER = "quicksilver=v2"

#: Maximum number of ``initial_items`` a session accepts.
MAX_INITIAL_ITEMS = 128

ContextChannel = Literal["speakable", "commentary", "developer"]
DelegationChannel = Literal["speakable", "commentary"]


#
# session configuration
#


class InputTextContent(BaseModel):
    """A text content part supplied by the client.

    Parameters:
        type: Content type, always "input_text".
        text: The text.
    """

    type: Literal["input_text"] = "input_text"
    text: str


class OutputTextContent(BaseModel):
    """A text content part attributed to the assistant.

    Parameters:
        type: Content type, always "output_text".
        text: The text.
    """

    type: Literal["output_text"] = "output_text"
    text: str


class InitialItem(BaseModel):
    """A prior text-only conversation message seeded at session start.

    Parameters:
        type: Item type, always "message".
        role: Message role. System, developer and user messages carry
            ``input_text`` content; assistant messages carry ``output_text``.
        content: Exactly one text content part.
    """

    type: Literal["message"] = "message"
    role: Literal["system", "developer", "user", "assistant"]
    content: list[InputTextContent | OutputTextContent]


class AudioOutputConfig(BaseModel):
    """Output audio configuration.

    Parameters:
        voice: Output voice name (defaults to ``marin`` server-side).
    """

    voice: str | None = None


class AudioFormat(BaseModel):
    """WebSocket audio format, selected once at session start for input and output.

    Parameters:
        type: ``audio/pcm`` (24 kHz PCM16), ``audio/pcmu`` or ``audio/pcma`` (8 kHz G.711).
        rate: Sample rate in Hz.
    """

    type: Literal["audio/pcm", "audio/pcmu", "audio/pcma"] = "audio/pcm"
    rate: int = 24000


class AudioConfig(BaseModel):
    """Session audio configuration.

    Parameters:
        output: Output audio configuration.
        format: WebSocket audio format.
    """

    output: AudioOutputConfig | None = None
    format: AudioFormat | None = None


class ClientDelegationConfig(BaseModel):
    """Delegation configuration selecting client delegation.

    Parameters:
        type: Delegation type, always "client".
    """

    type: Literal["client"] = "client"


class ResponsesDelegationConfig(BaseModel):
    """Delegation configuration selecting Responses delegation.

    Parameters:
        type: Delegation type, always "responses".
        responses: Configuration of the backend Responses model: ``model``,
            ``instructions``, ``tools``, ``tool_choice``, ``reasoning``,
            ``max_output_tokens``, ``service_tier``, and other Responses
            request fields the Live API accepts.
    """

    type: Literal["responses"] = "responses"
    responses: dict[str, Any]


class SessionConfig(BaseModel):
    """Live session configuration sent with ``session.update``.

    The first ``session.update`` on a WebSocket configures the session
    (``model`` comes from the connection URL). Later updates are sparse:
    omitted fields keep their values.

    Parameters:
        instructions: System instructions for the live model. Immutable after start.
        audio: Audio configuration. The output voice is immutable after start.
        delegation: Delegation mode. Omitted selects client delegation.
        initial_items: Prior text-only conversation messages. Startup-only.
    """

    instructions: str | None = None
    audio: AudioConfig | None = None
    delegation: ClientDelegationConfig | ResponsesDelegationConfig | None = None
    initial_items: list[InitialItem] | None = None


#
# client events
#


class ClientEvent(BaseModel):
    """Base class for events sent to the Live API.

    Parameters:
        event_id: Client-chosen identifier, echoed back only on errors.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SessionUpdateEvent(ClientEvent):
    """Configure the session (first event) or apply a sparse update.

    Parameters:
        type: Event type, always "session.update".
        session: The session configuration.
    """

    type: Literal["session.update"] = "session.update"
    session: SessionConfig


class InputAudioAppendEvent(ClientEvent):
    """Append audio in the session's audio format.

    Parameters:
        type: Event type, always "input_audio.append".
        audio: Base64-encoded audio bytes.
    """

    type: Literal["input_audio.append"] = "input_audio.append"
    audio: str


class SessionContextAppendEvent(ClientEvent):
    """Append general text context to the session.

    Parameters:
        type: Event type, always "session.context.append".
        channel: ``speakable`` (default), ``commentary`` or ``developer``.
        content: Exactly one ``input_text`` part.
    """

    type: Literal["session.context.append"] = "session.context.append"
    channel: ContextChannel | None = None
    content: list[InputTextContent]


class DelegationContextAppendEvent(ClientEvent):
    """Return context for a client-targeted delegation.

    Parameters:
        type: Event type, always "delegation.context.append".
        delegation_item_id: The ``item.id`` from the ``delegation.created`` event.
        channel: ``speakable`` (default) or ``commentary``.
        content: Exactly one ``input_text`` part.
    """

    type: Literal["delegation.context.append"] = "delegation.context.append"
    delegation_item_id: str
    channel: DelegationChannel | None = None
    content: list[InputTextContent]


class FunctionCallOutputItem(BaseModel):
    """The result of a client-actionable function call from a Responses delegation.

    Parameters:
        type: Item type, always "function_call_output".
        call_id: The ``call_id`` of the completed ``function_call`` item.
        output: The function output, as a string.
    """

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


class DelegationFunctionCallOutputCreateEvent(ClientEvent):
    """Return one function-call result for a Responses delegation.

    Parameters:
        type: Event type, always "delegation.function_call_output.create".
        item: The function call output.
    """

    type: Literal["delegation.function_call_output.create"] = (
        "delegation.function_call_output.create"
    )
    item: FunctionCallOutputItem


class SessionCloseEvent(ClientEvent):
    """Request a graceful shutdown; the server answers with ``session.closed``.

    Parameters:
        type: Event type, always "session.close".
    """

    type: Literal["session.close"] = "session.close"


#
# server events
#


class ServerEvent(BaseModel):
    """Base class for events received from the Live API.

    Unknown fields are retained so events keep parsing as the alpha evolves.

    Parameters:
        type: The event type.
    """

    model_config = ConfigDict(extra="allow")

    type: str


class SessionResource(BaseModel):
    """The public session resource echoed by ``session.started`` / ``session.updated``.

    Parameters:
        id: Session identifier.
        expires_at: Session expiry as a Unix timestamp in seconds.
        model: The live model.
        instructions: System instructions.
        audio: Audio configuration.
        delegation: Delegation configuration.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    expires_at: int | None = None
    model: str | None = None
    instructions: str | None = None
    audio: dict[str, Any] | None = None
    delegation: dict[str, Any] | None = None


class SessionStartedEvent(ServerEvent):
    """The session is configured and ready for audio.

    Parameters:
        type: Event type, always "session.started".
        session: The session resource.
    """

    type: Literal["session.started"]
    session: SessionResource


class SessionUpdatedEvent(ServerEvent):
    """A sparse ``session.update`` was applied.

    Parameters:
        type: Event type, always "session.updated".
        session: The complete session resource.
    """

    type: Literal["session.updated"]
    session: SessionResource


class OutputAudioDeltaEvent(ServerEvent):
    """A chunk of output audio in the session's audio format.

    Parameters:
        type: Event type, always "output_audio.delta".
        audio: Base64-encoded audio bytes.
        start_ms: Start of the chunk on the server timeline.
        end_ms: End of the chunk on the server timeline.
    """

    type: Literal["output_audio.delta"]
    audio: str
    start_ms: int | None = None
    end_ms: int | None = None


class Turn(BaseModel):
    """A projected transcript turn.

    Parameters:
        id: Turn identifier.
        role: "user" or "assistant".
        start_ms: Start on the server timeline.
        end_ms: End on the server timeline.
        transcript: The transcript observed so far (complete on ``turn.done``).
    """

    model_config = ConfigDict(extra="allow")

    id: str
    role: str
    start_ms: int | None = None
    end_ms: int | None = None
    transcript: str = ""


class TurnCreatedEvent(ServerEvent):
    """A projected transcript turn started.

    Parameters:
        type: Event type, always "turn.created".
        turn: The turn.
    """

    type: Literal["turn.created"]
    turn: Turn


class TurnDeltaEvent(ServerEvent):
    """A projected transcript turn grew.

    Parameters:
        type: Event type, always "turn.delta".
        turn_id: The turn identifier.
        delta: Transcript text appended to the turn.
        start_ms: Start of the delta on the server timeline.
        end_ms: End of the delta on the server timeline.
    """

    type: Literal["turn.delta"]
    turn_id: str
    delta: str = ""
    start_ms: int | None = None
    end_ms: int | None = None


class TurnDoneEvent(ServerEvent):
    """A projected transcript turn reached its final form.

    Parameters:
        type: Event type, always "turn.done".
        turn: The complete turn.
    """

    type: Literal["turn.done"]
    turn: Turn


class TranscriptItem(BaseModel):
    """A timed transcript fragment.

    Parameters:
        id: Item identifier.
        type: "input_transcript" or "output_transcript".
        text: The fragment text.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    type: str | None = None
    text: str = ""


class TranscriptAddedEvent(ServerEvent):
    """A complete timed input or output transcript fragment.

    Parameters:
        type: "input_transcript.added" or "output_transcript.added".
        item: The fragment.
        start_ms: Start on the server timeline.
        end_ms: End on the server timeline.
    """

    type: Literal["input_transcript.added", "output_transcript.added"]
    item: TranscriptItem
    start_ms: int | None = None
    end_ms: int | None = None


class DelegationItem(BaseModel):
    """A unit of work the live model delegated.

    Parameters:
        id: Item identifier; the ``delegation_item_id`` for client delegations.
        type: Item type, "delegation".
        target: "client" or "responses".
        response_id: For Responses delegations, the id of the Responses run.
        content: The delegated request as ``input_text`` parts.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    type: str = "delegation"
    target: str
    response_id: str | None = None
    content: list[InputTextContent] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """The delegated request text."""
        return "\n".join(part.text for part in self.content if part.text).strip()


class DelegationCreatedEvent(ServerEvent):
    """The live model created a delegation.

    Parameters:
        type: Event type, always "delegation.created".
        offset_ms: Position on the server timeline.
        item: The delegation item.
    """

    type: Literal["delegation.created"]
    offset_ms: int | None = None
    item: DelegationItem


class ResponseOutputItem(BaseModel):
    """An output item of a Responses delegation.

    Parameters:
        id: Item identifier.
        type: Item type, e.g. "function_call" or "message".
        status: Item status, e.g. "completed".
        call_id: For function calls, the call identifier to answer with.
        name: For function calls, the function name.
        arguments: For function calls, the JSON-encoded arguments.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    type: str | None = None
    status: str | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None


class ResponseOutputItemDoneEvent(ServerEvent):
    """A Responses delegation completed an output item.

    Parameters:
        type: Event type, always "response.output_item.done".
        item: The completed item.
        output_index: Index of the item in the response output.
        sequence_number: Event sequence number.
    """

    type: Literal["response.output_item.done"]
    item: ResponseOutputItem
    output_index: int | None = None
    sequence_number: int | None = None


class ResponseEvent(ServerEvent):
    """Any other ``response.*`` event of a Responses delegation, passed through unwrapped.

    Parameters:
        type: The Responses event type.
        response: The response resource, when the event carries one.
    """

    response: dict[str, Any] | None = None


class SessionContextAppendedEvent(ServerEvent):
    """General context was placed.

    Parameters:
        type: Event type, always "session.context.appended".
        start_ms: Start of the placement range.
        end_ms: End of the placement range.
    """

    type: Literal["session.context.appended"]
    start_ms: int | None = None
    end_ms: int | None = None


class DelegationContextAppendedEvent(ServerEvent):
    """Context for a client delegation was placed.

    Parameters:
        type: Event type, always "delegation.context.appended".
        delegation_item_id: The delegation the context belongs to.
        start_ms: Start of the placement range.
        end_ms: End of the placement range.
    """

    type: Literal["delegation.context.appended"]
    delegation_item_id: str
    start_ms: int | None = None
    end_ms: int | None = None


class DelegationFunctionCallOutputCreatedEvent(ServerEvent):
    """A function-call result was accepted.

    Parameters:
        type: Event type, always "delegation.function_call_output.created".
        item: The accepted output item, including its assigned id.
    """

    type: Literal["delegation.function_call_output.created"]
    item: dict[str, Any]


class UsageTokenDetails(BaseModel):
    """Token breakdown by modality.

    Parameters:
        text_tokens: Text tokens.
        audio_tokens: Audio tokens.
        image_tokens: Image tokens.
        cached_tokens: Tokens served from cache.
    """

    model_config = ConfigDict(extra="allow")

    text_tokens: int | None = None
    audio_tokens: int | None = None
    image_tokens: int | None = None
    cached_tokens: int | None = None


class Usage(BaseModel):
    """Cumulative session usage.

    Two shapes exist: token counts (``total_tokens`` and details), or
    durations (``audio_duration_ms`` plus per-backend-model token usage).

    Parameters:
        total_tokens: Total tokens.
        input_tokens: Input tokens.
        output_tokens: Output tokens.
        input_token_details: Input token breakdown.
        output_token_details: Output token breakdown.
        audio_duration_ms: Frontend audio duration (duration shape).
        backend_model_usage: Per-backend-model token usage (duration shape).
    """

    model_config = ConfigDict(extra="allow")

    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_token_details: UsageTokenDetails | None = None
    output_token_details: UsageTokenDetails | None = None
    audio_duration_ms: int | None = None
    backend_model_usage: list[dict[str, Any]] | None = None


class SessionUsageUpdatedEvent(ServerEvent):
    """Periodic cumulative usage report.

    Parameters:
        type: Event type, always "session.usage.updated".
        usage: Cumulative usage.
        usage_limit: Usage limit status, when reported.
    """

    type: Literal["session.usage.updated"]
    usage: Usage
    usage_limit: dict[str, Any] | None = None


class SessionClosedEvent(ServerEvent):
    """Graceful shutdown completed.

    Parameters:
        type: Event type, always "session.closed".
        reason: Why the session closed.
        usage: Final cumulative usage.
    """

    type: Literal["session.closed"]
    reason: str | None = None
    usage: Usage | None = None


class SessionContextWindowApproachingEvent(ServerEvent):
    """Replacement inference context is being prepared.

    Parameters:
        type: Event type, always "session.context_window.approaching".
        rollover_id: Correlation id for the rollover.
        expires_at: Unix timestamp in seconds.
    """

    type: Literal["session.context_window.approaching"]
    rollover_id: str | None = None
    expires_at: int | None = None


class SessionContextWindowRolledOverEvent(ServerEvent):
    """Replacement inference context finished initializing.

    Parameters:
        type: Event type, always "session.context_window.rolled_over".
        rollover_id: Correlation id for the rollover.
    """

    type: Literal["session.context_window.rolled_over"]
    rollover_id: str | None = None


class InputAudioPausedEvent(ServerEvent):
    """Microphone input was replaced with silence.

    Parameters:
        type: Event type, always "input_audio.paused".
    """

    type: Literal["input_audio.paused"]


class InputAudioResumedEvent(ServerEvent):
    """Microphone input resumed.

    Parameters:
        type: Event type, always "input_audio.resumed".
    """

    type: Literal["input_audio.resumed"]


class InputAudioDTMFEventReceivedEvent(ServerEvent):
    """A SIP caller sent a DTMF keypress.

    Parameters:
        type: Event type, always "input_audio.dtmf_event_received".
        event: The key: 0-9, *, # or A-D.
    """

    type: Literal["input_audio.dtmf_event_received"]
    event: str


class ErrorDetails(BaseModel):
    """The error envelope shared by Live API and normalized Responses errors.

    Parameters:
        type: Error type, e.g. "invalid_request_error".
        code: Error code.
        message: Human-readable message.
        param: The offending field, when one caused the failure.
        event_id: The client event the error correlates to, when known.
    """

    model_config = ConfigDict(extra="allow")

    type: str | None = None
    code: str | None = None
    message: str | None = None
    param: str | None = None
    event_id: str | None = None


class ErrorEvent(ServerEvent):
    """A startup, validation, command or delegated-Responses error.

    Parameters:
        type: Event type, always "error".
        error: The error details.
    """

    type: Literal["error"]
    error: ErrorDetails


class UnknownServerEvent(ServerEvent):
    """A server event this module doesn't model; its fields are retained as extras."""


_server_event_types: dict[str, type[ServerEvent]] = {
    "error": ErrorEvent,
    "session.started": SessionStartedEvent,
    "session.updated": SessionUpdatedEvent,
    "session.context_window.approaching": SessionContextWindowApproachingEvent,
    "session.context_window.rolled_over": SessionContextWindowRolledOverEvent,
    "session.context.appended": SessionContextAppendedEvent,
    "session.usage.updated": SessionUsageUpdatedEvent,
    "session.closed": SessionClosedEvent,
    "output_audio.delta": OutputAudioDeltaEvent,
    "input_audio.paused": InputAudioPausedEvent,
    "input_audio.resumed": InputAudioResumedEvent,
    "input_audio.dtmf_event_received": InputAudioDTMFEventReceivedEvent,
    "input_transcript.added": TranscriptAddedEvent,
    "output_transcript.added": TranscriptAddedEvent,
    "turn.created": TurnCreatedEvent,
    "turn.delta": TurnDeltaEvent,
    "turn.done": TurnDoneEvent,
    "delegation.created": DelegationCreatedEvent,
    "delegation.context.appended": DelegationContextAppendedEvent,
    "delegation.function_call_output.created": DelegationFunctionCallOutputCreatedEvent,
    "response.output_item.done": ResponseOutputItemDoneEvent,
}


def parse_server_event(message: str) -> ServerEvent:
    """Parse a server event from its JSON text.

    ``response.*`` events other than ``response.output_item.done`` become
    :class:`ResponseEvent`; event types this module doesn't model become
    :class:`UnknownServerEvent`.

    Args:
        message: The JSON text of one server event.

    Returns:
        The parsed event.

    Raises:
        ValueError: If the text isn't a JSON object with a string ``type``, or
            a modeled event fails validation.
    """
    try:
        data = json.loads(message)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid server event JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("type"), str):
        raise ValueError(f"Server event is not an object with a string type: {message}")

    event_type = data["type"]
    model = _server_event_types.get(event_type)
    if model is None:
        model = ResponseEvent if event_type.startswith("response.") else UnknownServerEvent
    try:
        return model.model_validate(data)
    except Exception as e:
        raise ValueError(f"Invalid {event_type} server event: {e}") from e
