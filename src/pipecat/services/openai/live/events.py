#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Event and session models for the OpenAI Live API (WebSocket transport).

The Live API is a full-duplex speech-to-speech API: the model listens and
speaks at the same time and decides on its own when to talk. Clients start a
session, stream audio in, receive audio and transcript events out, and answer
the model's *delegations* (units of work it hands to a backend model).

Server events are parsed leniently: fields the alpha adds are kept, and event
types this module doesn't model yet become :class:`UnknownServerEvent` rather
than errors, since the API is expected to change during the alpha.
"""

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Value of the ``OpenAI-Alpha`` header that selects the Live API alpha.
OPENAI_LIVE_ALPHA_HEADER = "quicksilver=v3"

#: Maximum number of startup ``input`` messages a session accepts.
MAX_INPUT_ITEMS = 128


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


class InputItem(BaseModel):
    """A prior text-only conversation message seeded at session start.

    Parameters:
        type: Item type, always "message".
        role: Message role. Developer and user messages carry ``input_text``
            content; assistant messages carry ``output_text``. A ``system``
            role is not accepted here: application instructions belong in
            :attr:`SessionConfig.instructions` or a developer message.
        content: Exactly one text content part.
    """

    type: Literal["message"] = "message"
    role: Literal["developer", "user", "assistant"]
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
        type: ``audio/pcm`` (16 or 24 kHz PCM16), ``audio/pcmu`` or
            ``audio/pcma`` (8 kHz G.711).
        rate: Sample rate in Hz.
    """

    type: Literal["audio/pcm", "audio/pcmu", "audio/pcma"] = "audio/pcm"
    rate: int = 24000


class AudioConfig(BaseModel):
    """Session audio configuration.

    Parameters:
        output: Output audio configuration.
        format: WebSocket audio format, shared by input and output.
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
    """Live session configuration, sent once with ``session.start``.

    Parameters:
        model: The live model. Required, and immutable after start.
        instructions: Instructions for the live model. Immutable after start;
            :class:`SessionInstructionsAppendEvent` adds more later.
        audio: Audio configuration. The output voice and format are immutable
            after start.
        delegation: Delegation mode. Omitted selects client delegation. The
            mode is immutable: switching needs a new session.
        input: Prior text-only conversation messages. Startup-only.
    """

    model: str
    instructions: str | None = None
    audio: AudioConfig | None = None
    delegation: ClientDelegationConfig | ResponsesDelegationConfig | None = None
    input: list[InputItem] | None = None


class SessionUpdateConfig(BaseModel):
    """The sparse session update a running session accepts.

    Only delegation settings can change, and only within the mode chosen at
    startup. Startup fields (``model``, ``instructions``, ``audio``, ``input``)
    are not update fields.

    Parameters:
        delegation: Replacement delegation settings, of the session's mode.
    """

    delegation: ClientDelegationConfig | ResponsesDelegationConfig | None = None


#
# client events
#


class ClientEvent(BaseModel):
    """Base class for events sent to the Live API.

    Parameters:
        event_id: Client-chosen identifier, echoed back as ``client_event_id``
            on acknowledgments and errors.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def to_payload(self) -> dict[str, Any]:
        """Render the event as the JSON payload to send.

        Unset optional fields are omitted: the session schema is strict about
        unknown fields, and a ``null`` is not the same as an absent field.

        Returns:
            The payload.
        """
        return self.model_dump(exclude_none=True)


class SessionStartEvent(ClientEvent):
    """Start the session. The first message on a WebSocket, sent exactly once.

    Parameters:
        type: Event type, always "session.start".
        session: The session configuration, including the model.
    """

    type: Literal["session.start"] = "session.start"
    session: SessionConfig


class SessionUpdateEvent(ClientEvent):
    """Apply a sparse update to a running session.

    Parameters:
        type: Event type, always "session.update".
        session: The sparse update.
    """

    type: Literal["session.update"] = "session.update"
    session: SessionUpdateConfig


class InputAudioAppendEvent(ClientEvent):
    """Append audio in the session's audio format. Not acknowledged.

    Parameters:
        type: Event type, always "session.input_audio.append".
        audio: Base64-encoded audio bytes.
    """

    type: Literal["session.input_audio.append"] = "session.input_audio.append"
    audio: str


class ContextAppendEvent(ClientEvent):
    """Base class for the three context-append events.

    Parameters:
        delegation_id: The client delegation this content belongs to, or
            ``None`` for general session context. Always sent, including when
            it is ``None``.
        content: The text, at most 500 tokens.
    """

    delegation_id: str | None
    content: str

    def to_payload(self) -> dict[str, Any]:
        """Render the event, keeping ``delegation_id`` even when it is ``None``.

        The field is required on these events, and ``None`` is meaningful: it
        marks the content as general session context rather than belonging to
        a delegation.

        Returns:
            The payload.
        """
        return {**super().to_payload(), "delegation_id": self.delegation_id}


class SessionInstructionsAppendEvent(ContextAppendEvent):
    """Append instructions the live model will follow, without replacing the startup ones.

    Parameters:
        type: Event type, always "session.instructions.append".
    """

    type: Literal["session.instructions.append"] = "session.instructions.append"


class SessionThinkingAppendEvent(ContextAppendEvent):
    """Append information to the live model's internal reasoning.

    The content is not spoken when appended, though the model can draw on it
    when a later user request makes it relevant. It is not a secrecy boundary.

    Parameters:
        type: Event type, always "session.thinking.append".
    """

    type: Literal["session.thinking.append"] = "session.thinking.append"


class SessionCommentaryAppendEvent(ContextAppendEvent):
    """Append information for the live model to speak aloud, in its own words.

    Parameters:
        type: Event type, always "session.commentary.append".
    """

    type: Literal["session.commentary.append"] = "session.commentary.append"


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


class ResponseItemCreateEvent(ClientEvent):
    """Queue an input item for the backend Responses model. Not acknowledged.

    Queueing an item does not start inference: :class:`ResponseCreateEvent`
    does. Responses delegation only.

    Parameters:
        type: Event type, always "response.item.create".
        item: The input item, such as a function call output.
    """

    type: Literal["response.item.create"] = "response.item.create"
    item: FunctionCallOutputItem | dict[str, Any]


class ResponseCreateEvent(ClientEvent):
    """Start or continue delegated Responses work. Responses delegation only.

    Every output required by the pending response must be queued first, or
    the command is rejected with ``function_call_outputs_required``.

    Parameters:
        type: Event type, always "response.create".
    """

    type: Literal["response.create"] = "response.create"


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
        event_id: Identifier of this server event.
        client_event_id: The client event this one answers, when correlatable.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    event_id: str | None = None
    client_event_id: str | None = None


class SessionResource(BaseModel):
    """The resolved session, echoed by ``session.started``, ``session.updated`` and ``session.closed``.

    Parameters:
        id: Session identifier. Opaque: forward it unchanged.
        expires_at: Session expiry as a Unix timestamp in seconds.
        status: Session status.
        model: The live model.
        instructions: The startup instructions.
        audio: Audio configuration.
        delegation: Delegation configuration.
        input: The startup conversation history.
        context_management: Long-session context handling.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    expires_at: int | None = None
    status: str | None = None
    model: str | None = None
    instructions: str | None = None
    audio: dict[str, Any] | None = None
    delegation: dict[str, Any] | None = None
    input: list[dict[str, Any]] | None = None
    context_management: dict[str, Any] | None = None


class SessionStartedEvent(ServerEvent):
    """The session is configured and ready for audio.

    Parameters:
        type: Event type, always "session.started".
        session: The resolved session.
    """

    type: Literal["session.started"]
    session: SessionResource


class SessionUpdatedEvent(ServerEvent):
    """A sparse ``session.update`` was applied.

    Parameters:
        type: Event type, always "session.updated".
        session: The complete resolved session, not just the changed fields.
    """

    type: Literal["session.updated"]
    session: SessionResource


class OutputAudioDeltaEvent(ServerEvent):
    """A chunk of output audio in the session's audio format.

    Parameters:
        type: Event type, always "session.output_audio.delta".
        delta: Base64-encoded audio bytes.
    """

    type: Literal["session.output_audio.delta"]
    delta: str


class TranscriptDeltaEvent(ServerEvent):
    """A timed transcript fragment of user or assistant speech.

    Fragments are frame-aligned, not word- or turn-aligned: accumulate them in
    order, and group them into turns in the application if it needs turns.

    Parameters:
        type: "session.input_transcript.delta" (the user) or
            "session.output_transcript.delta" (the assistant).
        delta: The fragment text.
        start_ms: Start of the fragment on the session timeline.
        end_ms: End of the fragment on the session timeline.
    """

    type: Literal["session.input_transcript.delta", "session.output_transcript.delta"]
    delta: str = ""
    start_ms: int | None = None
    end_ms: int | None = None

    @property
    def role(self) -> Literal["user", "assistant"]:
        """The speaker."""
        return "user" if self.type == "session.input_transcript.delta" else "assistant"


class DelegationMetadata(BaseModel):
    """A unit of work the live model delegated.

    The metadata carries no task text: a client delegator works out what to do
    from the conversation and application state it keeps itself.

    Parameters:
        id: Delegation identifier, used to correlate returned context. Opaque:
            return it unchanged.
        type: Item type, always "delegation".
        target: "client" or "responses".
        response_id: For Responses delegations, the id of the Responses run.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    type: str = "delegation"
    target: str
    response_id: str | None = None


class SessionDelegationCreatedEvent(ServerEvent):
    """The live model created a delegation.

    Parameters:
        type: Event type, always "session.delegation.created".
        offset_ms: Position on the session timeline.
        delegation: The delegation metadata.
    """

    type: Literal["session.delegation.created"]
    offset_ms: int | None = None
    delegation: DelegationMetadata


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


class ResponseEventEnvelope(ServerEvent):
    """A Responses lifecycle event, wrapped with the delegation it belongs to.

    The nested event keeps its Responses meaning and is dispatched by its own
    complete ``type``. Lifecycle snapshots inside are reduced — ``output`` is
    an empty array even at ``response.completed`` — so collect output items
    from their individual events rather than from a snapshot.

    Parameters:
        type: Event type, always "response.event".
        delegation_id: The delegation this response belongs to, when known.
        event: The nested Responses event.
    """

    type: Literal["response.event"]
    delegation_id: str | None = None
    event: dict[str, Any] = Field(default_factory=dict)

    @property
    def inner_type(self) -> str:
        """The nested Responses event's type, or ``""`` if it has none."""
        inner = self.event.get("type")
        return inner if isinstance(inner, str) else ""


class ContextAppendedEvent(ServerEvent):
    """Appended context was accepted and placed on the session timeline.

    Acceptance is not proof the model has spoken the content.

    Parameters:
        type: "session.instructions.appended", "session.thinking.appended" or
            "session.commentary.appended".
        start_ms: Start of the accepted range.
        end_ms: End of the accepted range.
    """

    type: Literal[
        "session.instructions.appended",
        "session.thinking.appended",
        "session.commentary.appended",
    ]
    start_ms: int | None = None
    end_ms: int | None = None


class ContextWindowUsage(BaseModel):
    """Context utilization, when reported.

    Parameters:
        usage_ratio: Fraction of the context window in use.
    """

    model_config = ConfigDict(extra="allow")

    usage_ratio: float | None = None


class Usage(BaseModel):
    """Cumulative live session usage.

    Backend token usage is not here: it belongs to the wrapped Responses
    lifecycle, on the nested ``response.completed`` event.

    Parameters:
        seconds: Cumulative live audio duration in seconds.
    """

    model_config = ConfigDict(extra="allow")

    seconds: float | None = None


class SessionUsageUpdatedEvent(ServerEvent):
    """Periodic cumulative usage report.

    Parameters:
        type: Event type, always "session.usage.updated".
        usage: Cumulative usage, not an increment to sum.
        context_window: Context utilization, when reported.
    """

    type: Literal["session.usage.updated"]
    usage: Usage
    context_window: ContextWindowUsage | None = None


class SessionClosedEvent(ServerEvent):
    """Graceful shutdown completed. Transport closure alone is not finalization.

    Parameters:
        type: Event type, always "session.closed".
        reason: Why the session closed.
        session: The session snapshot, as configuration rather than a live session.
        usage: Final cumulative usage.
    """

    type: Literal["session.closed"]
    reason: str | None = None
    session: SessionResource | None = None
    usage: Usage | None = None


class ErrorDetails(BaseModel):
    """The error envelope shared by Live API and normalized Responses errors.

    Parameters:
        type: Error type, e.g. "invalid_request_error".
        code: Error code.
        message: Human-readable message.
        param: The offending field, when one caused the failure.
        client_event_id: The client event the error correlates to, when known.
    """

    model_config = ConfigDict(extra="allow")

    type: str | None = None
    code: str | None = None
    message: str | None = None
    param: str | None = None
    client_event_id: str | None = None


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
    "session.usage.updated": SessionUsageUpdatedEvent,
    "session.closed": SessionClosedEvent,
    "session.output_audio.delta": OutputAudioDeltaEvent,
    "session.input_transcript.delta": TranscriptDeltaEvent,
    "session.output_transcript.delta": TranscriptDeltaEvent,
    "session.delegation.created": SessionDelegationCreatedEvent,
    "session.instructions.appended": ContextAppendedEvent,
    "session.thinking.appended": ContextAppendedEvent,
    "session.commentary.appended": ContextAppendedEvent,
    "response.event": ResponseEventEnvelope,
}


def parse_server_event(message: str | bytes) -> ServerEvent:
    """Parse a server event from its JSON text.

    Event types this module doesn't model become :class:`UnknownServerEvent`.

    Args:
        message: The JSON text of one server event, as ``str`` or UTF-8 ``bytes``.

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
    model = _server_event_types.get(event_type, UnknownServerEvent)
    try:
        return model.model_validate(data)
    except Exception as e:
        raise ValueError(f"Invalid {event_type} server event: {e}") from e
