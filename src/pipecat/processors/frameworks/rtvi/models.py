#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI protocol v1 message models.

Contains all RTVI protocol v1 message definitions and data structures.
Import this module under the ``RTVI`` alias to use as a namespace::

    import pipecat.processors.frameworks.rtvi.models as RTVI

    msg = RTVI.BotReady(id="1", data=RTVI.BotReadyData(version=RTVI.PROTOCOL_VERSION))
"""

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
)

from pydantic import BaseModel

from pipecat.frames.frames import (
    AggregationType,
)

# -- Constants --
PROTOCOL_VERSION = "1.3.0"

MESSAGE_LABEL = "rtvi-ai"
MessageLiteral = Literal["rtvi-ai"]

# -- Base Message Structure --


class Message(BaseModel):
    """Base RTVI message structure.

    Represents the standard format for RTVI protocol messages.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: str
    id: str
    data: dict[str, Any] | None = None


# -- Client -> Pipecat messages.


class RawClientMessageData(BaseModel):
    """Data structure expected from client messages sent to the RTVI server."""

    t: str
    d: Any | None = None


class ClientMessage(BaseModel):
    """Cleansed data structure for client messages for handling."""

    msg_id: str
    type: str
    data: Any | None = None


class RawServerResponseData(BaseModel):
    """Data structure for server responses to client messages."""

    t: str
    d: Any | None = None


class ServerResponse(BaseModel):
    """The RTVI-formatted message response from the server to the client.

    This message is used to respond to custom messages sent by the client.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["server-response"] = "server-response"
    id: str
    data: RawServerResponseData


class AboutClientData(BaseModel):
    """Data about the RTVI client.

    Contains information about the client, including which RTVI library it
    is using, what platform it is on and any additional details, if available.
    """

    library: str
    library_version: str | None = None
    platform: str | None = None
    platform_version: str | None = None
    platform_details: Any | None = None


class ClientReadyData(BaseModel):
    """Data format of client ready messages.

    Contains the RTVI protocol version and client information.
    """

    version: str
    about: AboutClientData


# -- Pipecat -> Client errors


class ErrorResponseData(BaseModel):
    """Data for an RTVI error response.

    Contains the error message to send back to the client.
    """

    error: str


class ErrorResponse(BaseModel):
    """RTVI error response message.

    RTVI formatted error response message for relaying failed client requests.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["error-response"] = "error-response"
    id: str
    data: ErrorResponseData


class ErrorData(BaseModel):
    """Data for an RTVI error event.

    Contains error information including whether it's fatal.
    """

    error: str
    fatal: bool  # Indicates the pipeline has stopped due to this error


class Error(BaseModel):
    """RTVI error event message.

    RTVI formatted error message for relaying errors in the pipeline.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["error"] = "error"
    data: ErrorData


# -- Pipecat -> Client responses and messages.


class BotReadyData(BaseModel):
    """Data for bot ready notification.

    Contains protocol version and initial configuration.
    """

    version: str
    about: Mapping[str, Any] | None = None


class BotReady(BaseModel):
    """Message indicating bot is ready for interaction.

    Sent after bot initialization is complete.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-ready"] = "bot-ready"
    id: str
    data: BotReadyData


class LLMFunctionCallMessageData(BaseModel):
    """Data for LLM function call notification.

    Contains function call details including name, ID, and arguments.

    .. deprecated:: 0.0.102
        Use ``LLMFunctionCallInProgressMessageData`` instead.
    """

    function_name: str
    tool_call_id: str
    args: Mapping[str, Any]


class LLMFunctionCallMessage(BaseModel):
    """Message notifying of an LLM function call.

    Sent when the LLM makes a function call.

    .. deprecated:: 0.0.102
        Use ``LLMFunctionCallInProgressMessage`` with the
        ``llm-function-call-in-progress`` event type instead.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["llm-function-call"] = "llm-function-call"
    data: LLMFunctionCallMessageData


class SendTextOptions(BaseModel):
    """Options for sending text input to the LLM.

    Contains options for how the pipeline should process the text input.
    """

    run_immediately: bool = True
    audio_response: bool = True


class SendTextData(BaseModel):
    """Data format for sending text input to the LLM.

    Contains the text content to send and any options for how the pipeline should process it.
    """

    content: str
    options: SendTextOptions | None = None


class LLMFunctionCallStartMessageData(BaseModel):
    """Data for LLM function call start notification.

    Contains the function name being called. Fields may be omitted based on
    the configured function_call_report_level for security.
    """

    function_name: str | None = None


class LLMFunctionCallStartMessage(BaseModel):
    """Message notifying that an LLM function call has started.

    Sent when the LLM begins a function call.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["llm-function-call-started"] = "llm-function-call-started"
    data: LLMFunctionCallStartMessageData


class LLMFunctionCallResultData(BaseModel):
    """Data for LLM function call result.

    Contains function call details and result.
    """

    function_name: str
    tool_call_id: str
    arguments: dict
    result: dict | str


class LLMFunctionCallInProgressMessageData(BaseModel):
    """Data for LLM function call in-progress notification.

    Contains function call details including name, ID, and arguments.
    Fields may be omitted based on the configured function_call_report_level for security.
    """

    tool_call_id: str
    function_name: str | None = None
    arguments: Mapping[str, Any] | None = None


class LLMFunctionCallInProgressMessage(BaseModel):
    """Message notifying that an LLM function call is in progress.

    Sent when the LLM function call execution begins.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["llm-function-call-in-progress"] = "llm-function-call-in-progress"
    data: LLMFunctionCallInProgressMessageData


class LLMFunctionCallStoppedMessageData(BaseModel):
    """Data for LLM function call stopped notification.

    Contains details about the function call that stopped, including
    whether it was cancelled or completed with a result.
    Fields may be omitted based on the configured function_call_report_level for security.
    """

    tool_call_id: str
    cancelled: bool
    function_name: str | None = None
    result: Any | None = None


class LLMFunctionCallStoppedMessage(BaseModel):
    """Message notifying that an LLM function call has stopped.

    Sent when a function call completes (with result) or is cancelled.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["llm-function-call-stopped"] = "llm-function-call-stopped"
    data: LLMFunctionCallStoppedMessageData


class BotLLMStartedMessage(BaseModel):
    """Message indicating bot LLM processing has started."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-llm-started"] = "bot-llm-started"


class BotLLMStoppedMessage(BaseModel):
    """Message indicating bot LLM processing has stopped."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-llm-stopped"] = "bot-llm-stopped"


class BotTTSStartedMessage(BaseModel):
    """Message indicating bot TTS processing has started."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-tts-started"] = "bot-tts-started"


class BotTTSStoppedMessage(BaseModel):
    """Message indicating bot TTS processing has stopped."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-tts-stopped"] = "bot-tts-stopped"


class TextMessageData(BaseModel):
    """Data for text-based RTVI messages.

    Contains text content.
    """

    text: str


class BotOutputMessageData(TextMessageData):
    """Data for bot output RTVI messages.

    Extends TextMessageData to include metadata about the output.
    """

    spoken: bool = False  # Indicates if the text has been spoken by TTS
    aggregated_by: AggregationType | str
    # Indicates what form the text is in (e.g., by word, sentence, etc.)


class BotOutputMessage(BaseModel):
    """Message containing bot output text.

    An event meant to holistically represent what the bot is outputting,
    along with metadata about the output and if it has been spoken.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-output"] = "bot-output"
    data: BotOutputMessageData


class BotTranscriptionMessage(BaseModel):
    """Message containing bot transcription text.

    Sent when the bot's speech is transcribed.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-transcription"] = "bot-transcription"
    data: TextMessageData


class BotLLMTextMessage(BaseModel):
    """Message containing bot LLM text output.

    Sent when the bot's LLM generates text.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-llm-text"] = "bot-llm-text"
    data: TextMessageData


class BotTTSTextMessage(BaseModel):
    """Message containing bot TTS text output.

    Sent when text is being processed by TTS.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-tts-text"] = "bot-tts-text"
    data: TextMessageData


class AudioMessageData(BaseModel):
    """Data for audio-based RTVI messages.

    Contains audio data and metadata.
    """

    audio: str
    sample_rate: int
    num_channels: int


class BotTTSAudioMessage(BaseModel):
    """Message containing bot TTS audio output.

    Sent when the bot's TTS generates audio.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-tts-audio"] = "bot-tts-audio"
    data: AudioMessageData


class UserTranscriptionMessageData(BaseModel):
    """Data for user transcription messages.

    Contains transcription text and metadata.
    """

    text: str
    user_id: str
    timestamp: str
    final: bool


class UserTranscriptionMessage(BaseModel):
    """Message containing user transcription.

    Sent when user speech is transcribed.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-transcription"] = "user-transcription"
    data: UserTranscriptionMessageData


class UserLLMTextMessage(BaseModel):
    """Message containing user text input for LLM.

    Sent when user text is processed by the LLM.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-llm-text"] = "user-llm-text"
    data: TextMessageData


class UserStartedSpeakingMessage(BaseModel):
    """Message indicating user has started speaking."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-started-speaking"] = "user-started-speaking"


class UserStoppedSpeakingMessage(BaseModel):
    """Message indicating user has stopped speaking."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-stopped-speaking"] = "user-stopped-speaking"


class UserMuteStartedMessage(BaseModel):
    """Message indicating user has been muted."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-mute-started"] = "user-mute-started"


class UserMuteStoppedMessage(BaseModel):
    """Message indicating user has been unmuted."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-mute-stopped"] = "user-mute-stopped"


class BotStartedSpeakingMessage(BaseModel):
    """Message indicating bot has started speaking."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-started-speaking"] = "bot-started-speaking"


class BotStoppedSpeakingMessage(BaseModel):
    """Message indicating bot has stopped speaking."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-stopped-speaking"] = "bot-stopped-speaking"


class MetricsMessage(BaseModel):
    """Message containing performance metrics.

    Sent to provide performance and usage metrics.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["metrics"] = "metrics"
    data: Mapping[str, Any]


class ServerMessage(BaseModel):
    """Generic server message.

    Used for custom server-to-client messages.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["server-message"] = "server-message"
    data: Any


class AudioLevelMessageData(BaseModel):
    """Data format for sending audio levels."""

    value: float


class UserAudioLevelMessage(BaseModel):
    """Message indicating user audio level."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["user-audio-level"] = "user-audio-level"
    data: AudioLevelMessageData


class BotAudioLevelMessage(BaseModel):
    """Message indicating bot audio level."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["bot-audio-level"] = "bot-audio-level"
    data: AudioLevelMessageData


class SystemLogMessage(BaseModel):
    """Message including a system log."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["system-log"] = "system-log"
    data: TextMessageData


# -- UI Agent Protocol -------------------------------------------------------
#
# A structured RTVI message vocabulary that lets server-side AI agents
# observe and drive a GUI app on the client side. The protocol covers
# five first-class RTVI message types:
#
#   ui-event         client-to-server event message
#   ui-command       server-to-client command message
#   ui-snapshot      client-to-server accessibility snapshot
#   ui-cancel-task   client-to-server cancellation request
#   ui-task          server-to-client task lifecycle envelope
#
# This section is data only (constants and payload models, no
# behavior). Higher-level frameworks like ``pipecat-ai-subagents``
# build the agent abstractions on top, and single-LLM Pipecat apps can
# target the same wire format directly via custom tools that emit
# typed RTVI messages with these types. The matching client-side
# implementation lives in ``@pipecat-ai/client-js`` and
# ``@pipecat-ai/client-react``.

# The wire-format ``type`` strings (``"ui-event"``, ``"ui-command"``,
# ``"ui-snapshot"``, ``"ui-cancel-task"``, ``"ui-task"``) are pinned
# as ``Literal[...]`` field defaults on the corresponding ``*Message``
# pydantic class below, matching the convention used for every other
# RTVI message type in this module.

# Each ``ui-task`` envelope carries a ``kind`` field that the client's
# task reducer dispatches on. The four kinds form the lifecycle of a
# user-facing task group:
#
#   group_started → task_update* → task_completed × N → group_completed
#
# where N is the number of workers in the group. The kind strings are
# pinned as ``Literal[...]`` defaults on the matching ``UITask*Data``
# class below.


# -- UI envelope data classes --


class UIEventData(BaseModel):
    """Inner ``data`` for a ``ui-event`` message.

    Parameters:
        name: App-defined event name.
        payload: App-defined payload, schemaless by design.
    """

    name: str
    payload: Any | None = None


class UICommandData(BaseModel):
    """Inner ``data`` for a ``ui-command`` message.

    Parameters:
        name: App-defined command name.
        payload: App-defined payload (already a plain dict by the
            time it lands on the wire). The standard payload models
            below produce the right shape via ``model_dump()``.
    """

    name: str
    payload: Any | None = None


class UISnapshotData(BaseModel):
    """Inner ``data`` for a ``ui-snapshot`` message.

    The accessibility snapshot tree is opaque on the server side.
    The client owns its shape; the server stores it as-is for
    rendering into the LLM context.

    Parameters:
        tree: The serialized accessibility tree.
    """

    tree: Any | None = None


class UICancelTaskData(BaseModel):
    """Inner ``data`` for a ``ui-cancel-task`` message.

    Parameters:
        task_id: The task group id the client wants cancelled.
        reason: Optional human-readable reason.
    """

    task_id: str
    reason: str | None = None


class UITaskGroupStartedData(BaseModel):
    """``data`` for a ``ui-task`` envelope with kind ``group_started``.

    Parameters:
        kind: Always ``"group_started"``.
        task_id: Shared task identifier for the group.
        agents: Names of the agents the work was dispatched to.
        label: Optional human-readable label for the group.
        cancellable: Whether the client may request cancellation.
        at: Epoch milliseconds when the group started.
    """

    kind: Literal["group_started"] = "group_started"
    task_id: str
    agents: list[str] | None = None
    label: str | None = None
    cancellable: bool = True
    at: int = 0


class UITaskUpdateData(BaseModel):
    """``data`` for a ``ui-task`` envelope with kind ``task_update``.

    Parameters:
        kind: Always ``"task_update"``.
        task_id: The shared task identifier.
        agent_name: The worker that produced the update.
        data: The worker's update payload, forwarded verbatim.
        at: Epoch milliseconds when the update was emitted.
    """

    kind: Literal["task_update"] = "task_update"
    task_id: str
    agent_name: str
    data: Any | None = None
    at: int = 0


class UITaskCompletedData(BaseModel):
    """``data`` for a ``ui-task`` envelope with kind ``task_completed``.

    Parameters:
        kind: Always ``"task_completed"``.
        task_id: The shared task identifier.
        agent_name: The worker that produced the response.
        status: Completion status string.
        response: The worker's response payload.
        at: Epoch milliseconds when the response was received.
    """

    kind: Literal["task_completed"] = "task_completed"
    task_id: str
    agent_name: str
    status: str
    response: Any | None = None
    at: int = 0


class UITaskGroupCompletedData(BaseModel):
    """``data`` for a ``ui-task`` envelope with kind ``group_completed``.

    Parameters:
        kind: Always ``"group_completed"``.
        task_id: The shared task identifier.
        at: Epoch milliseconds when the group completed.
    """

    kind: Literal["group_completed"] = "group_completed"
    task_id: str
    at: int = 0


#: Discriminated union over the four task-lifecycle data shapes,
#: keyed by the ``kind`` field.
UITaskData = (
    UITaskGroupStartedData | UITaskUpdateData | UITaskCompletedData | UITaskGroupCompletedData
)


# -- UI envelope message classes --


class UIEventMessage(BaseModel):
    """RTVI ``ui-event`` message (client → server)."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["ui-event"] = "ui-event"
    id: str
    data: UIEventData


class UICommandMessage(BaseModel):
    """RTVI ``ui-command`` message (server → client)."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["ui-command"] = "ui-command"
    data: UICommandData


class UISnapshotMessage(BaseModel):
    """RTVI ``ui-snapshot`` message (client → server)."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["ui-snapshot"] = "ui-snapshot"
    id: str
    data: UISnapshotData


class UICancelTaskMessage(BaseModel):
    """RTVI ``ui-cancel-task`` message (client → server)."""

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["ui-cancel-task"] = "ui-cancel-task"
    id: str
    data: UICancelTaskData


class UITaskMessage(BaseModel):
    """RTVI ``ui-task`` message (server → client).

    The ``data`` field is one of the four task-lifecycle
    discriminated by the ``kind`` field.
    """

    label: MessageLiteral = MESSAGE_LABEL
    type: Literal["ui-task"] = "ui-task"
    data: UITaskData


# -- UI command payloads --
#
# These models describe commands that have matching default React
# handlers in ``@pipecat-ai/client-react``'s ``standardHandlers``.
# Apps can use them as-is, override the client handler to customize
# rendering, or ignore them entirely and define their own command
# names.
#
# Server-side helpers that send commands accept these models directly.
# ``BaseModel.model_dump()`` converts them to the plain-dict shape
# that travels over the wire.


class Toast(BaseModel):
    """A transient notification surface shown on the client.

    Parameters:
        title: Required headline.
        subtitle: Optional second line beneath the title.
        description: Optional body text.
        image_url: Optional leading image.
        duration_ms: Optional dismiss timer. Client default applies
            when None.
    """

    title: str
    subtitle: str | None = None
    description: str | None = None
    image_url: str | None = None
    duration_ms: int | None = None


class Navigate(BaseModel):
    """Client-side navigation to a named view.

    Parameters:
        view: App-defined view name (route, screen id, tab key, etc.).
        params: Optional view-specific parameters.
    """

    view: str
    params: dict | None = None


class ScrollTo(BaseModel):
    """Scroll a target element into view.

    The client resolves the target by ``ref`` first (a snapshot ref
    like ``"e42"`` assigned by the a11y walker), then falls back to
    ``target_id`` (``document.getElementById``). Supply whichever you
    have; ``ref`` is the normal choice when acting on a node from
    ``<ui_state>``.

    Parameters:
        ref: Snapshot ref from ``<ui_state>``.
        target_id: Element id registered on the client.
        behavior: Optional scroll behavior hint. Typical values:
            ``"smooth"`` or ``"instant"``. Clients may ignore.
    """

    ref: str | None = None
    target_id: str | None = None
    behavior: str | None = None


class Highlight(BaseModel):
    """Briefly emphasize a target element (flash, glow, pulse).

    Parameters:
        ref: Snapshot ref from ``<ui_state>``.
        target_id: Element id registered on the client.
        duration_ms: Optional highlight duration. Client default
            applies when None.
    """

    ref: str | None = None
    target_id: str | None = None
    duration_ms: int | None = None


class Focus(BaseModel):
    """Move input focus to a target element.

    Parameters:
        ref: Snapshot ref from ``<ui_state>``.
        target_id: Element id registered on the client.
    """

    ref: str | None = None
    target_id: str | None = None


class Click(BaseModel):
    """Click an element on the client.

    Closes the form-fill loop for non-text inputs (checkboxes, radios)
    and exposes the rest of the action vocabulary (submit buttons,
    links, app-specific clickable nodes). The standard handler
    silently no-ops on ``disabled`` targets so the agent can't bypass
    UI affordances the user is meant to control.

    For native ``<select>``, prefer ``SetInputValue`` (clicking
    options doesn't reliably change the selection); for custom
    comboboxes (ARIA listbox + popup), apps wire their own command
    matching the library's interaction model.

    Parameters:
        ref: Snapshot ref from ``<ui_state>``.
        target_id: Element id registered on the client. Used as a
            fallback when ``ref`` is not set or has gone stale.
    """

    ref: str | None = None
    target_id: str | None = None


class SetInputValue(BaseModel):
    """Write a value into a text input or textarea on the client.

    Use this for form-filling: the agent has decided what should go
    into a field (clarifying answer, tax form entry, etc.) and asks
    the client to populate it. With ``replace=True`` (the default),
    the existing value is overwritten; with ``replace=False`` the
    value is appended.

    The standard handler silently no-ops on ``disabled``, ``readonly``,
    and ``<input type="hidden">`` targets so the agent can't write
    into fields the user can't.

    Parameters:
        value: The text to write.
        ref: Snapshot ref from ``<ui_state>``. Typically the ref of
            an ``<input>`` or ``<textarea>``.
        target_id: Element id registered on the client. Used as a
            fallback when ``ref`` is not set or has gone stale.
        replace: When True (the default), overwrite the current
            value. When False, append to it.
    """

    value: str = ""
    ref: str | None = None
    target_id: str | None = None
    replace: bool = True


class SelectText(BaseModel):
    """Select text on the page so the user can see what the agent means.

    Mirror of the ``selection`` field surfaced in the snapshot. Use
    this to point the user's attention at a specific paragraph or
    range after the agent has decided what it's referring to.

    With ``start_offset`` and ``end_offset`` omitted, the entire
    target's text content is selected (``Range.selectNodeContents``
    for document elements; ``el.select()`` for ``<input>`` /
    ``<textarea>``).

    Parameters:
        ref: Snapshot ref from ``<ui_state>``. Typically the ref of
            a paragraph or input element.
        target_id: Element id registered on the client. Used as a
            fallback when ``ref`` is not set or has gone stale.
        start_offset: Character offset within the target's text
            where the selection should start. For ``<input>`` and
            ``<textarea>`` this is the value offset; for document
            elements it is computed against the concatenation of
            descendant text nodes in document order.
        end_offset: End character offset, exclusive. Same coordinate
            system as ``start_offset``.
    """

    ref: str | None = None
    target_id: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
