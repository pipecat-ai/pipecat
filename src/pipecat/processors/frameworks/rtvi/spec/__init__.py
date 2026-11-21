#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Machine-readable specification of the RTVI protocol.

:data:`MESSAGES` lists every message a Pipecat server accepts from an RTVI
client and every message it sends back, with the direction, the payload type,
an example and what makes the server send it, and :data:`BEHAVIORS` the rules
the server follows across messages. :func:`build_spec` turns those tables into
an `AsyncAPI 3.0 <https://www.asyncapi.com/docs/reference/specification/v3.0.0>`_
document, written from the server's point of view: it receives the client's
messages and sends its own.

Each protocol version's document is kept in the repository as
``<version>/rtvi.asyncapi.json`` next to this module, so client SDKs can
generate their protocol code from it at a pinned version. The current
version's document is the one :func:`build_spec` builds, and
``tests/test_rtvi_spec.py`` fails when it drifts from the models; rerun the
export script to refresh it::

    uv run python scripts/rtvi/write_rtvi_spec.py

``README.md`` next to this module describes the documents and how to change
the protocol.

``released.json`` records each protocol version a Pipecat release has shipped,
with the hash of its wire format, which never changes after the release.
"""

import hashlib
import inspect
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, Union, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode, JsonSchemaValue
from pydantic_core import core_schema

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.processors.frameworks.rtvi.observer import RTVIObserverParams

RTVIMessageDirection = Literal["client-to-server", "server-to-client"]

_OBSERVER_DEFAULTS = {field.name: field.default for field in fields(RTVIObserverParams)}
"""Each observer setting's default, for whether a message it turns on is on by default."""

_NO_DATA = object()
"""Marks a message that carries no ``data`` field."""


@dataclass(frozen=True)
class RTVIMessageSpec:
    """One RTVI message type, as the specification describes it.

    Parameters:
        type: The message's ``type`` on the wire, e.g. ``"bot-ready"``.
        direction: Who sends it.
        description: What the message means.
        data: The type of the message's ``data``, or ``None`` when it carries none.
        data_required: Whether a message of this type must carry ``data``.
        has_id: Whether the message carries an ``id``. A response repeats the
            ``id`` of the client message it answers.
        example: An example ``data``, or ``None`` when the message carries none.
        enabled_by: The :class:`~pipecat.processors.frameworks.rtvi.RTVIObserverParams`
            field that turns the message on, for a message the server only sends
            when it is set.
        min_client_version: The lowest client protocol version the server sends
            the message to once the client has sent ``client-ready``.
        deprecated: What replaces a deprecated message.
        details: More about the message, for a reader who needs it beyond the
            description, as Markdown.
        data_description: What the message's ``data`` holds, for data whose type
            has no description of its own.
    """

    type: str
    direction: RTVIMessageDirection
    description: str
    data: Any
    data_required: bool
    has_id: bool
    example: Any
    enabled_by: str | None = None
    min_client_version: str | None = None
    deprecated: str | None = None
    details: str | None = None
    data_description: str | None = None


_ROLE = re.compile(r":(?:class|meth|attr|func|mod|data|obj):`~?([^`]+)`")
_JSON_VALUES = {"`None`": "`null`", "`True`": "`true`", "`False`": "`false`"}
# A deprecation in a field's entry, which ends it: its first sentence names the
# replacement, and the rest is about Pipecat's releases, not the protocol.
_DEPRECATED = re.compile(r"\.\. deprecated::[^\n]*\n\s*([^.]*\.).*", re.S)
_DEPRECATION = ".. deprecated::"


def _markdown(text: str) -> str:
    """Docstring text as one line of Markdown, as a reader of the document needs it.

    Sphinx roles and markup become code, Python's values JSON's, and a
    deprecation its replacement.
    """
    text = _DEPRECATED.sub(lambda match: f"Deprecated: {match.group(1)}", text)
    text = _ROLE.sub(lambda match: f"`{match.group(1).rsplit('.', 1)[-1]}`", text)
    text = text.replace("``", "`")
    for python, json_value in _JSON_VALUES.items():
        text = text.replace(python, json_value)
    return " ".join(text.split())


def _docstring(cls: type) -> tuple[str, dict[str, str]]:
    """A class's summary and its fields' descriptions, from its Google-style docstring.

    A field a base class declares keeps the description from the base's docstring,
    unless the class describes it again.
    """
    fields = {name: _markdown(text) for name, text in _entries(cls).items()}
    return _summary(_own_doc(cls)), fields


def _deprecations(cls: type) -> tuple[bool, set[str]]:
    """Whether a class is deprecated, and which of its fields are."""
    deprecated = _DEPRECATION in _own_doc(cls).split("\nParameters:")[0]
    fields = {name for name, text in _entries(cls).items() if _DEPRECATION in text}
    return deprecated, fields


def _entries(cls: type) -> dict[str, str]:
    """The entries of a class's ``Parameters:`` section and its bases', by field."""
    entries = {}
    for klass in reversed(cls.__mro__):
        entries.update(_parameters(_own_doc(klass)))
    return entries


def _own_doc(cls: type) -> str:
    # A class without a docstring of its own would get its base's from getdoc.
    return inspect.cleandoc(cls.__dict__.get("__doc__") or "")


def _parameters(doc: str) -> dict[str, str]:
    """The field entries in a docstring's ``Parameters:`` section, as written."""
    fields = {}
    section = re.search(r"^Parameters:\n(.*?)(?=^\S|\Z)", doc, re.M | re.S)
    if section:
        for entry in re.split(r"^    (?=\S)", section.group(1), flags=re.M):
            name, _, text = entry.partition(":")
            if name.strip():
                fields[name.strip()] = text
    return fields


def _summary(doc: str) -> str:
    """The first paragraph of a docstring, as Markdown."""
    return _markdown(doc.split("\n\n")[0])


def _client(
    type: str,
    description: str,
    data: Any = _NO_DATA,
    example: Any = None,
    details: str | None = None,
) -> RTVIMessageSpec:
    """Describe a message the client sends. Every client message carries an ``id``."""
    has_data = data is not _NO_DATA
    return RTVIMessageSpec(
        type=type,
        direction="client-to-server",
        description=description,
        data=data if has_data else None,
        data_required=has_data,
        has_id=True,
        example=example,
        details=details,
    )


def _server(
    model: type[BaseModel],
    description: str,
    example: Any = None,
    data_type: Any = None,
    **kwargs,
) -> RTVIMessageSpec:
    """Describe a message the server sends, from the model it sends it with.

    ``data_type`` describes the message's ``data`` more precisely than the
    model does, for data the model leaves open but the server sends in a
    known shape.
    """
    fields = model.model_fields
    data = fields.get("data")
    return RTVIMessageSpec(
        type=fields["type"].default,
        direction="server-to-client",
        description=description,
        data=data_type or (data.annotation if data else None),
        data_required=bool(data and data.is_required()),
        has_id="id" in fields,
        example=example,
        data_description=_docstring(model)[1].get("data"),
        **kwargs,
    )


def _reserve_standard_commands(schema: dict[str, Any]) -> None:
    """Keep the standard commands' names out of an application command's schema."""
    command = cast(dict[str, Any], cast(dict[str, Any], schema["properties"])["command"])
    command["not"] = {"enum": list(RTVI.STANDARD_UI_COMMANDS)}


class UIAppCommand(RTVI.UICommandData):
    """A command of the application's own, with a name and payload it defines.

    Parameters:
        command: The command's name, other than a standard command's.
        payload: The command's payload, a JSON object.
    """

    # The standard commands' names are theirs, with their payloads.
    model_config = ConfigDict(json_schema_extra=_reserve_standard_commands)


def _ui_command_data() -> Any:
    """A ui-command's data: one of the standard commands, or one of the application's own."""
    commands = []
    for name, payload in RTVI.STANDARD_UI_COMMANDS.items():
        summary = " ".join((inspect.getdoc(payload) or "").split("\n\n")[0].split())
        command_type: Any = Literal[name]  # pyright: ignore[reportInvalidTypeForm]
        commands.append(
            create_model(
                "UI" + "".join(part.title() for part in name.split("_")) + "Command",
                __doc__=(
                    f"The standard `{name}` command: {summary[0].lower()}{summary[1:]}\n\n"
                    "Parameters:\n"
                    "    command: The command's name.\n"
                    "    payload: The command's payload."
                ),
                command=(command_type, ...),
                payload=(payload, ...),
            )
        )
    return Union[(*commands, UIAppCommand)]  # pyright: ignore[reportInvalidTypeArguments]


_TIMESTAMP = "2026-01-01T12:00:00.000Z"

_FUNCTION_CALL_REPORT_LEVELS = (
    "How much it carries depends on the server's report level for the function: "
    "`none` sends only the call's id, `name` adds the function's name, `arguments` its "
    "arguments, and `full` its result. At `disabled`, the server sends no messages about "
    "the call."
)

_FUNCTION_CALL_STARTED_REPORT_LEVELS = (
    "The server's report level for the function decides whether it carries the "
    "function's name: `name`, `arguments` and `full` do, and `none` doesn't. At "
    "`disabled`, the server sends no messages about the call."
)

MESSAGES: tuple[RTVIMessageSpec, ...] = (
    # -- Client to server --
    _client(
        "client-ready",
        "Tells the server the client is ready, with the protocol version it speaks. "
        "The server answers with bot-ready.",
        RTVI.ClientReadyData,
        {
            "version": RTVI.PROTOCOL_VERSION,
            "about": {"library": "@pipecat-ai/client-js", "library_version": "1.16.0"},
        },
        details="Typically sent once the transport has connected.",
    ),
    _client(
        "disconnect-bot",
        "Asks the server to end the session.",
        details=(
            "Disconnecting from the transport ends the session too. This message is for a "
            "client that wants to stay connected to the transport but end its conversation "
            "with the bot."
        ),
    ),
    _client(
        "client-message",
        "A custom message for the application. The server answers with server-response "
        "or error-response.",
        RTVI.RawClientMessageData,
        {"t": "get-weather", "d": {"city": "Lisbon"}},
    ),
    _client(
        "send-text",
        "Text the user typed, sent to the bot as if the user had said it.",
        RTVI.SendTextData,
        {"content": "What's the weather like?", "options": {"run_immediately": True}},
    ),
    _client(
        "send-file",
        "A file for the bot, with text that goes with it.",
        RTVI.SendFileData,
        {
            "content": "What is in this picture?",
            "file": {
                "format": "image/png",
                "name": "photo.png",
                "source": {"type": "url", "url": "https://example.com/photo.png"},
            },
        },
    ),
    _client(
        "dtmf",
        "Keypad presses, delivered in order.",
        RTVI.DTMFInputData,
        {"buttons": ["1", "#"]},
    ),
    _client(
        "llm-function-call-result",
        "The result of a function call the client ran for the bot's LLM.",
        RTVI.LLMFunctionCallResultData,
        {
            "function_name": "get_weather",
            "tool_call_id": "call_123",
            "arguments": {"city": "Lisbon"},
            "result": {"temperature": 21},
        },
    ),
    _client(
        "raw-audio",
        "A chunk of the user's audio, for transports that don't carry media.",
        RTVI.RawAudioData,
        {"base64Audio": "AAAAAA==", "sampleRate": 16000, "numChannels": 1},
    ),
    _client(
        "raw-audio-batch",
        "Several chunks of the user's audio, for transports that don't carry media.",
        RTVI.RawAudioBatchData,
        {"base64AudioBatch": ["AAAAAA==", "AAAAAA=="], "sampleRate": 16000, "numChannels": 1},
    ),
    _client(
        "ui-event",
        "An event from the client's UI, for a UI worker.",
        RTVI.UIEventData,
        {"event": "button-clicked", "payload": {"id": "checkout"}},
    ),
    _client(
        "ui-snapshot",
        "An accessibility snapshot of the client's UI, for a UI worker.",
        RTVI.UISnapshotData,
        {
            "tree": {
                "root": {
                    "ref": "e1",
                    "role": "main",
                    "children": [{"ref": "e2", "role": "button"}],
                },
                "captured_at": 1767268800000,
            }
        },
        details="Sent whenever the client's UI changes, so the bot works from what is on screen.",
    ),
    _client(
        "ui-cancel-job-group",
        "Asks the server to cancel a job group it reported with ui-job-group.",
        RTVI.UICancelJobGroupData,
        {"job_id": "job_1", "reason": "User closed the panel"},
        details="The server cancels the group only if it reported the group as cancellable.",
    ),
    # -- Server to client: session --
    _server(
        RTVI.BotReady,
        "Answers client-ready once the bot is ready to talk, with the protocol version "
        "the server speaks to this client, information about the bot, and what it can do. "
        "Its id is the client-ready's id.",
        {
            "version": RTVI.PROTOCOL_VERSION,
            "about": {"library": "pipecat-ai", "library_version": "1.3.0"},
            "capabilities": {"audio_in": True, "audio_out": True, "video_in": False},
        },
        details=(
            "`about` is the bot's own information: Pipecat's library and version, unless the "
            "application replaces it. An application shouldn't put anything sensitive in it."
        ),
    ),
    _server(
        RTVI.Error,
        "Reports an error in the bot, such as an AI service that failed. When fatal is "
        "true, the bot can't go on and the session ends.",
        {"error": "The LLM service failed", "fatal": False},
    ),
    _server(
        RTVI.ErrorResponse,
        "Answers a client message the server couldn't handle, with what went wrong. Its "
        "id is the id of the message it answers.",
        {"error": "Unsupported type foo"},
    ),
    _server(
        RTVI.ServerResponse,
        "The bot's answer to a client-message. Its id is the client-message's id, and t "
        "is the client-message's t.",
        {"t": "get-weather", "d": {"temperature": 21}},
    ),
    _server(
        RTVI.ServerMessage,
        "A message from the bot's application to the client, with data of the application's own.",
        {"event": "order-updated", "status": "shipped"},
    ),
    _server(
        RTVI.MetricsMessage,
        "Performance metrics of the bot's services, such as how long each took to respond "
        "and how much it processed.",
        {"ttfb": [{"processor": "OpenAILLMService#0", "value": 0.42}]},
        data_type=RTVI.MetricsMessageData,
        enabled_by="metrics_enabled",
    ),
    _server(
        RTVI.SystemLogMessage,
        "A line of the server's log, for debugging.",
        {"text": "Pipeline started"},
        enabled_by="system_logs_enabled",
    ),
    # -- Server to client: speaking --
    _server(
        RTVI.UserStartedSpeakingMessage,
        "The user's turn started: the bot considers the user to be speaking.",
        enabled_by="user_speaking_enabled",
    ),
    _server(
        RTVI.UserStoppedSpeakingMessage,
        "The user's turn ended: the bot considers the user done speaking.",
        enabled_by="user_speaking_enabled",
    ),
    _server(
        RTVI.VADUserStartedSpeakingMessage,
        "Voice activity detection (VAD) heard the user start speaking. It reacts to any "
        "voice, sooner than user-started-speaking, which comes when the bot decides the "
        "user's turn started.",
        enabled_by="vad_user_speaking_enabled",
    ),
    _server(
        RTVI.VADUserStoppedSpeakingMessage,
        "Voice activity detection (VAD) heard the user stop speaking. It reacts to any "
        "pause, sooner than user-stopped-speaking, which comes when the bot decides the "
        "user's turn ended.",
        enabled_by="vad_user_speaking_enabled",
    ),
    _server(
        RTVI.UserMuteStartedMessage,
        "The bot stopped listening to the user, for example while it says something that "
        "must not be interrupted.",
        enabled_by="user_mute_enabled",
        details=(
            "The client should keep sending audio. It may show the user that their input "
            "isn't being processed."
        ),
    ),
    _server(
        RTVI.UserMuteStoppedMessage,
        "The bot listens to the user again.",
        enabled_by="user_mute_enabled",
        details="The client may show the user that their input is being processed again.",
    ),
    _server(
        RTVI.BotStartedSpeakingMessage,
        "The bot started speaking: its audio started playing to the user.",
        enabled_by="bot_speaking_enabled",
    ),
    _server(
        RTVI.BotStoppedSpeakingMessage,
        "The bot stopped speaking.",
        enabled_by="bot_speaking_enabled",
    ),
    _server(
        RTVI.BotInterruptedMessage,
        "The bot was interrupted and stopped what it was saying. A client should drop the "
        "bot's text that hadn't been spoken yet.",
        enabled_by="bot_speaking_enabled",
        details=(
            "The user speaking over the bot interrupts it, and so does a send-text with "
            "`run_immediately`."
        ),
    ),
    _server(
        RTVI.UserAudioLevelMessage,
        "How loud the user's audio is, from 0 (quiet) to 1 (loud), sent regularly while "
        "the user's audio comes in.",
        {"value": 0.3},
        enabled_by="user_audio_level_enabled",
        details=(
            "Sent every 0.15 seconds by default, the server's `audio_level_period_secs` setting."
        ),
    ),
    _server(
        RTVI.BotAudioLevelMessage,
        "How loud the bot's audio is, from 0 (quiet) to 1 (loud), sent regularly while "
        "the bot speaks.",
        {"value": 0.5},
        enabled_by="bot_audio_level_enabled",
        details=(
            "Sent every 0.15 seconds by default, the server's `audio_level_period_secs` setting."
        ),
    ),
    # -- Server to client: what the user said --
    _server(
        RTVI.UserInputMessage,
        "What the user said or typed: a transcription of their speech, text the client "
        "sent, or a short acknowledgment the user said while the bot talks.",
        {
            "text": "What's the weather like?",
            "input_type": "transcription",
            "timestamp": _TIMESTAMP,
            "final": True,
            "user_id": "user_1",
        },
        enabled_by="user_input_enabled",
        details=(
            "Sent for each transcription of the user's speech; for each short acknowledgment "
            "the user says while the bot talks, which neither interrupts the bot nor reaches "
            "its LLM; and for each send-text message, with the send-text's id in `msg_id`."
        ),
    ),
    _server(
        RTVI.UserLLMTextMessage,
        "The user's message as the bot's LLM receives it.",
        {"text": "What's the weather like?"},
        enabled_by="user_llm_enabled",
    ),
    _server(
        RTVI.STTRawTextMessage,
        "A transcription of the user's speech as the speech-to-text service produces it, "
        "including interim transcriptions that may still change.",
        {"text": "What's the", "user_id": "user_1", "timestamp": _TIMESTAMP, "final": False},
        enabled_by="stt_raw_text_enabled",
        min_client_version="2.2.0",
    ),
    _server(
        RTVI.UserTranscriptionMessage,
        "A transcription of the user's speech, including interim transcriptions that may "
        "still change.",
        {"text": "What's the", "user_id": "user_1", "timestamp": _TIMESTAMP, "final": False},
        enabled_by="user_transcription_enabled",
        deprecated="Use user-input or stt-raw-text instead.",
    ),
    # -- Server to client: what the bot says --
    _server(
        RTVI.BotOutputMessage,
        "Text the bot says or shows, a sentence or another unit at a time, with whether "
        "it will be spoken and how much of it has been spoken.",
        {
            "text": "It's sunny in Lisbon.",
            "output_type": "sentence",
            "segment_id": 1,
            "will_be_spoken": True,
            "spoken_status": "in-progress",
            "spoken_progress": {"accumulated_text": "It's sunny", "remaining_text": " in Lisbon."},
        },
        enabled_by="bot_output_enabled",
    ),
    _server(
        RTVI.BotLLMStartedMessage,
        "The bot's LLM started generating a response.",
        enabled_by="bot_llm_enabled",
    ),
    _server(
        RTVI.BotLLMStoppedMessage,
        "The bot's LLM finished generating a response.",
        enabled_by="bot_llm_enabled",
    ),
    _server(
        RTVI.BotTTSStartedMessage,
        "The bot's text-to-speech service started turning a response into speech.",
        enabled_by="bot_tts_enabled",
    ),
    _server(
        RTVI.BotTTSStoppedMessage,
        "The bot's text-to-speech service finished turning a response into speech.",
        enabled_by="bot_tts_enabled",
    ),
    _server(
        RTVI.LLMRawTextMessage,
        "Text exactly as the bot's LLM streams it, a chunk at a time.",
        {"text": "It's sunny"},
        enabled_by="llm_raw_text_enabled",
        min_client_version="2.2.0",
    ),
    _server(
        RTVI.TTSRawTextMessage,
        "Text the bot's text-to-speech service speaks, as it speaks it.",
        {"text": "It's"},
        enabled_by="tts_raw_text_enabled",
        min_client_version="2.2.0",
    ),
    _server(
        RTVI.BotLLMMarkerMessage,
        "The marker the bot's LLM put in a response, outside the text it speaks, such as "
        "one saying whether the user's turn was complete. Sent once per response, when it "
        "ends.",
        {"text": "✓", "kind": "complete", "raw": "✓ It's sunny in Lisbon.", "markers": ["✓"]},
        enabled_by="bot_llm_marker_enabled",
        details=(
            "It carries the raw text of the whole response, so it's meant for evaluation and "
            "debugging rather than for the user."
        ),
    ),
    _server(
        RTVI.BotTranscriptionMessage,
        "The text of the bot's LLM response, a whole sentence at a time.",
        {"text": "It's sunny in Lisbon."},
        enabled_by="bot_llm_enabled",
        details="It carries no timing. For text as the bot speaks it, use tts-raw-text.",
        deprecated="Use bot-output instead.",
    ),
    _server(
        RTVI.BotLLMTextMessage,
        "Text as the bot's LLM streams it, a chunk at a time.",
        {"text": "It's sunny"},
        enabled_by="bot_llm_enabled",
        deprecated="Use llm-raw-text instead.",
    ),
    _server(
        RTVI.BotTTSTextMessage,
        "Text the bot's text-to-speech service speaks, as it speaks it.",
        {"text": "It's"},
        enabled_by="bot_tts_enabled",
        deprecated="Use tts-raw-text instead.",
    ),
    # -- Server to client: function calls --
    _server(
        RTVI.LLMFunctionCallStartMessage,
        "The bot's LLM started calling a function. The server's settings decide whether "
        "it includes the function's name.",
        {"function_name": "get_weather"},
        details=_FUNCTION_CALL_STARTED_REPORT_LEVELS,
    ),
    _server(
        RTVI.LLMFunctionCallInProgressMessage,
        "A function call is running, with its id and, as the server's settings decide, "
        "the function's name and arguments.",
        {
            "tool_call_id": "call_123",
            "function_name": "get_weather",
            "arguments": {"city": "Lisbon"},
        },
        details=_FUNCTION_CALL_REPORT_LEVELS
        + " A client that runs the function itself answers with llm-function-call-result.",
    ),
    _server(
        RTVI.LLMFunctionCallStoppedMessage,
        "A function call finished or was cancelled, with its id and, as the server's "
        "settings decide, the function's name and result.",
        {"tool_call_id": "call_123", "cancelled": False, "function_name": "get_weather"},
        details=_FUNCTION_CALL_REPORT_LEVELS,
    ),
    _server(
        RTVI.LLMFunctionCallMessage,
        "The bot's LLM called a function, with its name, id and arguments.",
        {"function_name": "get_weather", "tool_call_id": "call_123", "args": {"city": "Lisbon"}},
        deprecated="Use llm-function-call-in-progress instead.",
    ),
    # -- Server to client: conversation flows --
    _server(
        RTVI.FlowNodeEnteredMessage,
        "The bot's conversation flow entered a node, so a client can follow the "
        "conversation through the flow.",
        {
            "node": "verify.check_birthday",
            "previous_node": "greet",
            "function": "wants_to_book",
            "metadata": {"step": 2},
            "parent": "verify",
        },
        details=(
            "Sent each time a Pipecat Flows conversation enters a node, including its first "
            "node and a node it enters again, before the bot responds in that node."
        ),
        min_client_version="2.4.0",
    ),
    # -- Server to client: UI worker --
    _server(
        RTVI.UICommandMessage,
        "A command from the bot to the client's UI, such as showing a notification.",
        {"command": "toast", "payload": {"title": "Order shipped"}},
        data_type=_ui_command_data(),
        details=(
            "The standard commands each have a payload of their own, and "
            "`@pipecat-ai/client-react` has a default handler for each. An application can "
            "also send commands of its own."
        ),
    ),
    _server(
        RTVI.UIJobGroupMessage,
        "Progress of work the bot handed to other workers, for the client's UI: the work "
        "started, a worker's update or result, or all of it completed.",
        {
            "kind": "group_started",
            "job_id": "job_1",
            "workers": ["researcher"],
            "label": "Researching",
            "cancellable": True,
            "at": 1767268800000,
        },
    ),
)
"""Every message of the RTVI protocol, in the order the specification lists them."""


@dataclass(frozen=True)
class RTVIMessageGroup:
    """A group of related messages, as the documentation shows them together.

    Groups only organize the documentation: they become the document's tags,
    which the generated reference uses for its sections. They aren't part of
    the protocol, so regrouping messages changes nothing on the wire and
    leaves a released version's contract hash as it is.

    Parameters:
        name: The group's name, e.g. ``"User Interface"``.
        description: What the group's messages are about.
        messages: The types of the group's messages, in the order to show them.
    """

    name: str
    description: str
    messages: tuple[str, ...]


GROUPS: tuple[RTVIMessageGroup, ...] = (
    RTVIMessageGroup(
        "Connection Management",
        "Starting and ending a session, and errors.",
        ("client-ready", "bot-ready", "disconnect-bot", "error", "system-log"),
    ),
    RTVIMessageGroup(
        "Speaking and Transcription",
        "Who is speaking, the user's audio, and what the user and the bot said.",
        (
            "user-started-speaking",
            "user-stopped-speaking",
            "vad-user-started-speaking",
            "vad-user-stopped-speaking",
            "bot-started-speaking",
            "bot-stopped-speaking",
            "bot-interrupted",
            "user-audio-level",
            "bot-audio-level",
            "user-mute-started",
            "user-mute-stopped",
            "raw-audio",
            "raw-audio-batch",
            "user-transcription",
            "user-input",
            "bot-output",
            "bot-transcription",
        ),
    ),
    RTVIMessageGroup(
        "Client-Server Messaging",
        "Messages of the application's own, and the answers to client messages.",
        ("server-message", "client-message", "server-response", "error-response"),
    ),
    RTVIMessageGroup(
        "Advanced LLM Interactions",
        "Text, files and keypresses for the bot, and the bot's function calls.",
        (
            "send-text",
            "send-file",
            "dtmf",
            "llm-function-call-started",
            "llm-function-call-in-progress",
            "llm-function-call-stopped",
            "llm-function-call",
            "llm-function-call-result",
        ),
    ),
    RTVIMessageGroup(
        "Service-Specific Insights",
        "What the bot's speech-to-text, LLM and text-to-speech services are doing.",
        (
            "stt-raw-text",
            "bot-llm-started",
            "bot-llm-stopped",
            "user-llm-text",
            "bot-llm-text",
            "llm-raw-text",
            "bot-llm-marker",
            "bot-tts-started",
            "bot-tts-stopped",
            "bot-tts-text",
            "tts-raw-text",
        ),
    ),
    RTVIMessageGroup(
        "Metrics and Monitoring",
        "How the bot's services perform.",
        ("metrics",),
    ),
    RTVIMessageGroup(
        "Conversation Flows",
        "Where the bot is in a conversation flow.",
        ("flow-node-entered",),
    ),
    RTVIMessageGroup(
        "User Interface",
        "The bot observing and driving the client's UI.",
        ("ui-snapshot", "ui-event", "ui-cancel-job-group", "ui-command", "ui-job-group"),
    ),
)
"""Every message in exactly one group, in the order the documentation shows them."""

_GROUP_OF = {type: group.name for group in GROUPS for type in group.messages}


@dataclass(frozen=True)
class RTVIBehaviorSpec:
    """A rule the server follows across messages, beyond the shape of each one.

    Parameters:
        id: The behavior's name, e.g. ``"bot-ready-answers-client-ready"``.
        description: What the server does.
        messages: The types of the messages the behavior involves.
    """

    id: str
    description: str
    messages: tuple[str, ...]


BEHAVIORS: tuple[RTVIBehaviorSpec, ...] = (
    RTVIBehaviorSpec(
        "bot-ready-answers-client-ready",
        "Once the bot is ready, the server answers client-ready with a bot-ready carrying "
        "the client-ready's id.",
        ("client-ready", "bot-ready"),
    ),
    RTVIBehaviorSpec(
        "bot-ready-version",
        "bot-ready's version is the protocol version the server speaks to the client: the "
        "server's own for a client of the same major version, and the client's own for a "
        "1.x client.",
        ("client-ready", "bot-ready"),
    ),
    RTVIBehaviorSpec(
        "current-version-before-client-ready",
        "Until client-ready, the server sends messages as it does to a client of its own "
        "protocol version, so what the bot does before the client says its version, such "
        "as a greeting, reaches the client. From client-ready on, the client's version "
        "decides which messages it gets.",
        ("client-ready",),
    ),
    RTVIBehaviorSpec(
        "client-ready-version-check",
        "A client-ready without a version, with a malformed one, or of a major version the "
        "server doesn't speak gets an error-response carrying its id. The session goes on, "
        "and the server still answers with bot-ready.",
        ("client-ready", "error-response", "bot-ready"),
    ),
    RTVIBehaviorSpec(
        "failed-request-gets-error-response",
        "A client message the server can't handle, of an unknown type or with data that "
        "doesn't match its payload, gets an error-response carrying the message's id.",
        ("error-response",),
    ),
    RTVIBehaviorSpec(
        "invalid-message-gets-error",
        'A transport message labeled "rtvi-ai" that isn\'t an RTVI message, such as one '
        "without an id, gets an error, since there is no id to answer. The server ignores "
        'a message without the "rtvi-ai" label.',
        ("error",),
    ),
    RTVIBehaviorSpec(
        "client-message-answer",
        "The bot answers a client-message with a server-response or an error-response "
        "carrying the client-message's id. A server-response's t is the client-message's t.",
        ("client-message", "server-response", "error-response"),
    ),
    RTVIBehaviorSpec(
        "disconnect-bot-ends-session",
        "disconnect-bot ends the bot's session.",
        ("disconnect-bot",),
    ),
    RTVIBehaviorSpec(
        "messages-turned-off",
        "The server sends a message that has x-rtvi-enabled-by only while the server "
        "setting it names is on. The settings are fields of Pipecat's RTVIObserverParams.",
        (),
    ),
    RTVIBehaviorSpec(
        "spoken-output-waits-for-speech",
        "bot-output for text the bot will speak waits for the bot to start speaking: the "
        "server sends it after bot-started-speaking, in the order the text came.",
        ("bot-started-speaking", "bot-output"),
    ),
    RTVIBehaviorSpec(
        "interruption-drops-unspoken-output",
        "On an interruption, the server sends bot-interrupted, and never sends the "
        "bot-output it was holding for speech that the interruption cut off.",
        ("bot-interrupted", "bot-output"),
    ),
)
"""The rules the server follows across messages, each checked by a conformance test."""


class _SchemaGenerator(GenerateJsonSchema):
    """Describes each model by its docstring's summary, and each field by its entry.

    A model's whole docstring is written for Python readers; the document gives a
    client's author the first paragraph, and a description for each field from the
    docstring's ``Parameters:`` section. An enum's description is trimmed after
    generation, since Pydantic sets it after this generator runs. A model or field
    the docstring deprecates is marked ``deprecated``.

    Fields have no titles, which would only repeat their names. A field that
    refers to another schema and has its own description wraps the reference in
    ``allOf``, since JSON Schema draft 7, which AsyncAPI's schemas follow, ignores
    the keywords next to a ``$ref``.

    The server's messages are described as the server sends them: it leaves out
    each field whose value is ``None``. So in them a field that can be ``None`` is
    optional and never ``null``, and every other field is always there.
    """

    def field_title_should_be_set(self, schema: Any) -> bool:
        return False

    def _describe(self, json_schema: JsonSchemaValue, cls: type):
        summary, fields = _docstring(cls)
        if summary:
            json_schema["description"] = summary
        else:
            json_schema.pop("description", None)
        deprecated, deprecated_fields = _deprecations(cls)
        if deprecated:
            json_schema["deprecated"] = True
        if self.mode == "serialization":
            _as_sent(json_schema)
        properties = json_schema.get("properties", {})
        for name, field in properties.items():
            if name in fields:
                field.setdefault("description", fields[name])
            if name in deprecated_fields:
                field["deprecated"] = True
            if "$ref" in field and len(field) > 1:
                properties[name] = {"allOf": [{"$ref": field.pop("$ref")}], **field}

    def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
        json_schema = super().model_schema(schema)
        self._describe(json_schema, schema["cls"])
        return json_schema

    def dataclass_schema(self, schema: core_schema.DataclassSchema) -> JsonSchemaValue:
        json_schema = super().dataclass_schema(schema)
        self._describe(json_schema, schema["cls"])
        return json_schema


_NULL = {"type": "null"}
# Keywords that constrain a value: a schema without any of them allows any value, None too.
_TYPED = frozenset({"type", "$ref", "anyOf", "allOf", "oneOf", "const", "enum"})


def _as_sent(json_schema: JsonSchemaValue):
    """An object's schema as the server sends it, leaving out each field that is ``None``."""
    properties = json_schema.get("properties", {})
    required = []
    for name, field in properties.items():
        options = field.get("anyOf", [])
        nullable = _NULL in options or field.get("default", _NO_DATA) is None
        if _NULL in options:
            rest = [option for option in options if option != _NULL]
            field = {key: value for key, value in field.items() if key != "anyOf"}
            field = {**rest[0], **field} if len(rest) == 1 else {"anyOf": rest, **field}
        if field.get("default", _NO_DATA) is None:
            del field["default"]
        properties[name] = field
        if not nullable and _TYPED & field.keys():
            required.append(name)
    json_schema.pop("required", None)
    if required:
        json_schema["required"] = required


def _message_key(spec: RTVIMessageSpec) -> str:
    """The message's name in the document's ``components``."""
    return spec.type.replace("-", "_")


def _envelope(spec: RTVIMessageSpec, data_schema: Any) -> dict:
    """The schema of a whole message: its envelope and its data."""
    properties = {
        "label": {"const": RTVI.MESSAGE_LABEL},
        "type": {"const": spec.type},
    }
    required = ["label", "type"]
    if spec.has_id:
        properties["id"] = {"type": "string"}
        required.append("id")
    if spec.data is not None:
        if spec.data_description and not {"$ref", "anyOf", "description"} & data_schema.keys():
            data_schema = {**data_schema, "description": spec.data_description}
        properties["data"] = data_schema
        if spec.data_required:
            required.append("data")
    return {"type": "object", "properties": properties, "required": required}


def _example(spec: RTVIMessageSpec) -> dict:
    """An example of a whole message."""
    message: dict[str, Any] = {"label": RTVI.MESSAGE_LABEL, "type": spec.type}
    if spec.has_id:
        message["id"] = "msg_1"
    if spec.example is not None:
        message["data"] = spec.example
    return message


def _notes(spec: RTVIMessageSpec) -> list[str]:
    """What the message's extension fields say, as sentences for its summary."""
    notes = []
    if spec.enabled_by:
        if _OBSERVER_DEFAULTS[spec.enabled_by]:
            notes.append(f"On by default; turned off by the server's `{spec.enabled_by}` setting.")
        else:
            notes.append(f"Off by default; turned on by the server's `{spec.enabled_by}` setting.")
    if spec.min_client_version:
        notes.append(
            f"Sent to clients of protocol {spec.min_client_version} or later, and to any "
            "client before its client-ready."
        )
    if spec.deprecated:
        notes.append(f"Deprecated. {spec.deprecated}")
    return notes


def _message(spec: RTVIMessageSpec, data_schema: Any) -> dict:
    """A message's entry in the document's ``components``."""
    message: dict[str, Any] = {
        "name": spec.type,
        "title": spec.type,
        "summary": " ".join([spec.description, *_notes(spec)]),
        "tags": [{"name": _GROUP_OF[spec.type]}],
        "payload": _envelope(spec, data_schema),
        "examples": [{"payload": _example(spec)}],
    }
    if spec.details:
        message["description"] = spec.details
    if spec.enabled_by:
        message["x-rtvi-enabled-by"] = spec.enabled_by
    if spec.min_client_version:
        message["x-rtvi-min-client-version"] = spec.min_client_version
    if spec.deprecated:
        message["x-rtvi-deprecated"] = spec.deprecated
    return message


def build_spec() -> dict:
    """Build the AsyncAPI document of the RTVI protocol.

    Returns:
        The document, as JSON-compatible data.
    """
    # What the server accepts is the client's messages as the models validate
    # them, and what it produces is its own as the models serialize them.
    inputs: list[tuple[str, JsonSchemaMode, TypeAdapter[Any]]] = [
        (
            spec.type,
            "validation" if spec.direction == "client-to-server" else "serialization",
            TypeAdapter(spec.data),
        )
        for spec in MESSAGES
        if spec.data is not None
    ]
    schemas, definitions = TypeAdapter.json_schemas(
        inputs, ref_template="#/components/schemas/{model}", schema_generator=_SchemaGenerator
    )
    data_schemas = {key: schema for (key, _), schema in schemas.items()}
    for schema in definitions.get("$defs", {}).values():
        if "description" in schema:
            schema["description"] = _summary(schema["description"])

    messages = {
        _message_key(spec): _message(spec, data_schemas.get(spec.type)) for spec in MESSAGES
    }

    def refs(direction: RTVIMessageDirection) -> list[dict]:
        return [
            {"$ref": f"#/channels/rtvi/messages/{_message_key(spec)}"}
            for spec in MESSAGES
            if spec.direction == direction
        ]

    return {
        "asyncapi": "3.0.0",
        "info": {
            "title": "RTVI",
            "version": RTVI.PROTOCOL_VERSION,
            "description": (
                "RTVI (Real-Time Voice Interface) is the protocol a client app and a Pipecat "
                "bot use to talk to each other during a session. This document describes it "
                "from the server's point of view: receiveClientMessages lists the messages a "
                "client sends, and sendServerMessages the messages the server sends.\n\n"
                f'Every message is a JSON object with the label "{RTVI.MESSAGE_LABEL}", a '
                "type, its data, and an id on every message a client sends and on the "
                "server's answers to them. The session's transport carries messages as data "
                "messages, next to the audio and video.\n\n"
                "A receiver ignores a field it doesn't know in a message's data, so a later "
                "protocol version can add fields without breaking an earlier client or "
                "server. A client also ignores a message of a type it doesn't know; the "
                "server answers one with an error-response.\n\n"
                "A message can also have x-rtvi-enabled-by, the server setting that turns it "
                "on (a field of Pipecat's RTVIObserverParams); x-rtvi-min-client-version, the "
                "lowest client protocol version the server sends it to; and "
                "x-rtvi-deprecated, what replaces a deprecated message. x-rtvi-behaviors lists "
                "the rules the server follows across messages."
            ),
            "tags": [{"name": group.name, "description": group.description} for group in GROUPS],
        },
        "defaultContentType": "application/json",
        "channels": {
            "rtvi": {
                "description": "The session's data channel, in both directions.",
                "messages": {key: {"$ref": f"#/components/messages/{key}"} for key in messages},
            }
        },
        "operations": {
            "receiveClientMessages": {
                "action": "receive",
                "summary": "Messages the client sends to the server.",
                "channel": {"$ref": "#/channels/rtvi"},
                "messages": refs("client-to-server"),
            },
            "sendServerMessages": {
                "action": "send",
                "summary": "Messages the server sends to the client.",
                "channel": {"$ref": "#/channels/rtvi"},
                "messages": refs("server-to-client"),
            },
        },
        "components": {
            "messages": messages,
            "schemas": definitions.get("$defs", {}),
        },
        "x-rtvi-behaviors": [
            {"id": b.id, "description": b.description, "messages": list(b.messages)}
            for b in BEHAVIORS
        ],
    }


def spec_path(version: str = RTVI.PROTOCOL_VERSION) -> Path:
    """The path of a protocol version's document.

    Args:
        version: The protocol version. Defaults to the current one.

    Returns:
        The path of the version's ``rtvi.asyncapi.json``.
    """
    return Path(__file__).parent / version / "rtvi.asyncapi.json"


def render_spec() -> str:
    """The content of the current protocol version's ``rtvi.asyncapi.json``."""
    return json.dumps(build_spec(), indent=2, ensure_ascii=False) + "\n"


def released_path() -> Path:
    """The path of ``released.json``, the record of released protocol versions."""
    return Path(__file__).parent / "released.json"


def released() -> dict[str, dict]:
    """The protocol versions a Pipecat release has shipped.

    Returns:
        For each version, the SHA-256 of its contract (``contract_sha256``, see
        :func:`contract_sha256`) and the Pipecat release that first shipped it
        (``pipecat``).
    """
    return json.loads(released_path().read_text(encoding="utf-8"))


# Keys that only describe, and maps whose keys are names rather than keywords. A
# deprecation describes too: a deprecated message or field is sent and accepted as
# before.
_PROSE = frozenset(
    {"description", "summary", "title", "examples", "tags", "deprecated", "x-rtvi-deprecated"}
)
_NAMED = frozenset(
    {"properties", "patternProperties", "$defs", "schemas", "messages", "channels", "operations"}
)


def _contract(node: Any, named: bool = False) -> Any:
    """A document without its prose: what a client and a server must agree on."""
    if isinstance(node, list):
        return [_contract(item) for item in node]
    if not isinstance(node, dict):
        return node
    if named:
        return {key: _contract(value) for key, value in node.items()}
    return {
        key: _contract(value, key in _NAMED) for key, value in node.items() if key not in _PROSE
    }


def contract_sha256(version: str = RTVI.PROTOCOL_VERSION) -> str:
    """The SHA-256 of a protocol version's contract, as hex.

    The contract is the document without its descriptions, summaries, titles,
    examples, tags and deprecations, so clearer wording doesn't change it, and
    any change to the wire format does.

    Args:
        version: The protocol version. Defaults to the current one.
    """
    document = json.loads(spec_path(version).read_text(encoding="utf-8"))
    contract = json.dumps(_contract(document), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(contract.encode()).hexdigest()
