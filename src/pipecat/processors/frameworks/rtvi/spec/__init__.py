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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, TypeAdapter
from pydantic.json_schema import JsonSchemaMode

import pipecat.processors.frameworks.rtvi.models as RTVI

RTVIMessageDirection = Literal["client-to-server", "server-to-client"]

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
            the message to.
        deprecated: What replaces a deprecated message.
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


def _summary(obj: Any) -> str:
    """The first paragraph of an object's docstring, on one line."""
    return " ".join((inspect.getdoc(obj) or "").split("\n\n")[0].split())


def _client(
    type: str, description: str, data: Any = _NO_DATA, example: Any = None
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
    )


def _server(model: type[BaseModel], example: Any = None, **kwargs) -> RTVIMessageSpec:
    """Describe a message the server sends, from the model it sends it with."""
    fields = model.model_fields
    data = fields.get("data")
    return RTVIMessageSpec(
        type=fields["type"].default,
        direction="server-to-client",
        description=_summary(model),
        data=data.annotation if data else None,
        data_required=bool(data and data.is_required()),
        has_id="id" in fields,
        example=example,
        **kwargs,
    )


_TIMESTAMP = "2026-01-01T12:00:00.000Z"

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
    ),
    _client("disconnect-bot", "Asks the server to end the session."),
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
    ),
    _client(
        "ui-cancel-job-group",
        "Asks the server to cancel a job group it reported with ui-job-group.",
        RTVI.UICancelJobGroupData,
        {"job_id": "job_1", "reason": "User closed the panel"},
    ),
    # -- Server to client: session --
    _server(
        RTVI.BotReady,
        {
            "version": RTVI.PROTOCOL_VERSION,
            "about": {"library": "pipecat-ai", "library_version": "1.3.0"},
            "capabilities": {"audio_in": True, "audio_out": True, "video_in": False},
        },
    ),
    _server(RTVI.Error, {"error": "The LLM service failed", "fatal": False}),
    _server(RTVI.ErrorResponse, {"error": "Unsupported type foo"}),
    _server(RTVI.ServerResponse, {"t": "get-weather", "d": {"temperature": 21}}),
    _server(RTVI.ServerMessage, {"event": "order-updated", "status": "shipped"}),
    _server(
        RTVI.MetricsMessage,
        {"ttfb": [{"processor": "OpenAILLMService#0", "value": 0.42}]},
        enabled_by="metrics_enabled",
    ),
    _server(RTVI.SystemLogMessage, {"text": "Pipeline started"}, enabled_by="system_logs_enabled"),
    # -- Server to client: speaking --
    _server(RTVI.UserStartedSpeakingMessage, enabled_by="user_speaking_enabled"),
    _server(RTVI.UserStoppedSpeakingMessage, enabled_by="user_speaking_enabled"),
    _server(RTVI.VADUserStartedSpeakingMessage, enabled_by="vad_user_speaking_enabled"),
    _server(RTVI.VADUserStoppedSpeakingMessage, enabled_by="vad_user_speaking_enabled"),
    _server(RTVI.UserMuteStartedMessage, enabled_by="user_mute_enabled"),
    _server(RTVI.UserMuteStoppedMessage, enabled_by="user_mute_enabled"),
    _server(RTVI.BotStartedSpeakingMessage, enabled_by="bot_speaking_enabled"),
    _server(RTVI.BotStoppedSpeakingMessage, enabled_by="bot_speaking_enabled"),
    _server(RTVI.BotInterruptedMessage, enabled_by="bot_speaking_enabled"),
    _server(RTVI.UserAudioLevelMessage, {"value": 0.3}, enabled_by="user_audio_level_enabled"),
    _server(RTVI.BotAudioLevelMessage, {"value": 0.5}, enabled_by="bot_audio_level_enabled"),
    # -- Server to client: what the user said --
    _server(
        RTVI.UserInputMessage,
        {
            "text": "What's the weather like?",
            "input_type": "transcription",
            "timestamp": _TIMESTAMP,
            "final": True,
            "user_id": "user_1",
        },
        enabled_by="user_input_enabled",
    ),
    _server(
        RTVI.UserLLMTextMessage, {"text": "What's the weather like?"}, enabled_by="user_llm_enabled"
    ),
    _server(
        RTVI.STTRawTextMessage,
        {"text": "What's the", "user_id": "user_1", "timestamp": _TIMESTAMP, "final": False},
        enabled_by="stt_raw_text_enabled",
        min_client_version="2.2.0",
    ),
    _server(
        RTVI.UserTranscriptionMessage,
        {"text": "What's the", "user_id": "user_1", "timestamp": _TIMESTAMP, "final": False},
        enabled_by="user_transcription_enabled",
        deprecated="Use user-input or stt-raw-text instead.",
    ),
    # -- Server to client: what the bot says --
    _server(
        RTVI.BotOutputMessage,
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
    _server(RTVI.BotLLMStartedMessage, enabled_by="bot_llm_enabled"),
    _server(RTVI.BotLLMStoppedMessage, enabled_by="bot_llm_enabled"),
    _server(RTVI.BotTTSStartedMessage, enabled_by="bot_tts_enabled"),
    _server(RTVI.BotTTSStoppedMessage, enabled_by="bot_tts_enabled"),
    _server(
        RTVI.LLMRawTextMessage,
        {"text": "It's sunny"},
        enabled_by="llm_raw_text_enabled",
        min_client_version="2.2.0",
    ),
    _server(
        RTVI.TTSRawTextMessage,
        {"text": "It's"},
        enabled_by="tts_raw_text_enabled",
        min_client_version="2.2.0",
    ),
    _server(
        RTVI.BotLLMMarkerMessage,
        {"text": "✓", "kind": "complete", "raw": "✓ It's sunny in Lisbon.", "markers": ["✓"]},
        enabled_by="bot_llm_marker_enabled",
    ),
    _server(
        RTVI.BotTranscriptionMessage,
        {"text": "It's sunny in Lisbon."},
        enabled_by="bot_llm_enabled",
    ),
    _server(
        RTVI.BotLLMTextMessage,
        {"text": "It's sunny"},
        enabled_by="bot_llm_enabled",
        deprecated="Use llm-raw-text instead.",
    ),
    _server(
        RTVI.BotTTSTextMessage,
        {"text": "It's"},
        enabled_by="bot_tts_enabled",
        deprecated="Use tts-raw-text instead.",
    ),
    # -- Server to client: function calls --
    _server(RTVI.LLMFunctionCallStartMessage, {"function_name": "get_weather"}),
    _server(
        RTVI.LLMFunctionCallInProgressMessage,
        {
            "tool_call_id": "call_123",
            "function_name": "get_weather",
            "arguments": {"city": "Lisbon"},
        },
    ),
    _server(
        RTVI.LLMFunctionCallStoppedMessage,
        {"tool_call_id": "call_123", "cancelled": False, "function_name": "get_weather"},
    ),
    _server(
        RTVI.LLMFunctionCallMessage,
        {"function_name": "get_weather", "tool_call_id": "call_123", "args": {"city": "Lisbon"}},
        deprecated="Use llm-function-call-in-progress instead.",
    ),
    # -- Server to client: UI worker --
    _server(RTVI.UICommandMessage, {"command": "toast", "payload": {"title": "Order shipped"}}),
    _server(
        RTVI.UIJobGroupMessage,
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
        "The server sends a message that has x-rtvi-enabled-by only while that "
        "RTVIObserverParams field is on.",
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


def _message(spec: RTVIMessageSpec, data_schema: Any) -> dict:
    """A message's entry in the document's ``components``."""
    message: dict[str, Any] = {
        "name": spec.type,
        "summary": spec.description,
        "payload": _envelope(spec, data_schema),
        "examples": [{"payload": _example(spec)}],
    }
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
        inputs, ref_template="#/components/schemas/{model}"
    )
    data_schemas = {key: schema for (key, _), schema in schemas.items()}

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
                "The Real-Time Voice Interface protocol between a client and a Pipecat "
                "server, from the server's point of view. Every message is a JSON object "
                f'labeled "{RTVI.MESSAGE_LABEL}" that the transport carries as a data '
                "message."
            ),
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


# Keys that only describe, and maps whose keys are names rather than keywords.
_PROSE = frozenset({"description", "summary", "title", "examples"})
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

    The contract is the document without its descriptions, summaries, titles and
    examples, so clearer wording doesn't change it, and any change to the wire
    format does.

    Args:
        version: The protocol version. Defaults to the current one.
    """
    document = json.loads(spec_path(version).read_text(encoding="utf-8"))
    contract = json.dumps(_contract(document), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(contract.encode()).hexdigest()
