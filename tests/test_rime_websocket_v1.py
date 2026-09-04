#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Rime WebSocket v1 framing and state management."""

import asyncio
from typing import Any

import pytest
import websockets
from google.protobuf import json_format
from websockets.asyncio.server import serve
from websockets.protocol import State

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregationType,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.rime._proto import websocket_v1_pb2 as proto
from pipecat.services.rime._websocket_v1 import (
    AudioEvent,
    CancelledEvent,
    ConnectionErrorEvent,
    ContextErrorEvent,
    DoneEvent,
    RimeV1ConnectionError,
    RimeV1ProtocolError,
    RimeV1ProviderError,
    RimeV1StateError,
    RimeWebSocketV1Client,
    StartedEvent,
    SynthesisOptions,
    model_from_websocket_url,
    subprotocol_for_protocol,
    validate_websocket_url,
)
from pipecat.services.rime.tts import RimeTTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test


class _ScriptedSocket:
    def __init__(self, protocol: str) -> None:
        self.subprotocol = protocol
        self.sent: list[str | bytes] = []
        self.incoming: asyncio.Queue[str | bytes | Exception] = asyncio.Queue()
        self.closed = False
        self.fail_writes = False

    async def send(self, message: str | bytes) -> None:
        if self.fail_writes:
            raise ConnectionError("secret write failure")
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        message = await self.incoming.get()
        if isinstance(message, Exception):
            raise message
        return message

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True


class _TerminalRaceSocket(_ScriptedSocket):
    def __init__(self, request_payload: str, response_payload: str) -> None:
        super().__init__("rime.v1.binary")
        self._request_payload = request_payload
        self._response_payload = response_payload
        self.allow_send_to_return = asyncio.Event()

    async def send(self, message: str | bytes) -> None:
        await super().send(message)
        request = _request(message)
        if request.WhichOneof("payload") == self._request_payload:
            await self.incoming.put(
                _response(
                    "binary",
                    contextId=request.context_id,
                    **{self._response_payload: {}},
                )
            )
            await self.allow_send_to_return.wait()


class _StartErrorRaceSocket(_ScriptedSocket):
    def __init__(self) -> None:
        super().__init__("rime.v1.binary")
        self.allow_send_to_return = asyncio.Event()

    async def send(self, message: str | bytes) -> None:
        await super().send(message)
        request = _request(message)
        if request.WhichOneof("payload") == "start":
            await self.incoming.put(
                _response(
                    "binary",
                    contextId=request.context_id,
                    error={"kind": "invalid_input", "message": "rejected"},
                )
            )
            await self.allow_send_to_return.wait()


def _response(protocol: str, **payload: Any) -> str | bytes:
    response = proto.WebSocketResponse()
    json_format.ParseDict(payload, response)
    if protocol == "binary":
        return response.SerializeToString()
    return json_format.MessageToJson(response, preserving_proto_field_name=False, indent=None)


def _request(message: str | bytes) -> proto.WebSocketRequest:
    request = proto.WebSocketRequest()
    if isinstance(message, bytes):
        request.ParseFromString(message)
    else:
        json_format.Parse(message, request)
    return request


async def _ready_client(protocol: str = "binary") -> tuple[RimeWebSocketV1Client, _ScriptedSocket]:
    socket = _ScriptedSocket(subprotocol_for_protocol(protocol))
    await socket.incoming.put(_response(protocol, ready={"protocol": 1, "languages": ["eng"]}))
    client = RimeWebSocketV1Client(socket, protocol=protocol)
    ready = await client.wait_ready(0.1)
    assert ready.protocol == 1
    return client, socket


def _options(**kwargs: Any) -> SynthesisOptions:
    values = {
        "model": "coda",
        "speaker": "astra",
        "language": "eng",
        "sample_rate": 24000,
    }
    values.update(kwargs)
    return SynthesisOptions(**values)


def test_binary_envelope_goldens_match_rime_field_numbers() -> None:
    request = proto.WebSocketRequest(context_id="turn-42", text="hello")
    response = proto.WebSocketResponse(context_id="turn-42", audio=b"\x01\x02")

    assert request.SerializeToString() == b"\x0a\x07turn-42\x22\x05hello"
    assert response.SerializeToString() == b"\x0a\x07turn-42\x22\x02\x01\x02"


@pytest.mark.parametrize("protocol", ["binary", "json"])
@pytest.mark.asyncio
async def test_start_text_and_end_use_one_typed_context(protocol: str) -> None:
    client, socket = await _ready_client(protocol)

    await client.send_text("turn", _options(text_lookahead_tokens=3), "First. ")
    await client.send_text("turn", _options(speaker="ignored"), "Second. ")
    await client.end("turn")

    requests = [_request(message) for message in socket.sent]
    assert [request.WhichOneof("payload") for request in requests] == [
        "start",
        "text",
        "text",
        "end",
    ]
    assert all(request.context_id == "turn" for request in requests)
    assert requests[0].start.text == ""
    assert requests[0].start.speaker == "astra"
    assert requests[0].start.audio_parameters.audio_format == "audio/pcm"
    assert requests[0].start.audio_parameters.sampling_rate == 24000
    assert requests[0].start.coda_parameters.text_lookahead_tokens == 3
    assert [request.text for request in requests[1:3]] == ["First. ", "Second. "]


@pytest.mark.parametrize("protocol", ["binary", "json"])
@pytest.mark.asyncio
async def test_multiplexed_events_remain_attached_to_their_context(protocol: str) -> None:
    client, socket = await _ready_client(protocol)
    await client.send_text("a", _options(), "A.")
    await client.send_text("b", _options(), "B.")
    await client.end("a")
    await client.end("b")

    responses = (
        {"contextId": "a", "started": {"requestId": "request-a"}},
        {"contextId": "b", "started": {"requestId": "request-b"}},
        {"contextId": "b", "audio": "AwQ="},
        {"contextId": "a", "audio": "AQI="},
        {"contextId": "b", "done": {}},
        {"contextId": "a", "done": {}},
    )
    for response in responses:
        await socket.incoming.put(_response(protocol, **response))

    events = client.events()
    received = [await anext(events) for _ in responses]

    assert received == [
        StartedEvent("a", "request-a"),
        StartedEvent("b", "request-b"),
        AudioEvent("b", b"\x03\x04"),
        AudioEvent("a", b"\x01\x02"),
        DoneEvent("b"),
        DoneEvent("a"),
    ]


@pytest.mark.asyncio
async def test_cancel_accepts_cancelled_and_a_racing_done() -> None:
    for payload, expected_type in (("cancelled", CancelledEvent), ("done", DoneEvent)):
        client, socket = await _ready_client()
        await client.send_text("turn", _options(), "Hello.")
        await client.cancel("turn")
        await socket.incoming.put(
            _response("binary", contextId="turn", started={"requestId": "request"})
        )
        await socket.incoming.put(_response("binary", contextId="turn", **{payload: {}}))

        events = client.events()
        assert isinstance(await anext(events), StartedEvent)
        assert isinstance(await anext(events), expected_type)


@pytest.mark.parametrize(
    ("request_payload", "response_payload", "expected_type"),
    [
        ("end", "done", DoneEvent),
        ("cancel", "cancelled", CancelledEvent),
    ],
)
@pytest.mark.asyncio
async def test_terminal_response_can_arrive_before_send_returns(
    request_payload: str,
    response_payload: str,
    expected_type: type[DoneEvent] | type[CancelledEvent],
) -> None:
    socket = _TerminalRaceSocket(request_payload, response_payload)
    await socket.incoming.put(_response("binary", ready={"protocol": 1}))
    client = RimeWebSocketV1Client(socket, protocol="binary")
    await client.wait_ready(0.1)
    await client.send_text("turn", _options(), "Hello.")
    await socket.incoming.put(
        _response("binary", contextId="turn", started={"requestId": "request"})
    )
    events = client.events()
    assert isinstance(await anext(events), StartedEvent)

    send_terminal = asyncio.create_task(getattr(client, request_payload)("turn"))
    try:
        event = await asyncio.wait_for(anext(events), timeout=0.1)
    finally:
        socket.allow_send_to_return.set()
        await send_terminal

    assert isinstance(event, expected_type)


@pytest.mark.asyncio
async def test_context_error_during_start_prevents_the_text_write() -> None:
    socket = _StartErrorRaceSocket()
    await socket.incoming.put(_response("binary", ready={"protocol": 1}))
    client = RimeWebSocketV1Client(socket, protocol="binary")
    await client.wait_ready(0.1)

    send_text = asyncio.create_task(client.send_text("turn", _options(), "Hello."))
    event = await asyncio.wait_for(anext(client.events()), timeout=0.1)
    socket.allow_send_to_return.set()
    await send_text

    assert event == ContextErrorEvent("turn", "invalid_input", None)
    assert [_request(message).WhichOneof("payload") for message in socket.sent] == ["start"]


@pytest.mark.asyncio
async def test_started_accepts_an_empty_request_id() -> None:
    client, socket = await _ready_client()
    await client.send_text("turn", _options(), "Hello.")
    await socket.incoming.put(_response("binary", contextId="turn", started={"requestId": ""}))

    assert await anext(client.events()) == StartedEvent("turn", "")


@pytest.mark.asyncio
async def test_audio_before_started_invalidates_the_protocol() -> None:
    client, socket = await _ready_client()
    await client.send_text("turn", _options(), "Hello.")
    await socket.incoming.put(_response("binary", contextId="turn", audio="AQI="))

    with pytest.raises(RimeV1ProtocolError, match="audio before started"):
        await anext(client.events())


@pytest.mark.asyncio
async def test_json_rejects_invalid_base64_without_echoing_the_value() -> None:
    client, socket = await _ready_client("json")
    await client.send_text("turn", _options(), "Hello.")
    await socket.incoming.put('{"contextId":"turn","audio":"secret%%%"}')

    with pytest.raises(RimeV1ProtocolError, match="invalid Base64") as exc_info:
        await anext(client.events())
    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_rejects_work_before_ready() -> None:
    socket = _ScriptedSocket("rime.v1.binary")
    client = RimeWebSocketV1Client(socket, protocol="binary")

    with pytest.raises(RimeV1StateError, match="not ready"):
        await client.send_text("turn", _options(), "Hello.")


@pytest.mark.asyncio
async def test_ready_gate_rejects_wrong_scope_version_and_sanitizes_errors() -> None:
    socket = _ScriptedSocket("rime.v1.binary")
    await socket.incoming.put(_response("binary", contextId="turn", ready={"protocol": 1}))
    with pytest.raises(RimeV1ProtocolError, match="connection-level"):
        await RimeWebSocketV1Client(socket, protocol="binary").wait_ready(0.1)

    socket = _ScriptedSocket("rime.v1.binary")
    await socket.incoming.put(_response("binary", ready={"protocol": 2}))
    with pytest.raises(RimeV1ProtocolError, match="protocol version"):
        await RimeWebSocketV1Client(socket, protocol="binary").wait_ready(0.1)

    socket = _ScriptedSocket("rime.v1.binary")
    await socket.incoming.put(
        _response(
            "binary",
            error={"kind": "unauthenticated", "message": "secret provider detail"},
        )
    )
    with pytest.raises(RimeV1ProviderError, match="unauthenticated") as exc_info:
        await RimeWebSocketV1Client(socket, protocol="binary").wait_ready(0.1)
    assert "secret provider detail" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_text_and_repeated_terminal_commands_send_no_extra_work() -> None:
    client, socket = await _ready_client()

    await client.send_text("turn", _options(), "")
    assert socket.sent == []

    await client.send_text("turn", _options(), "Hello.")
    await client.end("turn")
    await client.end("turn")
    await client.cancel("turn")
    await client.cancel("turn")

    assert [_request(message).WhichOneof("payload") for message in socket.sent] == [
        "start",
        "text",
        "end",
        "cancel",
    ]
    with pytest.raises(RimeV1StateError, match="no longer accepts text"):
        await client.send_text("turn", _options(), "Too late.")


@pytest.mark.asyncio
async def test_context_and_connection_errors_have_the_correct_scope() -> None:
    client, socket = await _ready_client()
    await client.send_text("a", _options(), "A.")
    await client.send_text("b", _options(), "B.")
    await socket.incoming.put(
        _response(
            "binary",
            contextId="a",
            error={"kind": "invalid_input", "message": "private detail"},
        )
    )
    await socket.incoming.put(
        _response(
            "binary",
            error={"kind": "unavailable", "message": "private detail"},
        )
    )

    events = client.events()
    assert await anext(events) == ContextErrorEvent("a", "invalid_input", None)
    assert await anext(events) == ConnectionErrorEvent("unavailable", None)
    assert not client.has_context("a")
    assert client.has_context("b")


@pytest.mark.asyncio
async def test_unknown_context_and_duplicate_terminal_are_protocol_errors() -> None:
    client, socket = await _ready_client()
    await client.send_text("turn", _options(), "Hello.")
    await socket.incoming.put(
        _response("binary", contextId="unknown", started={"requestId": "request"})
    )
    with pytest.raises(RimeV1ProtocolError, match="unknown context"):
        await anext(client.events())

    client, socket = await _ready_client()
    await client.send_text("turn", _options(), "Hello.")
    await client.end("turn")
    await socket.incoming.put(
        _response("binary", contextId="turn", started={"requestId": "request"})
    )
    await socket.incoming.put(_response("binary", contextId="turn", done={}))
    await socket.incoming.put(_response("binary", contextId="turn", done={}))
    events = client.events()
    assert isinstance(await anext(events), StartedEvent)
    assert isinstance(await anext(events), DoneEvent)
    with pytest.raises(RimeV1ProtocolError, match="after a terminal event"):
        await anext(events)


@pytest.mark.asyncio
async def test_discard_context_releases_terminal_state() -> None:
    client, socket = await _ready_client()
    events = client.events()

    for index in range(10):
        context_id = f"turn-{index}"
        await client.send_text(context_id, _options(), "Hello.")
        await client.end(context_id)
        await socket.incoming.put(
            _response(
                "binary",
                contextId=context_id,
                started={"requestId": f"request-{index}"},
            )
        )
        await socket.incoming.put(_response("binary", contextId=context_id, done={}))

        assert isinstance(await anext(events), StartedEvent)
        assert isinstance(await anext(events), DoneEvent)
        client.discard_context(context_id)

    assert client._contexts == {}
    assert client._closed_contexts == set()


@pytest.mark.asyncio
async def test_json_envelope_validation_is_strict_and_forward_compatible() -> None:
    client, socket = await _ready_client("json")
    await client.send_text("turn", _options(), "Hello.")
    await socket.incoming.put(
        '{"contextId":"turn","started":{"requestId":"request","futureField":1}}'
    )
    assert isinstance(await anext(client.events()), StartedEvent)

    await socket.incoming.put('{"contextId":"turn","done":{},"cancelled":{}}')
    with pytest.raises(RimeV1ProtocolError, match="exactly one payload"):
        await anext(client.events())


@pytest.mark.parametrize(
    "url",
    [
        "wss://rime.ai/coda/ws",
        "wss://api.rime.ai/coda/ws",
        "wss://API.RIME.AI./coda/ws",
        "ws://localhost/coda/ws",
        "ws://127.0.0.1/coda/ws",
        "ws://[::1]/coda/ws",
    ],
)
def test_endpoint_validation_accepts_trusted_and_loopback_hosts(url: str) -> None:
    validate_websocket_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.rime.ai/coda/ws",
        "wss://rime.ai.example.com/coda/ws",
        "ws://api.rime.ai/coda/ws",
        "wss://api.rime.ai/coda/ws/",
        "wss://user@api.rime.ai/coda/ws",
    ],
)
def test_endpoint_validation_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(ValueError):
        validate_websocket_url(url)


def test_custom_endpoint_requires_tls_and_opt_in() -> None:
    with pytest.raises(ValueError, match="allow_custom_endpoint"):
        validate_websocket_url("wss://example.com/coda/ws")
    validate_websocket_url("wss://example.com/coda/ws", allow_custom_endpoint=True)
    with pytest.raises(ValueError, match="must use wss"):
        validate_websocket_url("ws://example.com/coda/ws", allow_custom_endpoint=True)


def test_model_resolution_handles_route_and_dedicated_endpoint() -> None:
    assert model_from_websocket_url("wss://api.rime.ai/coda/ws") == "coda"
    assert model_from_websocket_url("wss://api.rime.ai/mist/ws") == "mistv3"
    assert model_from_websocket_url("wss://api.rime.ai/ws") is None
    with pytest.raises(ValueError, match="/mist/ws"):
        model_from_websocket_url("wss://api.rime.ai/mistv3/ws")


def test_service_selects_binary_v1_without_changing_legacy_default() -> None:
    legacy = RimeTTSService(api_key="key")
    v1 = RimeTTSService(api_key="key", websocket_url="wss://api.rime.ai/coda/ws")

    assert legacy._url == "wss://users-ws.rime.ai/ws3"
    assert not legacy._use_websocket_v1
    assert v1._websocket_protocol == "binary"
    assert v1._text_aggregation_mode is TextAggregationMode.SENTENCE
    assert not v1._push_text_frames
    assert v1._push_start_frame
    assert not v1._push_stop_frames


def test_dedicated_endpoint_uses_explicit_model() -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://customer.rime.ai/ws",
        settings=RimeTTSService.Settings(model="mistv3", pauseBetweenBrackets=True),
    )

    assert service._build_v1_options().model == "mistv3"
    assert service._build_v1_options().pause_between_brackets is True


def test_dedicated_endpoint_requires_explicit_model() -> None:
    with pytest.raises(ValueError, match="requires a model for a dedicated endpoint"):
        RimeTTSService(
            api_key="key",
            websocket_url="wss://customer.rime.ai/ws",
        )


def test_service_rejects_conflicting_modes_and_unsupported_settings() -> None:
    with pytest.raises(ValueError, match="cannot be used together"):
        RimeTTSService(
            api_key="key",
            url="wss://legacy.example/ws3",
            websocket_url="wss://api.rime.ai/coda/ws",
        )
    with pytest.raises(ValueError, match="requires websocket_url"):
        RimeTTSService(api_key="key", websocket_protocol="json")
    with pytest.raises(ValueError, match="sentence text aggregation"):
        RimeTTSService(
            api_key="key",
            websocket_url="wss://api.rime.ai/coda/ws",
            text_aggregation_mode=TextAggregationMode.TOKEN,
        )
    with pytest.raises(ValueError, match="repetition_penalty"):
        RimeTTSService(
            api_key="key",
            websocket_url="wss://api.rime.ai/coda/ws",
            settings=RimeTTSService.Settings(repetition_penalty=1.1),
        )
    with pytest.raises(ValueError, match="does not match"):
        RimeTTSService(
            api_key="key",
            websocket_url="wss://api.rime.ai/coda/ws",
            settings=RimeTTSService.Settings(model="mistv3"),
        )
    with pytest.raises(ValueError, match="requires Rime WebSocket v1"):
        RimeTTSService(
            api_key="key",
            settings=RimeTTSService.Settings(text_lookahead_tokens=3),
        )


def test_mist_route_maps_model_and_settings() -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/mist/ws",
        sample_rate=24000,
        settings=RimeTTSService.Settings(pauseBetweenBrackets=True),
    )
    service._sample_rate = 24000

    client_options = service._build_v1_options()
    assert client_options.model == "mistv3"
    assert client_options.pause_between_brackets is True


def test_service_maps_v1_settings_to_start_options() -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/coda/ws",
        sample_rate=22050,
        settings=RimeTTSService.Settings(
            voice="astra",
            language="eng",
            text_lookahead_tokens=4,
            timeScaleFactor=1.2,
        ),
    )
    service._sample_rate = 22050

    assert service._build_v1_options() == _options(
        sample_rate=22050,
        text_lookahead_tokens=4,
        time_scale_factor=1.2,
    )


def test_v1_pcm_remainders_are_isolated_by_context() -> None:
    service = RimeTTSService.__new__(RimeTTSService)
    service._v1_audio_remainders = {}

    assert service._sample_aligned_v1_audio("a", b"\x01\x02\x03") == b"\x01\x02"
    assert service._sample_aligned_v1_audio("b", b"\x10\x11") == b"\x10\x11"
    assert service._sample_aligned_v1_audio("a", b"\x04") == b"\x03\x04"
    assert service._v1_audio_remainders == {}


@pytest.mark.asyncio
async def test_runtime_settings_apply_to_the_next_context() -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/coda/ws",
        sample_rate=24000,
        settings=RimeTTSService.Settings(voice="old"),
    )
    service._sample_rate = 24000
    service._v1_options_by_context["a"] = service._build_v1_options()

    changed = await service._update_settings(RimeTTSService.Settings(voice="new"))

    assert changed == {"voice": "old"}
    assert service._v1_options_by_context["a"].speaker == "old"
    assert service._build_v1_options().speaker == "new"


@pytest.mark.asyncio
async def test_dedicated_endpoint_rejects_runtime_model_change() -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://customer.rime.ai/ws",
        settings=RimeTTSService.Settings(model="coda"),
    )

    with pytest.raises(ValueError, match="does not match the v1 endpoint model"):
        await service._update_settings(RimeTTSService.Settings(model="mistv3"))


@pytest.mark.asyncio
async def test_finish_v1_context_releases_terminal_tombstones() -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/coda/ws",
    )
    client, socket = await _ready_client()
    service._v1_client = client
    await client.send_text("turn", _options(), "Hello.")
    await client.end("turn")
    await socket.incoming.put(
        _response("binary", contextId="turn", started={"requestId": "request"})
    )
    await socket.incoming.put(_response("binary", contextId="turn", done={}))
    events = client.events()
    assert isinstance(await anext(events), StartedEvent)
    assert isinstance(await anext(events), DoneEvent)
    service._v1_closed_contexts.add("turn")

    await service._finish_v1_context("turn")

    assert service._v1_closed_contexts == set()
    assert client._contexts == {}
    assert client._closed_contexts == set()


@pytest.mark.asyncio
async def test_interruption_after_provider_completion_releases_local_context_state() -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/coda/ws",
    )
    client, _ = await _ready_client()
    service._v1_client = client
    service._v1_options_by_context["turn"] = _options()

    await service.on_audio_context_interrupted("turn")

    assert service._v1_closed_contexts == set()
    assert service._v1_options_by_context == {}


@pytest.mark.parametrize(
    ("request_payload", "response_payload", "task_map_name"),
    [
        ("end", "done", "_v1_terminal_watchdogs"),
        ("cancel", "cancelled", "_v1_cancel_watchdogs"),
    ],
)
@pytest.mark.asyncio
async def test_terminal_send_race_releases_completed_watchdog(
    request_payload: str,
    response_payload: str,
    task_map_name: str,
) -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/coda/ws",
    )
    socket = _TerminalRaceSocket(request_payload, response_payload)
    await socket.incoming.put(_response("binary", ready={"protocol": 1}))
    client = RimeWebSocketV1Client(socket, protocol="binary")
    await client.wait_ready(0.1)
    service._websocket = socket
    service._v1_client = client

    service.create_task = lambda coroutine, name: asyncio.create_task(coroutine)

    async def cancel_task(task: asyncio.Task[None]) -> None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    service.cancel_task = cancel_task
    await client.send_text("turn", _options(), "Hello.")
    await socket.incoming.put(
        _response("binary", contextId="turn", started={"requestId": "request"})
    )
    receive_task = asyncio.create_task(service._receive_v1_messages())

    if request_payload == "end":
        send_task = asyncio.create_task(service.flush_audio("turn"))
    else:
        send_task = asyncio.create_task(service.on_audio_context_interrupted("turn"))

    try:
        for _ in range(10):
            if not client.has_context("turn"):
                break
            await asyncio.sleep(0)
        assert not client.has_context("turn")
        socket.allow_send_to_return.set()
        await send_task
        await asyncio.sleep(0)

        assert getattr(service, task_map_name) == {}
    finally:
        socket.allow_send_to_return.set()
        if not send_task.done():
            await send_task
        receive_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await receive_task


@pytest.mark.asyncio
async def test_context_error_is_queued_before_turn_completion_closes_context() -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/coda/ws",
    )
    context_id = "turn"
    service._turn_context_id = context_id
    service._audio_contexts = {context_id: asyncio.Queue()}
    cancellation_started = asyncio.Event()
    allow_cancellation_to_finish = asyncio.Event()

    async def block_context_task_cancellation(context_id: str, **kwargs: Any) -> None:
        cancellation_started.set()
        await allow_cancellation_to_finish.wait()

    service._cancel_v1_context_tasks = block_context_task_cancellation
    error = ErrorFrame(error="Rime WebSocket v1 request failed")
    finish_task = asyncio.create_task(service._finish_v1_context(context_id, error=error))
    await cancellation_started.wait()

    await service.on_turn_context_completed()
    allow_cancellation_to_finish.set()
    await finish_task

    queue = service._audio_contexts[context_id]
    queued_error = queue.get_nowait()
    queued_stop = queue.get_nowait()
    assert queued_error is error
    assert isinstance(queued_stop, TTSStoppedFrame)
    assert queued_stop.context_id == context_id
    assert queue.get_nowait() is None


@pytest.mark.asyncio
async def test_failed_v1_request_releases_service_tombstone() -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/coda/ws",
    )
    client, socket = await _ready_client()
    socket.state = object()
    service._websocket = socket
    service._v1_client = client

    frames = [frame async for frame in service._run_tts_v1("Hello.", "")]

    assert [type(frame) for frame in frames] == [ErrorFrame, TTSStoppedFrame]
    assert service._v1_closed_contexts == set()


@pytest.mark.asyncio
async def test_old_receive_failure_does_not_close_replacement_connection() -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/coda/ws",
    )
    old_client, old_socket = await _ready_client()
    old_socket.state = State.CLOSED
    service._websocket = old_socket
    service._v1_client = old_client

    new_socket = _ScriptedSocket("rime.v1.binary")
    new_socket.state = State.OPEN
    await new_socket.incoming.put(_response("binary", ready={"protocol": 1, "languages": ["eng"]}))

    async def connect(*args: Any, **kwargs: Any) -> _ScriptedSocket:
        return new_socket

    service._websocket_connect = connect
    failure_started = asyncio.Event()
    allow_failure_cleanup = asyncio.Event()

    async def block_failure_cleanup(message: str, category: Any) -> None:
        failure_started.set()
        await allow_failure_cleanup.wait()

    service._fail_all_v1_contexts = block_failure_cleanup

    receive_task = asyncio.create_task(service._receive_v1_messages())
    await old_socket.incoming.put(ConnectionError("old socket failed"))
    await failure_started.wait()

    connect_task = asyncio.create_task(service._connect_websocket_v1())
    _, pending = await asyncio.wait({connect_task}, timeout=0.01)
    allow_failure_cleanup.set()

    with pytest.raises(RimeV1ConnectionError, match="Failed to read from the Rime v1 WebSocket"):
        await receive_task
    await connect_task

    assert service._websocket is new_socket
    assert service._v1_client is not old_client
    assert not new_socket.closed
    assert pending == {connect_task}


@pytest.mark.asyncio
async def test_old_receive_loop_does_not_reconnect_over_replacement_connection() -> None:
    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/coda/ws",
    )
    old_client, _ = await _ready_client()
    service._v1_client = old_client
    receive_task = asyncio.create_task(service._receive_v1_messages())
    await asyncio.sleep(0)

    new_client, new_socket = await _ready_client()
    new_socket.state = State.OPEN
    service._websocket = new_socket
    service._v1_client = new_client

    async def reject_connect(*args: Any, **kwargs: Any) -> _ScriptedSocket:
        raise ConnectionError("unexpected reconnect")

    service._websocket_connect = reject_connect
    try:
        assert await service._reconnect_websocket(1)
    finally:
        receive_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await receive_task

    assert service._websocket is new_socket
    assert service._v1_client is new_client
    assert not new_socket.closed


@pytest.mark.asyncio
async def test_run_tts_does_not_start_tasks_for_a_finished_context() -> None:
    class FinishedContextClient:
        async def send_text(self, context_id: str, options: SynthesisOptions, text: str) -> None:
            pass

        def has_context(self, context_id: str) -> bool:
            return False

    service = RimeTTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/coda/ws",
    )
    socket = _ScriptedSocket("rime.v1.binary")
    socket.state = object()
    service._websocket = socket
    service._v1_client = FinishedContextClient()
    started_tasks: list[tuple[str, str]] = []
    service._start_v1_keepalive = lambda context_id: started_tasks.append(("keepalive", context_id))
    service._start_v1_start_watchdog = lambda context_id: started_tasks.append(
        ("watchdog", context_id)
    )

    frames = [frame async for frame in service._run_tts_v1("Hello.", "turn")]

    assert frames == []
    assert started_tasks == []


def _server_message(protocol: str, response: proto.WebSocketResponse) -> str | bytes:
    if protocol == "rime.v1.binary":
        return response.SerializeToString()
    return json_format.MessageToJson(response, preserving_proto_field_name=False, indent=None)


def _server_request(message: str | bytes) -> proto.WebSocketRequest:
    request = proto.WebSocketRequest()
    if isinstance(message, bytes):
        request.ParseFromString(message)
    else:
        json_format.Parse(message, request)
    return request


@pytest.mark.parametrize("protocol", ["binary", "json"])
@pytest.mark.asyncio
async def test_service_round_trip_uses_v1_frames_and_pipecat_order(protocol: str) -> None:
    captured: dict[str, Any] = {"requests": []}
    audio = b"\x01\x02\x03\x04"

    async def handler(websocket) -> None:
        captured["authorization"] = websocket.request.headers.get("Authorization")
        captured["subprotocol"] = websocket.subprotocol
        await websocket.send(
            _server_message(
                websocket.subprotocol,
                proto.WebSocketResponse(ready=proto.WebSocketReady(protocol=1)),
            )
        )
        try:
            async for message in websocket:
                request = _server_request(message)
                captured["requests"].append(request)
                context_id = request.context_id
                payload = request.WhichOneof("payload")
                if payload == "start":
                    await websocket.send(
                        _server_message(
                            websocket.subprotocol,
                            proto.WebSocketResponse(
                                context_id=context_id,
                                started=proto.WebSocketStarted(request_id="request-1"),
                            ),
                        )
                    )
                elif payload == "text":
                    await websocket.send(
                        _server_message(
                            websocket.subprotocol,
                            proto.WebSocketResponse(context_id=context_id, audio=audio),
                        )
                    )
                elif payload == "end":
                    await websocket.send(
                        _server_message(
                            websocket.subprotocol,
                            proto.WebSocketResponse(
                                context_id=context_id,
                                done=proto.WebSocketDone(),
                            ),
                        )
                    )
        except websockets.ConnectionClosed:
            pass

    async with serve(
        handler,
        "127.0.0.1",
        0,
        subprotocols=["rime.v1.binary", "rime.v1.json"],
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        service = RimeTTSService(
            api_key="test-key",
            websocket_url=f"ws://{host}:{port}/coda/ws",
            websocket_protocol=protocol,
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            service,
            frames_to_send=[
                TTSSpeakFrame(text="Hello from Pipecat. This is a second sentence."),
                SleepFrame(sleep=0.2),
                BotStoppedSpeakingFrame(),
            ],
            start_timeout=3.0,
        )

    assert captured["authorization"] == "Bearer test-key"
    assert captured["subprotocol"] == f"rime.v1.{protocol}"
    requests = captured["requests"]
    assert [request.WhichOneof("payload") for request in requests] == [
        "start",
        "text",
        "text",
        "end",
    ]
    assert requests[0].start.audio_parameters.audio_format == "audio/pcm"
    assert requests[0].start.audio_parameters.sampling_rate == 24000
    assert requests[1].text == "Hello from Pipecat. "
    assert requests[2].text == "This is a second sentence. "
    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)

    output_types = [type(frame) for frame in down_frames]
    assert output_types.index(TTSStartedFrame) < output_types.index(TTSAudioRawFrame)
    assert output_types.index(TTSStartedFrame) < output_types.index(TTSTextFrame)
    assert output_types.index(TTSAudioRawFrame) < output_types.index(TTSStoppedFrame)
    assert output_types.index(TTSTextFrame) < output_types.index(TTSStoppedFrame)
    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert b"".join(frame.audio for frame in audio_frames) == audio * 2
    assert not service._v1_keepalive_tasks
    assert not service._v1_start_watchdogs
    assert not service._v1_terminal_watchdogs
    assert not service._v1_cancel_watchdogs


@pytest.mark.asyncio
async def test_context_error_without_audio_does_not_publish_tts_text() -> None:
    async def handler(websocket) -> None:
        await websocket.send(
            proto.WebSocketResponse(ready=proto.WebSocketReady(protocol=1)).SerializeToString()
        )
        try:
            async for message in websocket:
                request = _server_request(message)
                if request.WhichOneof("payload") == "start":
                    await websocket.send(
                        proto.WebSocketResponse(
                            context_id=request.context_id,
                            error=proto.WebSocketError(
                                kind="invalid_input",
                                message="rejected",
                            ),
                        ).SerializeToString()
                    )
        except websockets.ConnectionClosed:
            pass

    async with serve(
        handler,
        "127.0.0.1",
        0,
        subprotocols=["rime.v1.binary"],
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        service = RimeTTSService(
            api_key="test-key",
            websocket_url=f"ws://{host}:{port}/coda/ws",
            sample_rate=24000,
        )
        down_frames, up_frames = await run_test(
            service,
            frames_to_send=[
                TTSSpeakFrame(text="This must not enter history."),
                SleepFrame(sleep=0.1),
            ],
            start_timeout=3.0,
        )

    assert any(isinstance(frame, ErrorFrame) for frame in up_frames)
    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in down_frames)
    assert not any(isinstance(frame, TTSTextFrame) for frame in down_frames)
    assert service.is_usable


@pytest.mark.asyncio
async def test_done_without_audio_does_not_publish_tts_text() -> None:
    async def handler(websocket) -> None:
        await websocket.send(
            proto.WebSocketResponse(ready=proto.WebSocketReady(protocol=1)).SerializeToString()
        )
        try:
            async for message in websocket:
                request = _server_request(message)
                payload = request.WhichOneof("payload")
                if payload == "start":
                    await websocket.send(
                        proto.WebSocketResponse(
                            context_id=request.context_id,
                            started=proto.WebSocketStarted(request_id="request-1"),
                        ).SerializeToString()
                    )
                elif payload == "end":
                    await websocket.send(
                        proto.WebSocketResponse(
                            context_id=request.context_id,
                            done=proto.WebSocketDone(),
                        ).SerializeToString()
                    )
        except websockets.ConnectionClosed:
            pass

    async with serve(
        handler,
        "127.0.0.1",
        0,
        subprotocols=["rime.v1.binary"],
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        service = RimeTTSService(
            api_key="test-key",
            websocket_url=f"ws://{host}:{port}/coda/ws",
            sample_rate=24000,
        )
        down_frames, up_frames = await run_test(
            service,
            frames_to_send=[
                TTSSpeakFrame(text="This audio is empty."),
                SleepFrame(sleep=0.1),
            ],
            start_timeout=3.0,
        )

    assert any(isinstance(frame, ErrorFrame) for frame in up_frames)
    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in down_frames)
    assert not any(isinstance(frame, TTSTextFrame) for frame in down_frames)


@pytest.mark.asyncio
async def test_context_error_discards_later_sentences_in_the_same_turn() -> None:
    requests: list[str] = []

    async def handler(websocket) -> None:
        await websocket.send(
            proto.WebSocketResponse(ready=proto.WebSocketReady(protocol=1)).SerializeToString()
        )
        try:
            async for message in websocket:
                request = _server_request(message)
                payload = request.WhichOneof("payload")
                assert payload is not None
                requests.append(payload)
                if payload == "start":
                    await websocket.send(
                        proto.WebSocketResponse(
                            context_id=request.context_id,
                            started=proto.WebSocketStarted(request_id="request-1"),
                        ).SerializeToString()
                    )
                elif payload == "text":
                    await websocket.send(
                        proto.WebSocketResponse(
                            context_id=request.context_id,
                            error=proto.WebSocketError(
                                kind="internal",
                                message="synthesis failed",
                            ),
                        ).SerializeToString()
                    )
        except websockets.ConnectionClosed:
            pass

    async with serve(
        handler,
        "127.0.0.1",
        0,
        subprotocols=["rime.v1.binary"],
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        service = RimeTTSService(
            api_key="test-key",
            websocket_url=f"ws://{host}:{port}/coda/ws",
            sample_rate=24000,
        )
        down_frames, up_frames = await run_test(
            service,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                AggregatedTextFrame("First sentence.", AggregationType.SENTENCE),
                SleepFrame(sleep=0.05),
                AggregatedTextFrame("Second sentence.", AggregationType.SENTENCE),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.1),
            ],
            start_timeout=3.0,
        )

    assert requests == ["start", "text"]
    assert any(isinstance(frame, ErrorFrame) for frame in up_frames)
    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in down_frames)
    assert not any(isinstance(frame, TTSTextFrame) for frame in down_frames)


@pytest.mark.asyncio
async def test_service_interruption_sends_cancel_and_drops_late_audio() -> None:
    captured: list[str] = []
    connections = 0
    first_audio = b"\x01\x02"
    stale_audio = b"\x03\x04"

    async def handler(websocket) -> None:
        nonlocal connections
        connections += 1
        await websocket.send(
            proto.WebSocketResponse(ready=proto.WebSocketReady(protocol=1)).SerializeToString()
        )
        try:
            async for message in websocket:
                request = _server_request(message)
                payload = request.WhichOneof("payload")
                assert payload is not None
                captured.append(payload)
                if payload == "start":
                    await websocket.send(
                        proto.WebSocketResponse(
                            context_id=request.context_id,
                            started=proto.WebSocketStarted(request_id="request-1"),
                        ).SerializeToString()
                    )
                elif payload == "text":
                    await websocket.send(
                        proto.WebSocketResponse(
                            context_id=request.context_id,
                            audio=first_audio,
                        ).SerializeToString()
                    )
                elif payload == "cancel":
                    await websocket.send(
                        proto.WebSocketResponse(
                            context_id=request.context_id,
                            audio=stale_audio,
                        ).SerializeToString()
                    )
                    await websocket.send(
                        proto.WebSocketResponse(
                            context_id=request.context_id,
                            cancelled=proto.WebSocketCancelled(),
                        ).SerializeToString()
                    )
        except websockets.ConnectionClosed:
            pass

    async with serve(
        handler,
        "127.0.0.1",
        0,
        subprotocols=["rime.v1.binary"],
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        service = RimeTTSService(
            api_key="test-key",
            websocket_url=f"ws://{host}:{port}/coda/ws",
            sample_rate=24000,
        )
        down_frames, up_frames = await run_test(
            service,
            frames_to_send=[
                TTSSpeakFrame(text="Stop me."),
                SleepFrame(sleep=0.05),
                InterruptionFrame(),
                SleepFrame(sleep=0.1),
            ],
            start_timeout=3.0,
        )

    assert captured == ["start", "text", "end", "cancel"]
    assert connections == 1
    assert "flush" not in captured
    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert b"".join(frame.audio for frame in audio_frames) == first_audio
    assert not service._v1_keepalive_tasks
    assert not service._v1_start_watchdogs
    assert not service._v1_terminal_watchdogs
    assert not service._v1_cancel_watchdogs
