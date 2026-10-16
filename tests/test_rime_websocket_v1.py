#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for RimeV1TTSService."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
import websockets
from google.protobuf import json_format
from rime_api import text_to_speech_pb2 as proto
from websockets.asyncio.server import serve

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
from pipecat.services.rime.tts_v1 import RimeV1TTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.errors import ErrorCategory

Reply = Callable[[proto.WebSocketRequest], list[proto.WebSocketResponse]]


def _started(request: proto.WebSocketRequest) -> proto.WebSocketResponse:
    return proto.WebSocketResponse(
        context_id=request.context_id, started=proto.WebSocketStarted(request_id="request-1")
    )


def _audio(request: proto.WebSocketRequest, audio: bytes) -> proto.WebSocketResponse:
    return proto.WebSocketResponse(context_id=request.context_id, audio=audio)


def _done(request: proto.WebSocketRequest) -> proto.WebSocketResponse:
    return proto.WebSocketResponse(context_id=request.context_id, done=proto.WebSocketDone())


def _error(request: proto.WebSocketRequest, kind: str) -> proto.WebSocketResponse:
    return proto.WebSocketResponse(
        context_id=request.context_id,
        error=proto.WebSocketError(kind=kind, message="rejected"),
    )


def _speaking_reply(audio: bytes = b"\x01\x02") -> Reply:
    """Reply like Rime: started on start, audio per text, done on end."""

    def reply(request: proto.WebSocketRequest) -> list[proto.WebSocketResponse]:
        payload = request.WhichOneof("payload")
        if payload == "start":
            return [_started(request)]
        if payload == "text":
            return [_audio(request, audio)]
        if payload == "end":
            return [_done(request)]
        return []

    return reply


@asynccontextmanager
async def _rime_server(
    reply: Reply,
    *,
    ready: proto.WebSocketResponse | None = None,
) -> AsyncIterator[tuple[str, list[proto.WebSocketRequest]]]:
    """Serve Rime v1 locally; yield the server's base URL and the requests it receives."""
    requests: list[proto.WebSocketRequest] = []
    ready = ready or proto.WebSocketResponse(ready=proto.WebSocketReady(protocol=1))

    async def handler(websocket) -> None:
        binary = websocket.subprotocol == "rime.v1.binary"

        def encode(response: proto.WebSocketResponse) -> str | bytes:
            return response.SerializeToString() if binary else json_format.MessageToJson(response)

        await websocket.send(encode(ready))
        try:
            async for message in websocket:
                request = proto.WebSocketRequest()
                if binary:
                    request.ParseFromString(message)
                else:
                    json_format.Parse(message, request)
                requests.append(request)
                for response in reply(request):
                    await websocket.send(encode(response))
        except websockets.ConnectionClosed:
            pass

    async with serve(handler, "127.0.0.1", 0, subprotocols=["rime.v1.binary", "rime.v1.json"]) as s:
        host, port = next(iter(s.sockets)).getsockname()[:2]
        yield f"ws://{host}:{port}", requests


def _payloads(requests: list[proto.WebSocketRequest]) -> list[str | None]:
    return [request.WhichOneof("payload") for request in requests]


def _texts(frames: list[Any]) -> list[str]:
    return [frame.text for frame in frames if isinstance(frame, TTSTextFrame)]


def _audio_bytes(frames: list[Any]) -> bytes:
    return b"".join(frame.audio for frame in frames if isinstance(frame, TTSAudioRawFrame))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_service_defaults_to_binary_coda() -> None:
    service = RimeV1TTSService(api_key="key")

    assert service._websocket_url == "wss://api.rime.ai/coda/ws"
    assert service._websocket_protocol == "binary"
    assert service._settings.model == "coda"
    assert service._text_aggregation_mode is TextAggregationMode.SENTENCE


@pytest.mark.parametrize(
    "url, model",
    [
        ("wss://api.rime.ai/coda/ws", "coda"),
        ("wss://api.rime.ai/mist/ws", "mistv3"),
        ("wss://api.rime.ai/arcana/ws", "arcana"),
    ],
)
def test_model_comes_from_the_endpoint_path(url: str, model: str) -> None:
    assert RimeV1TTSService(api_key="key", websocket_url=url)._settings.model == model


def test_dedicated_endpoint_requires_an_explicit_model() -> None:
    service = RimeV1TTSService(
        api_key="key",
        websocket_url="wss://customer.rime.ai/ws",
        settings=RimeV1TTSService.Settings(model="mistv3"),
    )
    assert service._settings.model == "mistv3"

    with pytest.raises(ValueError, match="requires a model for a dedicated endpoint"):
        RimeV1TTSService(api_key="key", websocket_url="wss://customer.rime.ai/ws")


def test_service_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="binary"):
        RimeV1TTSService(api_key="key", websocket_protocol="xml")
    with pytest.raises(ValueError, match="must end with /ws"):
        RimeV1TTSService(api_key="key", websocket_url="wss://api.rime.ai/coda")
    with pytest.raises(ValueError, match="sentence text aggregation"):
        RimeV1TTSService(api_key="key", text_aggregation_mode=TextAggregationMode.TOKEN)
    with pytest.raises(ValueError, match="repetition_penalty"):
        RimeV1TTSService(
            api_key="key",
            settings=RimeV1TTSService.Settings.from_mapping({"repetition_penalty": 1.1}),
        )
    with pytest.raises(ValueError, match="does not match"):
        RimeV1TTSService(api_key="key", settings=RimeV1TTSService.Settings(model="mistv3"))
    with pytest.raises(ValueError, match="Mist settings require a Mist model"):
        RimeV1TTSService(
            api_key="key", settings=RimeV1TTSService.Settings(pause_between_brackets=True)
        )


@pytest.mark.parametrize("model", ["mist", "arcana"])
def test_non_coda_endpoint_rejects_lookahead(model: str) -> None:
    with pytest.raises(ValueError, match="text_lookahead_tokens requires the Coda model"):
        RimeV1TTSService(
            api_key="key",
            websocket_url=f"wss://api.rime.ai/{model}/ws",
            settings=RimeV1TTSService.Settings(text_lookahead_tokens=4),
        )


def test_coda_settings_map_to_the_start_request() -> None:
    service = RimeV1TTSService(
        api_key="key",
        sample_rate=22050,
        settings=RimeV1TTSService.Settings(
            voice="astra", language="eng", text_lookahead_tokens=0, time_scale_factor=1.2
        ),
    )
    service._sample_rate = 22050

    start = service._build_start_request("turn").start
    assert start.speaker == "astra"
    assert start.language == "eng"
    assert start.audio_parameters.audio_format == "audio/pcm"
    assert start.audio_parameters.sampling_rate == 22050
    assert start.audio_parameters.time_scale_factor == pytest.approx(1.2)
    assert start.coda_parameters.HasField("text_lookahead_tokens")
    assert start.coda_parameters.text_lookahead_tokens == 0
    assert not start.HasField("mist_parameters")


def test_mist_settings_map_to_the_start_request() -> None:
    service = RimeV1TTSService(
        api_key="key",
        websocket_url="wss://api.rime.ai/mist/ws",
        settings=RimeV1TTSService.Settings(pause_between_brackets=True, save_oovs=False),
    )
    service._sample_rate = 24000

    start = service._build_start_request("turn").start
    assert start.mist_parameters.pause_between_brackets is True
    assert start.mist_parameters.save_oovs is False
    assert not start.HasField("coda_parameters")


@pytest.mark.asyncio
async def test_runtime_settings_are_validated_against_the_endpoint() -> None:
    service = RimeV1TTSService(api_key="key", settings=RimeV1TTSService.Settings(voice="old"))
    service._sample_rate = 24000

    changed = await service._update_settings(RimeV1TTSService.Settings(voice="new"))
    assert changed == {"voice": "old"}
    assert service._build_start_request("turn").start.speaker == "new"

    with pytest.raises(ValueError, match="does not match the v1 endpoint model"):
        await service._update_settings(RimeV1TTSService.Settings(model="mistv3"))

    mist = RimeV1TTSService(api_key="key", websocket_url="wss://api.rime.ai/mist/ws")
    with pytest.raises(ValueError, match="text_lookahead_tokens requires the Coda model"):
        await mist._update_settings(RimeV1TTSService.Settings(text_lookahead_tokens=4))


# ---------------------------------------------------------------------------
# Synthesis against a local Rime v1 server
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["binary", "json"])
@pytest.mark.asyncio
async def test_round_trip_streams_one_context_per_turn(protocol: str) -> None:
    async with _rime_server(_speaking_reply(b"\x01\x02\x03\x04")) as (url, requests):
        service = RimeV1TTSService(
            api_key="test-key",
            websocket_url=f"{url}/coda/ws",
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

    assert _payloads(requests) == ["start", "text", "text", "end"]
    assert len({request.context_id for request in requests}) == 1
    assert requests[0].start.audio_parameters.sampling_rate == 24000
    assert [request.text for request in requests[1:3]] == [
        "Hello from Pipecat. ",
        "This is a second sentence. ",
    ]
    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert _audio_bytes(down_frames) == b"\x01\x02\x03\x04" * 2

    output_types = [type(frame) for frame in down_frames]
    assert output_types.index(TTSStartedFrame) < output_types.index(TTSAudioRawFrame)
    assert output_types.index(TTSAudioRawFrame) < output_types.index(TTSStoppedFrame)
    assert output_types.index(TTSTextFrame) < output_types.index(TTSStoppedFrame)


@pytest.mark.asyncio
async def test_interruption_cancels_the_context_and_drops_late_audio() -> None:
    def reply(request: proto.WebSocketRequest) -> list[proto.WebSocketResponse]:
        payload = request.WhichOneof("payload")
        if payload == "start":
            return [_started(request)]
        if payload == "text":
            return [_audio(request, b"\x01\x02")]
        if payload == "cancel":
            return [
                _audio(request, b"\x03\x04"),
                proto.WebSocketResponse(
                    context_id=request.context_id, cancelled=proto.WebSocketCancelled()
                ),
            ]
        return []

    async with _rime_server(reply) as (url, requests):
        service = RimeV1TTSService(api_key="key", websocket_url=f"{url}/coda/ws")
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

    assert _payloads(requests) == ["start", "text", "end", "cancel"]
    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert _audio_bytes(down_frames) == b"\x01\x02"


@pytest.mark.asyncio
async def test_context_error_reports_and_skips_the_rest_of_the_turn() -> None:
    def reply(request: proto.WebSocketRequest) -> list[proto.WebSocketResponse]:
        payload = request.WhichOneof("payload")
        if payload == "start":
            return [_started(request)]
        if payload == "text":
            return [_error(request, "internal")]
        return []

    async with _rime_server(reply) as (url, requests):
        service = RimeV1TTSService(api_key="key", websocket_url=f"{url}/coda/ws")
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

    assert _payloads(requests) == ["start", "text"]
    errors = [frame for frame in up_frames if isinstance(frame, ErrorFrame)]
    assert errors and "internal" in errors[0].error
    assert errors[0].category is ErrorCategory.SERVER
    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in down_frames)
    assert _texts(down_frames) == ["First sentence.", "Second sentence."]
    assert service.is_usable


@pytest.mark.asyncio
async def test_permanent_context_error_leaves_the_service_usable() -> None:
    def reply(request: proto.WebSocketRequest) -> list[proto.WebSocketResponse]:
        return [_error(request, "invalid_input")] if request.HasField("start") else []

    async with _rime_server(reply) as (url, _):
        service = RimeV1TTSService(api_key="key", websocket_url=f"{url}/coda/ws")
        down_frames, up_frames = await run_test(
            service,
            frames_to_send=[TTSSpeakFrame(text="Rejected text."), SleepFrame(sleep=0.1)],
            start_timeout=3.0,
        )

    errors = [frame for frame in up_frames if isinstance(frame, ErrorFrame)]
    assert errors and errors[0].category is ErrorCategory.UNKNOWN
    assert _texts(down_frames) == ["Rejected text."]
    assert service.is_usable


@pytest.mark.asyncio
async def test_done_without_audio_ends_the_context() -> None:
    def reply(request: proto.WebSocketRequest) -> list[proto.WebSocketResponse]:
        payload = request.WhichOneof("payload")
        if payload == "start":
            return [_started(request)]
        if payload == "end":
            return [_done(request)]
        return []

    async with _rime_server(reply) as (url, _):
        service = RimeV1TTSService(api_key="key", websocket_url=f"{url}/coda/ws")
        down_frames, _ = await run_test(
            service,
            frames_to_send=[TTSSpeakFrame(text="This audio is empty."), SleepFrame(sleep=0.1)],
            start_timeout=3.0,
        )

    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in down_frames)
    assert any(isinstance(frame, TTSStoppedFrame) for frame in down_frames)
    assert _texts(down_frames) == ["This audio is empty."]


@pytest.mark.asyncio
async def test_odd_length_audio_is_padded_without_an_error() -> None:
    def reply(request: proto.WebSocketRequest) -> list[proto.WebSocketResponse]:
        payload = request.WhichOneof("payload")
        if payload == "start":
            return [_started(request)]
        if payload == "text":
            return [_audio(request, b"\x01\x02\x03"), _audio(request, b"\x04\x05")]
        if payload == "end":
            return [_done(request)]
        return []

    async with _rime_server(reply) as (url, _):
        service = RimeV1TTSService(api_key="key", websocket_url=f"{url}/coda/ws")
        down_frames, up_frames = await run_test(
            service,
            frames_to_send=[
                TTSSpeakFrame(text="Odd audio."),
                SleepFrame(sleep=0.2),
                BotStoppedSpeakingFrame(),
            ],
            start_timeout=3.0,
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert _audio_bytes(down_frames) == b"\x01\x02\x03\x04\x05\x00"


@pytest.mark.asyncio
async def test_ready_error_is_reported_with_its_category() -> None:
    ready = proto.WebSocketResponse(
        error=proto.WebSocketError(kind="unauthenticated", message="bad key")
    )
    async with _rime_server(_speaking_reply(), ready=ready) as (url, requests):
        service = RimeV1TTSService(api_key="bad", websocket_url=f"{url}/coda/ws")
        _, up_frames = await run_test(
            service, frames_to_send=[SleepFrame(sleep=0.05)], start_timeout=3.0
        )

    errors = [frame for frame in up_frames if isinstance(frame, ErrorFrame)]
    assert errors and "unauthenticated" in errors[0].error
    assert errors[0].category is ErrorCategory.AUTHENTICATION
    assert not requests


@pytest.mark.asyncio
async def test_next_turn_reconnects_after_rime_closes_the_socket() -> None:
    connections = 0

    async def handler(websocket) -> None:
        nonlocal connections
        connections += 1
        await websocket.send(
            proto.WebSocketResponse(ready=proto.WebSocketReady(protocol=1)).SerializeToString()
        )
        reply = _speaking_reply()
        async for message in websocket:
            request = proto.WebSocketRequest.FromString(message)
            for response in reply(request):
                await websocket.send(response.SerializeToString())
            if request.HasField("end") and connections == 1:
                await websocket.close()
                return

    async with serve(handler, "127.0.0.1", 0, subprotocols=["rime.v1.binary"]) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        service = RimeV1TTSService(api_key="key", websocket_url=f"ws://{host}:{port}/coda/ws")
        down_frames, up_frames = await run_test(
            service,
            frames_to_send=[
                TTSSpeakFrame(text="First turn."),
                SleepFrame(sleep=0.2),
                BotStoppedSpeakingFrame(),
                TTSSpeakFrame(text="Second turn."),
                SleepFrame(sleep=0.2),
                BotStoppedSpeakingFrame(),
            ],
            start_timeout=3.0,
        )

    assert connections == 2
    assert _audio_bytes(down_frames) == b"\x01\x02" * 2
    assert _texts(down_frames) == ["First turn.", "Second turn."]
