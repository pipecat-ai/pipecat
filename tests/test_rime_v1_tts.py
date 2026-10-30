#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for RimeV1TTSService."""

import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
import websockets
from google.protobuf import json_format
from loguru import logger
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
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.rime.tts_v1 import RimeV1TTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
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


@pytest.fixture
def warning_log() -> Iterator[list[str]]:
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    yield messages
    logger.remove(handler_id)


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


def test_coda_uses_binary_and_public_endpoint() -> None:
    service = RimeV1TTSService(settings=RimeV1TTSService.Settings(model="coda"), api_key="key")

    assert service._websocket_url == "wss://coda.api.rime.ai/ws"
    assert service._websocket_protocol == "binary"
    assert service._settings.model == "coda"
    assert service._text_aggregation_mode is TextAggregationMode.SENTENCE


@pytest.mark.parametrize(
    "model, url",
    [
        ("coda", "wss://coda.api.rime.ai/ws"),
        ("mistv3", "wss://mist.api.rime.ai/ws"),
    ],
)
def test_model_selects_the_public_endpoint(model: str, url: str) -> None:
    service = RimeV1TTSService(api_key="key", settings=RimeV1TTSService.Settings(model=model))
    assert service._websocket_url == url
    assert service._settings.model == model


@pytest.mark.parametrize(
    "url",
    [
        "wss://coda.api.rime.ai/ws",
        "wss://mist.api.rime.ai/ws",
        "wss://voice.customer.example/custom/stream?deployment=test",
        "wss://api.rime.ai/coda/ws",
        "wss://api.rime.ai/mist/ws",
    ],
)
def test_custom_endpoint_uses_the_explicit_model(url: str) -> None:
    service = RimeV1TTSService(
        api_key="key",
        websocket_url=url,
        settings=RimeV1TTSService.Settings(model="custom-model"),
    )
    assert service._websocket_url == url
    assert service._settings.model == "custom-model"
    assert service._endpoint_model == "custom-model"


def test_custom_model_requires_an_endpoint() -> None:
    with pytest.raises(ValueError, match="websocket_url is required"):
        RimeV1TTSService(api_key="key", settings=RimeV1TTSService.Settings(model="custom-model"))


@pytest.mark.parametrize(
    "url",
    [
        None,
        "wss://coda.api.rime.ai/ws",
        "wss://mist.api.rime.ai/ws",
        "wss://voice.customer.example/custom/stream",
        "wss://api.rime.ai/coda/ws",
        "wss://api.rime.ai/mist/ws",
    ],
)
@pytest.mark.parametrize(
    "settings",
    [
        None,
        RimeV1TTSService.Settings(),
        RimeV1TTSService.Settings(model=None),
        RimeV1TTSService.Settings(model=""),
    ],
)
def test_every_endpoint_requires_an_explicit_model(
    url: str | None, settings: RimeV1TTSService.Settings | None
) -> None:
    with pytest.raises(ValueError, match="requires settings.model"):
        RimeV1TTSService(api_key="key", websocket_url=url, settings=settings)


def test_service_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="binary"):
        RimeV1TTSService(
            settings=RimeV1TTSService.Settings(model="coda"),
            api_key="key",
            websocket_protocol="xml",
        )


@pytest.mark.parametrize("url", ["https://coda.api.rime.ai/ws", "/ws", "wss:///ws"])
def test_service_rejects_non_websocket_urls(url: str) -> None:
    with pytest.raises(ValueError, match="absolute ws or wss URL"):
        RimeV1TTSService(
            api_key="key", websocket_url=url, settings=RimeV1TTSService.Settings(model="coda")
        )


@pytest.mark.parametrize(
    "url, settings, ignored",
    [
        (
            "wss://mist.api.rime.ai/ws",
            {"model": "mistv3", "text_lookahead_tokens": 4},
            "text_lookahead_tokens",
        ),
        (
            "wss://voice.customer.example/ws",
            {"model": "arcana", "text_lookahead_tokens": 4},
            "text_lookahead_tokens",
        ),
        (
            "wss://coda.api.rime.ai/ws",
            {"model": "coda", "pause_between_brackets": True},
            "pause_between_brackets",
        ),
        ("wss://coda.api.rime.ai/ws", {"model": "coda", "save_oovs": False}, "save_oovs"),
        (
            "wss://customer.rime.ai/ws",
            {"model": "mistv2", "time_scale_factor": 1.2},
            "time_scale_factor",
        ),
    ],
)
def test_settings_the_model_does_not_accept_are_left_out_with_a_warning(
    url: str, settings: dict[str, Any], ignored: str, warning_log: list[str]
) -> None:
    service = RimeV1TTSService(
        api_key="key", websocket_url=url, settings=RimeV1TTSService.Settings(**settings)
    )
    service._sample_rate = 24000

    start = service._build_start_request("turn").start
    assert not start.coda_parameters.HasField("text_lookahead_tokens")
    assert not start.HasField("mist_parameters")
    assert not start.audio_parameters.HasField("time_scale_factor")
    assert getattr(service._settings, ignored) == settings[ignored]
    assert any(ignored in message for message in warning_log)


def test_unknown_settings_are_ignored() -> None:
    service = RimeV1TTSService(
        api_key="key",
        settings=RimeV1TTSService.Settings.from_mapping(
            {"model": "coda", "repetition_penalty": 1.1}
        ),
    )
    service._sample_rate = 24000

    assert service._build_start_request("turn").start.speaker == ""


def test_coda_settings_map_to_the_start_request() -> None:
    service = RimeV1TTSService(
        api_key="key",
        sample_rate=22050,
        settings=RimeV1TTSService.Settings(
            model="coda",
            voice="astra",
            language=Language.EN,
            text_lookahead_tokens=0,
            time_scale_factor=1.2,
        ),
    )
    service._sample_rate = 22050

    start = service._build_start_request("turn").start
    assert start.speaker == "astra"
    assert start.language == "en"
    assert start.audio_parameters.audio_format == "audio/pcm"
    assert start.audio_parameters.sampling_rate == 22050
    assert start.audio_parameters.time_scale_factor == pytest.approx(1.2)
    assert start.coda_parameters.HasField("text_lookahead_tokens")
    assert start.coda_parameters.text_lookahead_tokens == 0
    assert not start.HasField("mist_parameters")


def test_token_mode_asks_rime_to_accumulate_sentences() -> None:
    sentences = RimeV1TTSService(settings=RimeV1TTSService.Settings(model="coda"), api_key="key")
    sentences._sample_rate = 24000
    assert not sentences._build_start_request("turn").start.HasField("split_strategy")

    tokens = RimeV1TTSService(
        settings=RimeV1TTSService.Settings(model="coda"),
        api_key="key",
        text_aggregation_mode=TextAggregationMode.TOKEN,
    )
    tokens._sample_rate = 24000
    start = tokens._build_start_request("turn").start
    assert start.split_strategy == proto.SPLIT_STRATEGY_ACCUMULATE_SENTENCE


@pytest.mark.asyncio
async def test_languages_map_to_bcp47_tags() -> None:
    service = RimeV1TTSService(
        api_key="key", settings=RimeV1TTSService.Settings(model="coda", language="ja-JP")
    )
    service._sample_rate = 24000
    assert service._build_start_request("turn").start.language == "ja"

    await service._update_settings(RimeV1TTSService.Settings(language=Language.DE_DE))
    assert service._build_start_request("turn").start.language == "de"


def test_mist_settings_map_to_the_start_request() -> None:
    service = RimeV1TTSService(
        api_key="key",
        websocket_url="wss://mist.api.rime.ai/ws",
        settings=RimeV1TTSService.Settings(
            model="mistv3", pause_between_brackets=True, save_oovs=False
        ),
    )
    service._sample_rate = 24000

    start = service._build_start_request("turn").start
    assert start.mist_parameters.pause_between_brackets is True
    assert start.mist_parameters.save_oovs is False
    assert not start.HasField("coda_parameters")


@pytest.mark.asyncio
async def test_runtime_settings_apply_to_the_next_context(warning_log: list[str]) -> None:
    service = RimeV1TTSService(
        api_key="key", settings=RimeV1TTSService.Settings(model="coda", voice="old")
    )
    service._sample_rate = 24000

    changed = await service._update_settings(RimeV1TTSService.Settings(voice="new"))
    assert changed == {"voice": "old"}
    assert service._build_start_request("turn").start.speaker == "new"
    assert not warning_log

    # The endpoint serves one model, so a model change is reported, not applied.
    await service._update_settings(RimeV1TTSService.Settings(model="mistv3"))
    start = service._build_start_request("turn").start
    assert not start.HasField("mist_parameters")
    assert any("mistv3 is not applied" in message for message in warning_log)

    mist = RimeV1TTSService(
        settings=RimeV1TTSService.Settings(model="mistv3"),
        api_key="key",
        websocket_url="wss://mist.api.rime.ai/ws",
    )
    mist._sample_rate = 24000
    await mist._update_settings(RimeV1TTSService.Settings(text_lookahead_tokens=4))
    assert not mist._build_start_request("turn").start.HasField("coda_parameters")
    assert any("text_lookahead_tokens" in message for message in warning_log)


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

# Fixed envelopes for each message the service sends or reads. The local server
# below shares the installed rime-api schema with the service, so only these
# catch a schema release that renumbers or renames a field.
_REQUEST_ENVELOPES = [
    (
        {
            "contextId": "ctx",
            "start": {
                "language": "en",
                "speaker": "luna",
                "text": "Hello.",
                "audioParameters": {
                    "audioFormat": "audio/pcm",
                    "samplingRate": 24000,
                    "timeScaleFactor": 1.25,
                },
                "splitStrategy": "SPLIT_STRATEGY_NONE",
                "codaParameters": {"textLookaheadTokens": 0},
            },
        },
        "0a036374781a2e1202656e1a046c756e61220648656c6c6f2e2a140a09617564696f2f70636d"
        "10c0bb011d0000a03f300242024000",
    ),
    ({"contextId": "ctx", "text": "Hello."}, "0a03637478220648656c6c6f2e"),
    ({"contextId": "ctx", "end": {}}, "0a036374782a00"),
    ({"contextId": "ctx", "cancel": {}}, "0a036374783200"),
]
_RESPONSE_ENVELOPES = [
    (
        {"ready": {"protocol": 1, "languages": ["en", "es"], "defaultLanguage": "en"}},
        "120e08011202656e120265731a02656e",
    ),
    ({"contextId": "ctx", "started": {"requestId": "req"}}, "0a036374781a050a03726571"),
    ({"contextId": "ctx", "audio": "AAH+/w=="}, "0a0363747822040001feff"),
    ({"contextId": "ctx", "done": {}}, "0a036374782a00"),
    ({"contextId": "ctx", "cancelled": {}}, "0a036374783200"),
    (
        {
            "contextId": "ctx",
            "error": {"kind": "invalid_input", "message": "test error", "requestId": "req"},
        },
        "0a036374783a200a0d696e76616c69645f696e707574120a74657374206572726f721a03726571",
    ),
]


@pytest.mark.parametrize("envelope, binary", _REQUEST_ENVELOPES)
def test_requests_match_fixed_wire_envelopes(envelope: dict[str, Any], binary: str) -> None:
    request = json_format.ParseDict(envelope, proto.WebSocketRequest())

    assert RimeV1TTSService(
        settings=RimeV1TTSService.Settings(model="coda"), api_key="key"
    )._encode(request) == bytes.fromhex(binary)
    encoded_json = RimeV1TTSService(
        settings=RimeV1TTSService.Settings(model="coda"), api_key="key", websocket_protocol="json"
    )._encode(request)
    assert json.loads(encoded_json) == envelope


@pytest.mark.parametrize("envelope, binary", _RESPONSE_ENVELOPES)
def test_responses_match_fixed_wire_envelopes(envelope: dict[str, Any], binary: str) -> None:
    from_binary = RimeV1TTSService(
        settings=RimeV1TTSService.Settings(model="coda"), api_key="key"
    )._decode(bytes.fromhex(binary))
    from_json = RimeV1TTSService(
        settings=RimeV1TTSService.Settings(model="coda"), api_key="key", websocket_protocol="json"
    )._decode(json.dumps(envelope))

    assert json_format.MessageToDict(from_binary) == envelope
    assert json_format.MessageToDict(from_json) == envelope


# ---------------------------------------------------------------------------
# Synthesis against a local Rime v1 server
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["binary", "json"])
@pytest.mark.asyncio
async def test_round_trip_streams_one_context_per_turn(protocol: str) -> None:
    async with _rime_server(_speaking_reply(b"\x01\x02\x03\x04")) as (url, requests):
        service = RimeV1TTSService(
            settings=RimeV1TTSService.Settings(model="coda"),
            api_key="test-key",
            websocket_url=f"{url}/custom/stream",
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
async def test_token_mode_streams_tokens_and_keeps_spell_tags_whole() -> None:
    async with _rime_server(_speaking_reply()) as (url, requests):
        service = RimeV1TTSService(
            settings=RimeV1TTSService.Settings(model="coda"),
            api_key="key",
            websocket_url=f"{url}/custom/stream",
            text_aggregation_mode=TextAggregationMode.TOKEN,
        )
        down_frames, up_frames = await run_test(
            service,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                *[LLMTextFrame(token) for token in ["Say ", "spe", "ll(A", "B) ", "now", "."]],
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.2),
            ],
            start_timeout=3.0,
        )

    assert _payloads(requests) == ["start", "text", "text", "text", "text", "end"]
    assert requests[0].start.split_strategy == proto.SPLIT_STRATEGY_ACCUMULATE_SENTENCE
    assert [request.text for request in requests[1:5]] == ["Say ", "spell(AB) ", "now", "."]
    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert sum(isinstance(frame, TTSStoppedFrame) for frame in down_frames) == 1


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
        service = RimeV1TTSService(
            settings=RimeV1TTSService.Settings(model="coda"),
            api_key="key",
            websocket_url=f"{url}/custom/stream",
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

    def failed_turn(first: str, second: str) -> list[Any]:
        return [
            LLMFullResponseStartFrame(),
            AggregatedTextFrame(first, AggregationType.SENTENCE),
            SleepFrame(sleep=0.1),
            AggregatedTextFrame(second, AggregationType.SENTENCE),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.1),
        ]

    async with _rime_server(reply) as (url, requests):
        service = RimeV1TTSService(
            settings=RimeV1TTSService.Settings(model="coda"),
            api_key="key",
            websocket_url=f"{url}/custom/stream",
        )
        started_at = time.monotonic()
        down_frames, up_frames = await run_test(
            service,
            frames_to_send=failed_turn("First one.", "First two.")
            + failed_turn("Second one.", "Second two."),
            start_timeout=3.0,
        )
        elapsed = time.monotonic() - started_at

    # Each turn's later sentence is skipped after Rime fails its context.
    assert _payloads(requests) == ["start", "text", "start", "text"]
    rime_errors = [f for f in up_frames if isinstance(f, ErrorFrame) and "internal" in f.error]
    assert len(rime_errors) == 2
    assert rime_errors[0].category is ErrorCategory.SERVER
    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in down_frames)
    # The failed context stays open for the rest of its turn instead of being
    # reopened empty, so each turn ends promptly with one started/stopped pair.
    assert sum(isinstance(frame, TTSStartedFrame) for frame in down_frames) == 2
    assert sum(isinstance(frame, TTSStoppedFrame) for frame in down_frames) == 2
    assert elapsed < 2.0
    assert _texts(down_frames) == ["First one.", "First two.", "Second one.", "Second two."]
    assert service.is_usable


@pytest.mark.asyncio
async def test_permanent_context_error_leaves_the_service_usable() -> None:
    def reply(request: proto.WebSocketRequest) -> list[proto.WebSocketResponse]:
        return [_error(request, "invalid_input")] if request.HasField("start") else []

    async with _rime_server(reply) as (url, _):
        service = RimeV1TTSService(
            settings=RimeV1TTSService.Settings(model="coda"),
            api_key="key",
            websocket_url=f"{url}/custom/stream",
        )
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
        service = RimeV1TTSService(
            settings=RimeV1TTSService.Settings(model="coda"),
            api_key="key",
            websocket_url=f"{url}/custom/stream",
        )
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
        service = RimeV1TTSService(
            settings=RimeV1TTSService.Settings(model="coda"),
            api_key="key",
            websocket_url=f"{url}/custom/stream",
        )
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
        service = RimeV1TTSService(
            settings=RimeV1TTSService.Settings(model="coda"),
            api_key="bad",
            websocket_url=f"{url}/custom/stream",
        )
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
        service = RimeV1TTSService(
            settings=RimeV1TTSService.Settings(model="coda"),
            api_key="key",
            websocket_url=f"ws://{host}:{port}/custom/stream",
        )
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
