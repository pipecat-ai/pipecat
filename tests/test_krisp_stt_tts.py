#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from websockets.protocol import State

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.krisp import stt as krisp_stt
from pipecat.services.krisp import tts as krisp_tts
from pipecat.services.krisp.api import (
    KrispServiceError,
    create_session_token,
    open_session,
    select_tts_voice,
)
from pipecat.services.krisp.stt import KrispSTTService
from pipecat.services.krisp.tts import KrispTTSService, _spread_words
from pipecat.transcriptions.language import Language
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.utils.errors import ErrorCategory, classify_http_exception


class _FakeWebsocket:
    def __init__(self, messages=()):
        self.state = State.OPEN
        self.send = AsyncMock()
        self.close = AsyncMock()
        self._messages = [m if isinstance(m, bytes) else json.dumps(m) for m in messages]

    async def recv(self):
        return self._messages.pop(0)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        while self._messages:
            yield self._messages.pop(0)


def _sent_json(ws) -> list[dict]:
    return [json.loads(c.args[0]) for c in ws.send.call_args_list if isinstance(c.args[0], str)]


def _final(*words, language=None):
    return {
        "type": "transcript",
        "transcript": {
            "isFinal": True,
            "words": [{"text": w, "start": 0, "end": 0, "language": language} for w in words],
        },
    }


def _partial(*words):
    return {
        "type": "transcript",
        "transcript": {"isFinal": False, "words": [{"text": w} for w in words]},
    }


def _stt(messages=(), **settings) -> KrispSTTService:
    service = KrispSTTService(api_key="k", settings=KrispSTTService.Settings(**settings))
    service._websocket = _FakeWebsocket(messages)
    service.push_frame = AsyncMock()
    service.broadcast_frame = AsyncMock()
    service.emit_stt_usage_metrics = AsyncMock()
    service._handle_transcription = AsyncMock()
    return service


def _pushed(service) -> list:
    return [c.args[0] for c in service.push_frame.call_args_list]


def _broadcast(service) -> list:
    return [c.args[0] for c in service.broadcast_frame.call_args_list]


# --- shared API -------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_session_raises_classified_error():
    ws = _FakeWebsocket(
        [
            {
                "type": "error",
                "error": {"code": "unauthorized", "status_code": 403, "message": "bad key"},
            }
        ]
    )

    with pytest.raises(KrispServiceError) as exc_info:
        await open_session(ws, {"model": "m"})

    assert exc_info.value.status_code == 403
    assert _sent_json(ws) == [{"model": "m"}]


@pytest.mark.asyncio
async def test_rejected_handshake_makes_the_service_unusable(monkeypatch):
    rejection = {
        "type": "error",
        "error": {"code": "invalid_api_key", "status_code": 401, "message": "bad"},
    }
    monkeypatch.setattr(krisp_stt, "create_session_token", AsyncMock(return_value="tok"))

    async def fake_connect(url, **kwargs):
        return _FakeWebsocket([rejection])

    monkeypatch.setattr("pipecat.services.websocket_service.websocket_connect", fake_connect)
    service = KrispSTTService(api_key="k")
    errors = []
    service.push_frame = AsyncMock(side_effect=lambda frame, *a, **kw: errors.append(frame))

    await service._connect_websocket()

    assert service._websocket is None
    assert errors[0].category == ErrorCategory.AUTHENTICATION
    assert not service.is_usable


async def _token_server(handler) -> TestServer:
    app = web.Application()
    app.router.add_post("/v2/sdk/tokens/session", handler)
    server = TestServer(app)
    await server.start_server()
    return server


@pytest.mark.asyncio
async def test_create_session_token_posts_usage_type():
    requests = []

    async def handler(request):
        requests.append((request.headers["Authorization"], await request.json()))
        return web.json_response({"success": True, "data": {"session_key": "sess-1"}})

    server = await _token_server(handler)
    try:
        base_url = str(server.make_url("")).rstrip("/")
        token = await create_session_token("master", "tts", expiration_ttl=30, base_url=base_url)
    finally:
        await server.close()

    assert token == "sess-1"
    assert requests == [("api-key master", {"usage_type": "tts", "expiration_ttl": 30})]


@pytest.mark.asyncio
async def test_create_session_token_raises_classified_error():
    async def handler(request):
        body = {"code": 1, "error_code": "PRODUCT_NOT_ALLOWED", "message": "no", "req_id": "r"}
        return web.json_response(body, status=403)

    server = await _token_server(handler)
    try:
        with pytest.raises(KrispServiceError) as exc_info:
            await create_session_token(
                "master", "stt", base_url=str(server.make_url("")).rstrip("/")
            )
    finally:
        await server.close()

    assert exc_info.value.code == "PRODUCT_NOT_ALLOWED"
    assert classify_http_exception(exc_info.value) == ErrorCategory.AUTHORIZATION


_CATALOG = [
    {"name": "Gero", "gender": "male", "languages": ["de-DE"]},
    {"name": "Amy", "gender": "female", "languages": ["en-US", "de-DE"]},
]


async def _voices_server(queries: list) -> TestServer:
    async def handler(request):
        queries.append(dict(request.query))
        return web.json_response({"success": True, "data": _CATALOG})

    app = web.Application()
    app.router.add_get("/v2/sdk/tts/voices", handler)
    server = TestServer(app)
    await server.start_server()
    return server


@pytest.mark.asyncio
async def test_select_tts_voice_picks_a_voice_for_the_language():
    queries = []
    server = await _voices_server(queries)
    try:
        base_url = str(server.make_url("")).rstrip("/")
        voice = await select_tts_voice("master", "de-DE", gender="male", base_url=base_url)
    finally:
        await server.close()

    assert voice == "Gero"
    assert queries == [{"model_name": "krisp-tts-v1", "language": "de-DE", "gender": "male"}]


@pytest.mark.asyncio
async def test_select_tts_voice_validates_the_preferred_voice():
    server = await _voices_server([])
    try:
        base_url = str(server.make_url("")).rstrip("/")
        assert await select_tts_voice("master", "de-DE", voice="amy", base_url=base_url) == "Amy"
        with pytest.raises(KrispServiceError, match="'Clara' does not speak de-DE"):
            await select_tts_voice("master", "de-DE", voice="Clara", base_url=base_url)
    finally:
        await server.close()


# --- STT --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stt_connect_creates_stt_token_and_sends_handshake(monkeypatch):
    create_token = AsyncMock(return_value="tok")
    monkeypatch.setattr(krisp_stt, "create_session_token", create_token)
    captured = {}

    async def fake_connect(url, **kwargs):
        captured["headers"] = kwargs["additional_headers"]
        captured["ws"] = _FakeWebsocket([{"type": "ready", "request_id": "r"}])
        return captured["ws"]

    monkeypatch.setattr("pipecat.services.websocket_service.websocket_connect", fake_connect)
    service = KrispSTTService(
        api_key="master",
        settings=KrispSTTService.Settings(
            language=Language.HY, vocabulary=["Krisp"], turn_max_silence_ms=800
        ),
    )

    await service._connect_websocket()

    assert create_token.call_args.args == ("master", "stt")
    assert captured["headers"] == {"Authorization": "api-key tok"}
    (handshake,) = _sent_json(captured["ws"])
    assert handshake == {
        "model": "krisp-realtime-stt-agents-v1",
        "sample_rate": 16000,
        "channels": 1,
        "audio_format": "pcm_s16le",
        "turn": {
            "prediction": True,
            "sensitivity": 0.9,
            "max_silence": 800,
            "silence_validation": 100,
        },
        "vocabulary": ["Krisp"],
        "language_hints": ["hy"],
    }


@pytest.mark.asyncio
async def test_stt_streams_interims_and_one_transcription_per_turn():
    service = _stt(
        [
            _partial(" what's", " the"),
            _final(" what's", language="en"),
            _partial(" the", " weather"),
            _final(" the", " weather"),
            {"type": "turn", "reason": "model", "confidence": 0.93},
        ]
    )

    await service._receive_messages()

    frames = _pushed(service)
    assert [type(f) for f in frames] == [
        InterimTranscriptionFrame,
        InterimTranscriptionFrame,
        TranscriptionFrame,
    ]
    assert [f.text for f in frames] == ["what's the", "what's the weather", "what's the weather"]
    assert frames[-1].finalized and frames[-1].language == Language.EN
    service._handle_transcription.assert_awaited_once()
    # Krisp only ends turns; starting them is left to the pipeline.
    assert _broadcast(service) == [ProposedUserStoppedSpeakingFrame]


@pytest.mark.asyncio
async def test_stt_turn_without_speech_pushes_nothing():
    service = _stt([{"type": "turn", "reason": "timeout"}])

    await service._receive_messages()

    service.push_frame.assert_not_called()
    service.broadcast_frame.assert_not_called()


@pytest.mark.asyncio
async def test_stt_end_message_flushes_pending_finals():
    service = _stt([_final(" bye"), {"type": "end"}])

    await service._receive_messages()

    (frame,) = _pushed(service)
    assert isinstance(frame, TranscriptionFrame) and frame.text == "bye"
    assert service._end_received.is_set()


@pytest.mark.asyncio
async def test_stt_without_turn_prediction_leaves_turns_to_the_pipeline():
    service = _stt([_partial(" hi"), _final(" hi"), {"type": "turn"}], turn_prediction=False)

    await service._receive_messages()

    service.broadcast_frame.assert_not_called()
    assert service.service_metadata_frame().user_turn_strategies is None
    assert isinstance(_pushed(service)[-1], TranscriptionFrame)


@pytest.mark.asyncio
async def test_stt_recommends_krisp_turn_stop_with_default_turn_start():
    service = _stt()

    strategies = service.service_metadata_frame().user_turn_strategies

    assert [type(s) for s in strategies.start] == [
        VADUserTurnStartStrategy,
        TranscriptionUserTurnStartStrategy,
    ]
    assert [type(s) for s in strategies.stop] == [ExternalUserTurnStopStrategy]


@pytest.mark.asyncio
async def test_stt_sends_80ms_chunks():
    service = _stt()
    service._sample_rate = 16000

    async for _ in service.run_stt(b"\x01" * 3000):
        pass

    (call,) = service._websocket.send.call_args_list
    assert len(call.args[0]) == 2560
    assert len(service._audio_buffer) == 440


@pytest.mark.asyncio
async def test_stt_keeps_audio_while_disconnected_and_sends_it_once_open():
    service = _stt()
    service._sample_rate = 16000
    service._websocket.state = State.CLOSED

    async for _ in service.run_stt(b"\x01" * 2560 * 2):
        pass

    service._websocket.send.assert_not_called()
    assert len(service._audio_buffer) == 2560 * 2

    service._websocket.state = State.OPEN
    async for _ in service.run_stt(b"\x02" * 2560):
        pass

    sent = [c.args[0] for c in service._websocket.send.call_args_list]
    assert sent == [b"\x01" * 2560, b"\x01" * 2560, b"\x02" * 2560]
    assert not service._audio_buffer


@pytest.mark.asyncio
async def test_stt_keeps_only_the_newest_pending_audio():
    service = _stt()
    service._sample_rate = 16000
    service._websocket.state = State.CLOSED

    async for _ in service.run_stt(b"\x00" * 2 * 16000 * 6):
        pass

    assert len(service._audio_buffer) == 2 * 16000 * 5


@pytest.mark.asyncio
async def test_stt_reconnect_pushes_pending_finals_and_keeps_the_turn_open(monkeypatch):
    service = _stt([_final(" hello")])
    await service._receive_messages()
    monkeypatch.setattr(
        "pipecat.services.stt_service.WebsocketSTTService._reconnect_websocket",
        AsyncMock(return_value=True),
    )

    assert await service._reconnect_websocket(1)

    (frame,) = _pushed(service)
    assert isinstance(frame, TranscriptionFrame) and frame.text == "hello"
    assert service._in_turn


@pytest.mark.asyncio
async def test_stt_websocket_is_exposed_only_after_ready(monkeypatch):
    monkeypatch.setattr(krisp_stt, "create_session_token", AsyncMock(return_value="tok"))
    service = KrispSTTService(api_key="k")
    seen_during_handshake = []

    class _HandshakeWebsocket(_FakeWebsocket):
        async def recv(self):
            seen_during_handshake.append(service._websocket)
            return await super().recv()

    async def fake_connect(url, **kwargs):
        return _HandshakeWebsocket([{"type": "ready"}])

    monkeypatch.setattr("pipecat.services.websocket_service.websocket_connect", fake_connect)

    await service._connect_websocket()

    assert seen_during_handshake == [None]
    assert service._websocket is not None


def _stt_ending_stream() -> KrispSTTService:
    """An STT service whose fake server answers end_of_audio with end."""
    service = _stt()
    service._sample_rate = 16000

    async def send(message):
        if isinstance(message, str) and json.loads(message)["type"] == "end_of_audio":
            service._end_received.set()

    service._websocket.send = AsyncMock(side_effect=send)
    return service


@pytest.mark.asyncio
async def test_stt_end_of_stream_flushes_in_chunks_then_blocks_audio():
    service = _stt_ending_stream()
    service._audio_buffer.extend(b"\x01" * (2560 * 2 + 100))

    await service._end_audio_stream()
    async for _ in service.run_stt(b"\x02" * 2560):
        pass
    await service._send_keepalive(b"\x00" * 3200)

    sent = [c.args[0] for c in service._websocket.send.call_args_list]
    assert [len(m) for m in sent[:3]] == [2560, 2560, 100]
    assert [json.loads(m) for m in sent[3:]] == [{"type": "end_of_audio"}]


@pytest.mark.asyncio
async def test_stt_end_of_stream_waits_for_an_in_flight_send():
    service = _stt_ending_stream()

    async with service._send_lock:
        end_task = asyncio.create_task(service._end_audio_stream())
        await asyncio.sleep(0.01)
        service._websocket.send.assert_not_called()
    await end_task

    assert service._websocket.send.await_count == 1


# --- TTS --------------------------------------------------------------------


def _tts(messages=()) -> KrispTTSService:
    service = KrispTTSService(api_key="k")
    service._sample_rate = 24000
    service._websocket = _FakeWebsocket(messages)
    service.start_tts_usage_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.stop_all_metrics = AsyncMock()
    service.append_to_audio_context = AsyncMock()
    service.add_word_timestamps = AsyncMock()
    service.remove_audio_context = AsyncMock()
    service.audio_context_available = lambda ctx: True
    return service


async def _run_tts(service, text, ctx):
    return [f async for f in service.run_tts(text, ctx)]


def _appended(service) -> list:
    return [c.args for c in service.append_to_audio_context.call_args_list]


@pytest.mark.asyncio
async def test_tts_handshake_uses_voice_language_and_speed(monkeypatch):
    create_token = AsyncMock(return_value="tok")
    monkeypatch.setattr(krisp_tts, "create_session_token", create_token)
    captured = {}

    async def fake_connect(url, **kwargs):
        captured["ws"] = _FakeWebsocket([{"type": "ready", "voice": "Gero"}])
        return captured["ws"]

    monkeypatch.setattr("pipecat.services.websocket_service.websocket_connect", fake_connect)
    service = KrispTTSService(
        api_key="k",
        settings=KrispTTSService.Settings(voice="Gero", language=Language.DE, speed=1.05),
    )

    await service._connect_websocket()

    assert create_token.call_args.args == ("k", "tts")
    (handshake,) = _sent_json(captured["ws"])
    assert handshake == {
        "model": "krisp-tts-v1",
        "voice": "Gero",
        "word_label_exporting": False,
        "language": "de-DE",
        "speed": 1.05,
    }


@pytest.mark.asyncio
async def test_tts_sends_text_then_end_of_sentence():
    service = _tts()

    assert await _run_tts(service, "Hello there.", "ctx") == [None]

    assert _sent_json(service._websocket) == [
        {"type": "text", "text": "Hello there."},
        {"type": "end_of_sentence"},
    ]


@pytest.mark.asyncio
async def test_tts_routes_audio_and_finishes_flushed_context():
    one_second = b"\x00\x00" * 24000
    service = _tts(
        [
            one_second,
            {"type": "end_of_sentence", "spoken_text": "Hi there."},
            one_second,
            {"type": "end_of_sentence", "spoken_text": "Bye."},
        ]
    )
    await _run_tts(service, "Hi there.", "ctx")
    await _run_tts(service, "Bye.", "ctx")
    await service.flush_audio("ctx")

    await service._receive_messages()

    appended = _appended(service)
    assert [type(f) for _, f in appended] == [TTSAudioRawFrame, TTSAudioRawFrame, TTSStoppedFrame]
    assert all(ctx == "ctx" for ctx, _ in appended)
    words = [c.args[0] for c in service.add_word_timestamps.call_args_list]
    assert words == [[("Hi", 0.0), ("there.", 0.25)], [("Bye.", 1.0)]]
    service.remove_audio_context.assert_awaited_once_with("ctx")


@pytest.mark.asyncio
async def test_tts_interrupt_discards_stale_audio_until_acknowledged():
    service = _tts()
    await _run_tts(service, "A long sentence.", "old")

    await service.on_audio_context_interrupted("old")
    await _run_tts(service, "New turn.", "new")
    service._websocket._messages = [
        json.dumps(m) if isinstance(m, dict) else m
        for m in [
            b"\x00\x00" * 100,  # in flight before the interrupt: discarded
            {"type": "end_of_sentence", "spoken_text": "A long"},  # interrupt ack
            b"\x00\x00" * 100,
        ]
    ]

    await service._receive_messages()

    assert {"type": "interrupt"} in _sent_json(service._websocket)
    ((ctx, frame),) = _appended(service)
    assert ctx == "new" and isinstance(frame, TTSAudioRawFrame)


@pytest.mark.asyncio
async def test_tts_interrupt_when_idle_sends_nothing():
    service = _tts()

    await service.on_audio_context_interrupted("ctx")

    service._websocket.send.assert_not_called()


def test_spread_words_by_length():
    assert _spread_words("ab cd", 1.0, 2.0) == [("ab", 1.0), ("cd", 1.5)]
    assert _spread_words("   ", 0.0, 1.0) == []


@pytest.mark.asyncio
async def test_tts_connect_replaces_a_finished_receive_task(monkeypatch):
    service = KrispTTSService(api_key="k")
    finished = AsyncMock()
    finished.done = lambda: True
    service._receive_task = finished
    service.cancel_task = AsyncMock()
    service._websocket = _FakeWebsocket()
    new_task = object()
    service.create_task = lambda coro, *a, **kw: (coro.close(), new_task)[1]

    await service._connect()

    service.cancel_task.assert_awaited_once_with(finished)
    assert service._receive_task is new_task
