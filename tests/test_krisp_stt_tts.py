#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from websockets.protocol import State

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    InterimTranscriptionFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.krisp import common as krisp_common
from pipecat.services.krisp import stt as krisp_stt
from pipecat.services.krisp import tts as krisp_tts
from pipecat.services.krisp.common import (
    KrispServiceError,
    connect_session,
    create_session_token,
    open_session,
)
from pipecat.services.krisp.stt import KrispSTTService
from pipecat.services.krisp.tts import KrispTTSService
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.utils.errors import ErrorCategory, classify_http_exception


@pytest.fixture(autouse=True)
def _clear_token_cache():
    """Session tokens are cached per process; start every test without one."""
    krisp_common._token_cache.clear()
    yield
    krisp_common._token_cache.clear()


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
    monkeypatch.setattr(krisp_common, "create_session_token", AsyncMock(return_value="tok"))

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


@pytest.mark.asyncio
async def test_handshake_deadline_covers_the_whole_handshake(monkeypatch):
    monkeypatch.setattr(krisp_common, "HANDSHAKE_TIMEOUT", 0.2)

    class _ChattyWebsocket(_FakeWebsocket):
        async def recv(self):
            await asyncio.sleep(0.05)  # each message well within a per-message timeout
            return json.dumps({"type": "info"})

    with pytest.raises(TimeoutError):
        await open_session(_ChattyWebsocket(), {"model": "m"})


@pytest.mark.asyncio
async def test_connect_session_closes_the_socket_when_the_handshake_fails(monkeypatch):
    monkeypatch.setattr(krisp_common, "create_session_token", AsyncMock(return_value="tok"))
    ws = _FakeWebsocket([{"type": "error", "error": {"code": "x", "status_code": 400}}])
    connect = AsyncMock(return_value=ws)

    with pytest.raises(KrispServiceError):
        await connect_session(
            connect,
            "wss://x",
            api_key="k",
            usage_type="stt",
            rest_base_url="https://r",
            handshake={},
        )

    assert connect.call_args.kwargs["additional_headers"] == {"Authorization": "api-key tok"}
    ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_session_reuses_the_session_token(monkeypatch):
    create_token = AsyncMock(return_value="tok")
    monkeypatch.setattr(krisp_common, "create_session_token", create_token)
    connect = AsyncMock(side_effect=lambda *a, **kw: _FakeWebsocket([{"type": "ready"}]))

    for _ in range(3):
        await connect_session(
            connect,
            "wss://x",
            api_key="k",
            usage_type="tts",
            rest_base_url="https://r",
            handshake={},
        )

    create_token.assert_awaited_once()
    assert create_token.call_args.kwargs["expiration_ttl"] == 60
    assert connect.await_count == 3


@pytest.mark.asyncio
async def test_connect_session_replaces_a_rejected_cached_token(monkeypatch):
    create_token = AsyncMock(side_effect=["old", "new"])
    monkeypatch.setattr(krisp_common, "create_session_token", create_token)
    rejected = {"type": "error", "error": {"code": "unauthorized", "status_code": 403}}
    sockets = [
        _FakeWebsocket([{"type": "ready"}]),  # first connect caches "old"
        _FakeWebsocket([rejected]),  # "old" is rejected later
        _FakeWebsocket([{"type": "ready"}]),  # retried with "new"
    ]
    connect = AsyncMock(side_effect=sockets)
    kwargs = dict(api_key="k", usage_type="stt", rest_base_url="https://r", handshake={})

    await connect_session(connect, "wss://x", **kwargs)
    ws, _ = await connect_session(connect, "wss://x", **kwargs)

    assert ws is sockets[2]
    tokens = [c.kwargs["additional_headers"]["Authorization"] for c in connect.call_args_list]
    assert tokens == ["api-key old", "api-key old", "api-key new"]
    sockets[1].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_stt_permanent_error_mid_session_closes_the_socket():
    rejected = {"type": "error", "error": {"code": "unauthorized", "status_code": 403}}
    service = _stt([rejected, _final(" never read")])
    service.push_error = AsyncMock()

    await service._receive_messages()

    service.push_error.assert_awaited_once()
    service._websocket.close.assert_awaited_once()
    service.push_frame.assert_not_called()


# --- STT --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stt_connect_creates_stt_token_and_sends_handshake(monkeypatch):
    create_token = AsyncMock(return_value="tok")
    monkeypatch.setattr(krisp_common, "create_session_token", create_token)
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
async def test_stt_stop_pushes_the_finals_received_so_far(monkeypatch):
    service = _stt([_final(" bye")])
    await service._receive_messages()
    monkeypatch.setattr("pipecat.services.stt_service.WebsocketSTTService.stop", AsyncMock())

    await service.stop(EndFrame())

    (frame,) = _pushed(service)
    assert isinstance(frame, TranscriptionFrame) and frame.text == "bye"


@pytest.mark.asyncio
async def test_stt_settings_reconnect_pushes_the_finals_received_so_far(monkeypatch):
    service = _stt([_final(" hello")])
    await service._receive_messages()
    monkeypatch.setattr(
        "pipecat.services.stt_service.WebsocketSTTService._do_reconnect", AsyncMock()
    )

    await service._do_reconnect()

    (frame,) = _pushed(service)
    assert isinstance(frame, TranscriptionFrame) and frame.text == "hello"


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
    monkeypatch.setattr(krisp_common, "create_session_token", AsyncMock(return_value="tok"))
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


@pytest.mark.asyncio
async def test_stt_sends_each_audio_frame_as_is():
    service = _stt()
    service._sample_rate = 16000

    for chunk in (b"\x01" * 320, b"\x02" * 640):
        async for _ in service.run_stt(chunk):
            pass

    sent = [c.args[0] for c in service._websocket.send.call_args_list]
    assert sent == [b"\x01" * 320, b"\x02" * 640]


@pytest.mark.asyncio
async def test_stt_drops_audio_while_disconnected():
    service = _stt()
    service._sample_rate = 16000
    service._websocket.state = State.CLOSED

    async for _ in service.run_stt(b"\x01" * 320):
        pass

    service._websocket.send.assert_not_called()


@pytest.mark.asyncio
async def test_stt_disconnect_sends_end_of_audio_then_closes():
    service = _stt()
    ws = service._websocket

    await service._disconnect_websocket()

    assert _sent_json(ws) == [{"type": "end_of_audio"}]
    ws.close.assert_awaited_once()
    assert service._websocket is None


@pytest.mark.asyncio
async def test_stt_detected_language_does_not_carry_over_to_the_next_turn():
    service = _stt(
        [
            _final(" bonjour", language="fr"),
            {"type": "turn"},
            _final(" hello"),
            {"type": "turn"},
        ]
    )

    await service._receive_messages()

    first, second = [f for f in _pushed(service) if isinstance(f, TranscriptionFrame)]
    assert first.language == Language.FR
    assert second.language is None


@pytest.mark.asyncio
async def test_stt_string_language_setting_labels_frames():
    service = _stt([_final(" hello"), {"type": "turn"}], language="en-US")

    await service._receive_messages()

    (frame,) = [f for f in _pushed(service) if isinstance(f, TranscriptionFrame)]
    assert frame.language == Language.EN_US


# --- TTS --------------------------------------------------------------------


def _tts(messages=()) -> KrispTTSService:
    service = KrispTTSService(api_key="k")
    service._sample_rate = 24000
    service._websocket = _FakeWebsocket(messages)
    service.start_tts_usage_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.stop_all_metrics = AsyncMock()
    service.append_to_audio_context = AsyncMock()
    service.remove_audio_context = AsyncMock()
    service.audio_context_available = lambda ctx: True
    service.get_active_audio_context_id = lambda: "ctx"
    # Background tasks (the sentence-gap keepalive) aren't run unless a test opts in.
    service.create_task = lambda coro, *a, **kw: coro.close()
    return service


async def _run_tts(service, text, ctx):
    return [f async for f in service.run_tts(text, ctx)]


def _appended(service) -> list:
    return [c.args for c in service.append_to_audio_context.call_args_list]


@pytest.mark.asyncio
async def test_tts_handshake_uses_voice_language_and_speed(monkeypatch):
    create_token = AsyncMock(return_value="tok")
    monkeypatch.setattr(krisp_common, "create_session_token", create_token)
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
        "language": "de-DE",
        "speed": 1.05,
    }


@pytest.mark.asyncio
async def test_tts_is_interruptible():
    assert isinstance(KrispTTSService(api_key="k"), InterruptibleTTSService)


@pytest.mark.asyncio
async def test_tts_run_tts_sends_only_text():
    service = _tts()

    assert await _run_tts(service, "Hello there.", "ctx") == [None]

    assert _sent_json(service._websocket) == [{"type": "text", "text": "Hello there."}]


@pytest.mark.asyncio
async def test_tts_routes_audio_to_the_active_context():
    service = _tts([b"\x00\x00" * 240])

    await service._receive_messages()

    ((ctx, frame),) = _appended(service)
    assert ctx == "ctx" and isinstance(frame, TTSAudioRawFrame)
    assert frame.audio == b"\x00\x00" * 240


@pytest.mark.asyncio
async def test_tts_ends_the_context_after_flush_and_every_text_answered():
    service = _tts()
    await _run_tts(service, "Hello there.", "ctx")
    await _run_tts(service, "How are you?", "ctx")
    await service.flush_audio("ctx")
    service._websocket._messages = [json.dumps({"type": "end_of_sentence"})]

    await service._receive_messages()  # first sentence answered: not done yet
    assert _appended(service) == []

    service._websocket._messages = [
        json.dumps({"type": "end_of_sentence"}),
        json.dumps({"type": "all_processed"}),
    ]
    await service._receive_messages()

    ((ctx, frame),) = _appended(service)
    assert ctx == "ctx" and isinstance(frame, TTSStoppedFrame)
    service.remove_audio_context.assert_awaited_once_with("ctx")


@pytest.mark.asyncio
async def test_tts_flush_after_every_text_answered_ends_the_context_at_once():
    service = _tts([{"type": "end_of_sentence"}])
    await _run_tts(service, "Hello there.", "ctx")
    await service._receive_messages()
    assert _appended(service) == []

    await service.flush_audio("ctx")

    ((ctx, frame),) = _appended(service)
    assert ctx == "ctx" and isinstance(frame, TTSStoppedFrame)


@pytest.mark.asyncio
async def test_tts_disconnect_sends_end_then_closes():
    service = _tts()
    service.remove_active_audio_context = AsyncMock()
    ws = service._websocket

    await service._disconnect_websocket()

    assert _sent_json(ws) == [{"type": "end"}]
    ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_settings_change_while_speaking_waits_for_the_bot_to_stop():
    service = _tts()
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()
    service._bot_speaking = True

    await service._update_settings(KrispTTSService.Settings(voice="Clara"))
    service._disconnect.assert_not_called()

    service._bot_speaking = False
    await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_settings_change_when_idle_reconnects_at_once():
    service = _tts()
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    await service._update_settings(KrispTTSService.Settings(voice="Clara"))

    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_failed_send_undoes_its_count_and_keeps_the_context():
    service = _tts()
    service.send_with_retry = AsyncMock(side_effect=ConnectionError("down"))

    frames = await _run_tts(service, "Hello.", "ctx")

    assert [type(f) for f in frames] == [ErrorFrame, TTSStoppedFrame]
    assert service._pending_texts == 0
    service.remove_audio_context.assert_not_called()


@pytest.mark.asyncio
async def test_tts_pauses_frame_processing_while_a_reply_plays():
    assert KrispTTSService(api_key="k")._pause_frame_processing


@pytest.mark.asyncio
async def test_tts_reconnect_during_a_send_counts_only_the_resent_text():
    service = _tts()
    service._pending_texts = 2  # texts on the connection that is about to drop

    async def send_after_reconnect(message, report_error):
        service._connection_id += 1  # what _connect_websocket does on reconnect
        service._pending_texts = 0

    service.send_with_retry = send_after_reconnect

    assert await _run_tts(service, "Hello.", "ctx") == [None]
    assert service._pending_texts == 1


@pytest.mark.asyncio
async def test_tts_new_connection_resets_the_count_and_finishes_a_flushed_reply(monkeypatch):
    monkeypatch.setattr(krisp_common, "create_session_token", AsyncMock(return_value="tok"))

    async def fake_connect(url, **kwargs):
        return _FakeWebsocket([{"type": "ready"}])

    monkeypatch.setattr("pipecat.services.websocket_service.websocket_connect", fake_connect)
    service = _tts()
    service._websocket = None
    service._pending_texts = 3  # never answered: their socket is gone
    service._flushed_context_id = "ctx"

    await service._connect_websocket()

    assert service._pending_texts == 0
    ((ctx, frame),) = _appended(service)
    assert ctx == "ctx" and isinstance(frame, TTSStoppedFrame)


@pytest.mark.asyncio
async def test_stt_settings_reconnect_keeps_an_open_turn_open(monkeypatch):
    service = _stt([_partial(" hello")])
    await service._receive_messages()
    assert service._in_turn

    async def base_do_reconnect(self):
        self._reset_stream_state()  # what _disconnect() does

    monkeypatch.setattr(
        "pipecat.services.stt_service.WebsocketSTTService._do_reconnect", base_do_reconnect
    )
    await service._do_reconnect()
    assert service._in_turn

    service._websocket._messages = [json.dumps(_final(" hello")), json.dumps({"type": "turn"})]
    await service._receive_messages()
    assert _broadcast(service) == [ProposedUserStoppedSpeakingFrame]


@pytest.mark.asyncio
@pytest.mark.parametrize("make", [_stt, _tts])
async def test_connect_replaces_a_finished_receive_task(make):
    service = make()
    finished = AsyncMock()
    finished.done = lambda: True
    service._receive_task = finished
    service.cancel_task = AsyncMock()
    service._connect_websocket = AsyncMock()
    new_task = object()
    service.create_task = lambda coro, *a, **kw: (coro.close(), new_task)[1]

    await service._connect()

    service.cancel_task.assert_awaited_once_with(finished)
    assert service._receive_task is new_task


def _closing_websocket(service):
    """Make the fake socket report CLOSED once closed, like a real one."""
    ws = service._websocket

    async def close():
        ws.state = State.CLOSED

    ws.close = AsyncMock(side_effect=close)
    return ws


@pytest.mark.asyncio
async def test_tts_error_mid_reply_closes_the_session_and_ends_the_reply():
    error = {"type": "error", "error": {"code": "internal_server_error", "status_code": 500}}
    service = _tts([error, b"\x00\x00" * 240])
    service.push_error = AsyncMock()
    ws = _closing_websocket(service)
    await _run_tts(service, "Hello there.", "ctx")

    await service._receive_messages()

    service.push_error.assert_awaited_once()
    ws.close.assert_awaited_once()
    assert service._disconnecting  # the receive loop must not reconnect on its own
    ((ctx, frame),) = _appended(service)  # reply ended; later audio not read
    assert ctx == "ctx" and isinstance(frame, TTSStoppedFrame)
    service.remove_audio_context.assert_awaited_once_with("ctx")
    assert service._pending_texts == 0


@pytest.mark.asyncio
async def test_tts_next_reply_after_an_error_opens_a_fresh_session():
    service = _tts()
    _closing_websocket(service)
    await service._abandon_session()
    service._connect = AsyncMock()
    service.send_with_retry = AsyncMock()

    await _run_tts(service, "Hello again.", "ctx")

    service._connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_silent_reply_reconnects():
    service = _tts()
    ws = _closing_websocket(service)
    await _run_tts(service, "Hello.", "ctx")
    await service.flush_audio("ctx")

    await service.on_audio_context_completed("ctx")

    ws.close.assert_awaited_once()
    assert service._disconnecting


@pytest.mark.asyncio
async def test_tts_reply_with_audio_keeps_the_session():
    service = _tts([b"\x00\x00" * 240])
    ws = _closing_websocket(service)
    await _run_tts(service, "Hello.", "ctx")
    await service._receive_messages()
    await service.flush_audio("ctx")

    await service.on_audio_context_completed("ctx")

    ws.close.assert_not_called()


@pytest.mark.asyncio
async def test_tts_pause_between_sentences_keeps_the_reply_open():
    service = _tts([{"type": "end_of_sentence"}])
    service._stop_frame_timeout_s = 0.02
    service._refresh_audio_context = MagicMock()
    tasks = []
    service.create_task = lambda coro, *a, **kw: (
        tasks.append(asyncio.create_task(coro)) or tasks[-1]
    )
    await _run_tts(service, "First sentence.", "ctx")

    await service._receive_messages()  # caught up; the LLM hasn't sent more text yet
    await asyncio.sleep(0.1)
    refreshes = service._refresh_audio_context.call_count
    assert refreshes >= 3

    await service.flush_audio("ctx")  # the reply's text is complete: stop refreshing
    await asyncio.sleep(0.05)
    assert service._refresh_audio_context.call_count <= refreshes + 1
    await asyncio.wait_for(tasks[0], 1)
