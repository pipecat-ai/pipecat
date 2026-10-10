#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Zoom Scribe Live and Fast STT services."""

import json
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from websockets.protocol import State

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.services.zoom_ai.stt import (
    SEND_CHUNK_BYTES,
    ZoomScribeFastSTTService,
    ZoomScribeLiveSTTService,
)
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory


class _FakeWebsocket:
    """A websocket that replays server messages and records what was sent."""

    def __init__(self, messages=None, *, state=State.OPEN):
        self._messages = messages or []
        self.state = state
        self.sent = []
        self.closed = False

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        self.closed = True
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def _patch_connect(monkeypatch, websocket: _FakeWebsocket) -> list[dict]:
    calls = []

    async def fake_websocket_connect(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return websocket

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )
    return calls


def _live(sample_rate: int = 16000, **kwargs) -> ZoomScribeLiveSTTService:
    service = ZoomScribeLiveSTTService(api_key="test-key", **kwargs)
    # sample_rate is normally set when the pipeline starts, which these tests skip.
    service._sample_rate = sample_rate
    return service


def _record_frames(monkeypatch, service) -> list:
    frames = []
    monkeypatch.setattr(service, "push_frame", AsyncMock(side_effect=lambda f: frames.append(f)))
    monkeypatch.setattr(
        service,
        "broadcast_frame",
        AsyncMock(side_effect=lambda frame_cls, **kw: frames.append(frame_cls)),
    )
    monkeypatch.setattr(service, "emit_stt_usage_metrics", AsyncMock())
    return frames


def _event(kind: str, **fields) -> str:
    return json.dumps({"type": kind, **fields})


# Scribe Live


@pytest.mark.asyncio
async def test_connect_sends_the_documented_session_update(monkeypatch):
    websocket = _FakeWebsocket()
    calls = _patch_connect(monkeypatch, websocket)
    service = _live()

    await service._connect_websocket()

    assert calls[0]["args"][0] == "wss://api.zoom.us/v2/aiservices/scribe/live"
    assert calls[0]["kwargs"]["subprotocols"] == ["live-asr"]
    headers = calls[0]["kwargs"]["additional_headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["User-Agent"].startswith("pipecat/")
    assert json.loads(websocket.sent[0]) == {
        "type": "session.update",
        "language": "en-US",
        "audio": {"format": "pcm16"},
    }


@pytest.mark.asyncio
async def test_language_setting_maps_to_a_scribe_locale(monkeypatch):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live(settings=ZoomScribeLiveSTTService.Settings(language=Language.DE))

    await service._connect_websocket()

    assert json.loads(websocket.sent[0])["language"] == "de-DE"


@pytest.mark.asyncio
async def test_audio_is_sent_in_chunks(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket()

    async for _ in service.run_stt(b"\x01" * (SEND_CHUNK_BYTES - 2)):
        pass
    assert service._websocket.sent == []

    async for _ in service.run_stt(b"\x01" * 2):
        pass
    assert service._websocket.sent == [b"\x01" * SEND_CHUNK_BYTES]


@pytest.mark.asyncio
async def test_off_rate_audio_is_resampled_to_16_khz(monkeypatch):
    service = _live(sample_rate=8000)
    service._websocket = _FakeWebsocket()

    for _ in range(10):
        async for _ in service.run_stt(b"\x00" * 1600):
            pass

    audio_sent = b"".join(service._websocket.sent) + bytes(service._send_buffer)
    # 8 kHz in, 16 kHz out: twice the bytes, less what the resampler still holds.
    assert 0.9 < len(audio_sent) / (2 * 10 * 1600) <= 1.0


@pytest.mark.asyncio
async def test_scribe_turn_brackets_the_final_transcript(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [
            _event("session.updated"),
            _event("input_audio_buffer.speech_started", item_id="item_1", audio_start_ms=100),
            _event("input_audio_buffer.speech_stopped", item_id="item_1", audio_end_ms=900),
            _event(
                "transcription.completed",
                item_id="item_1",
                transcript="[Speaker 1] What is the capital of France?",
                audio_start_ms=100,
                audio_end_ms=900,
            ),
        ]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    assert [f if isinstance(f, type) else type(f) for f in frames] == [
        ProposedUserStartedSpeakingFrame,
        TranscriptionFrame,
        ProposedUserStoppedSpeakingFrame,
    ]
    transcript = frames[1]
    assert transcript.text == "What is the capital of France?"
    assert transcript.finalized
    assert transcript.language is Language.EN_US


@pytest.mark.asyncio
async def test_repeated_speech_start_opens_one_turn(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [
            _event("input_audio_buffer.speech_started"),
            _event("input_audio_buffer.speech_started"),
        ]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    assert frames == [ProposedUserStartedSpeakingFrame]


@pytest.mark.asyncio
async def test_usage_is_reported_before_the_transcription_frame(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket([_event("transcription.completed", transcript="Berlin.")])
    events = []
    monkeypatch.setattr(
        service, "push_frame", AsyncMock(side_effect=lambda f: events.append(type(f).__name__))
    )
    monkeypatch.setattr(
        service, "emit_stt_usage_metrics", AsyncMock(side_effect=lambda: events.append("usage"))
    )

    await service._receive_messages()

    assert events == ["usage", "TranscriptionFrame"]


@pytest.mark.asyncio
async def test_disconnect_closes_an_open_turn(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket([_event("input_audio_buffer.speech_started")])
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()
    await service._disconnect_websocket()

    assert frames == [ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, category",
    [("invalid_config", ErrorCategory.INVALID_REQUEST), ("internal", ErrorCategory.UNKNOWN)],
)
async def test_fatal_error_event_is_reported(monkeypatch, code, category):
    service = _live()
    service._websocket = _FakeWebsocket(
        [_event("error", error={"code": code, "message": "bad things", "fatal": True})]
    )
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)

    await service._receive_messages()

    assert "bad things" in push_error.await_args.kwargs["error_msg"]
    assert push_error.await_args.kwargs["category"] == category


@pytest.mark.asyncio
async def test_non_fatal_error_event_is_only_logged(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [_event("error", error={"code": "audio_gap", "message": "late audio", "fatal": False})]
    )
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)

    await service._receive_messages()

    push_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_settings_change_reconnects(monkeypatch):
    service = _live()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(ZoomScribeLiveSTTService.Settings(language=Language.JA))

    assert service._settings.language == "ja-JP"
    reconnect.assert_awaited_once()


@pytest.mark.parametrize("should_interrupt", [True, False])
def test_metadata_recommends_external_turn_strategies(should_interrupt):
    service = _live(should_interrupt=should_interrupt)

    strategies = service.service_metadata_frame().user_turn_strategies

    assert isinstance(strategies, ExternalUserTurnStrategies)
    assert strategies.enable_interruptions is should_interrupt
    assert service.supports_ttfs is False


@pytest.mark.asyncio
async def test_regional_variant_falls_back_to_the_base_locale(monkeypatch):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live(settings=ZoomScribeLiveSTTService.Settings(language=Language.EN_GB))

    await service._connect_websocket()

    assert json.loads(websocket.sent[0])["language"] == "en-US"


@pytest.mark.asyncio
async def test_stop_flushes_audio_and_closes_the_session(monkeypatch):
    service = _live()
    websocket = _FakeWebsocket()
    service._websocket = websocket
    service._send_buffer.extend(b"\x02" * 10)
    monkeypatch.setattr(
        "pipecat.services.stt_service.WebsocketSTTService.stop", AsyncMock(return_value=None)
    )

    await service.stop(EndFrame())

    assert service._disconnecting
    assert websocket.sent == [b"\x02" * 10, json.dumps({"type": "session.close"})]


@pytest.mark.asyncio
async def test_disconnect_reports_once(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket()
    on_disconnected = AsyncMock()
    monkeypatch.setattr(service, "_call_event_handler", on_disconnected)

    await service._disconnect_websocket()
    await service._disconnect_websocket()

    on_disconnected.assert_awaited_once_with("on_disconnected")


@pytest.mark.asyncio
async def test_failed_first_connect_still_starts_the_receive_loop(monkeypatch):
    async def refuse(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr("pipecat.services.websocket_service.websocket_connect", refuse)
    service = _live()
    monkeypatch.setattr(service, "push_error", AsyncMock())
    started = []
    monkeypatch.setattr(
        service, "create_task", lambda coro, name=None: started.append(name) or coro.close()
    )

    await service._connect()

    assert service._websocket is None
    # The receive loop owns reconnecting, so it must run even without a socket.
    assert "receive" in started


# Scribe Fast


class _ScribeFast:
    """A local HTTP server standing in for Scribe Fast."""

    def __init__(self, status: int = 200, body: dict | None = None):
        self.status = status
        self.body = body
        self.requests: list[dict] = []

    async def handle(self, request: web.Request) -> web.Response:
        form = await request.post()
        upload = form["file"]
        assert isinstance(upload, web.FileField)
        self.requests.append(
            {
                "headers": dict(request.headers),
                "config": json.loads(str(form["config"])),
                "filename": upload.filename,
                "content_type": upload.content_type,
                "audio": upload.file.read(),
            }
        )
        return web.json_response(self.body, status=self.status)


async def _run_fast(fake: _ScribeFast, **kwargs) -> list:
    app = web.Application()
    app.router.add_post("/transcribe", fake.handle)
    server = TestServer(app)
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            service = ZoomScribeFastSTTService(
                api_key="test-key",
                url=str(server.make_url("/transcribe")),
                aiohttp_session=session,
                **kwargs,
            )
            service.start_processing_metrics = AsyncMock()
            service.stop_processing_metrics = AsyncMock()
            frames = [frame async for frame in service.run_stt(b"RIFF0000WAVE")]
            service.stop_processing_metrics.assert_awaited_once()
            return frames
    finally:
        await server.close()


_FAST_RESULT = {
    "request_id": "req_1",
    "result": {"text_display": "Hi there! What is the capital of France?", "segments": []},
}


@pytest.mark.asyncio
async def test_fast_uploads_the_utterance_and_config():
    fake = _ScribeFast(body=_FAST_RESULT)

    frames = await _run_fast(fake)

    [request] = fake.requests
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["headers"]["User-Agent"].startswith("pipecat/")
    assert (request["filename"], request["content_type"]) == ("utterance.wav", "audio/wav")
    assert request["audio"] == b"RIFF0000WAVE"
    assert request["config"] == {"language": "en-US"}

    [frame] = frames
    assert isinstance(frame, TranscriptionFrame)
    assert frame.text == "Hi there! What is the capital of France?"
    assert frame.language is Language.EN_US


@pytest.mark.asyncio
async def test_fast_diarization_and_language_are_sent():
    fake = _ScribeFast(body=_FAST_RESULT)

    await _run_fast(
        fake, settings=ZoomScribeFastSTTService.Settings(language=Language.FR, diarization=True)
    )

    assert fake.requests[0]["config"] == {"language": "fr-FR", "diarization": True}


@pytest.mark.asyncio
async def test_fast_empty_transcript_yields_nothing():
    assert await _run_fast(_ScribeFast(body={"result": {"text_display": ""}})) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, category",
    [
        (401, ErrorCategory.AUTHENTICATION),
        (429, ErrorCategory.RATE_LIMIT),
        (503, ErrorCategory.SERVER),
    ],
)
async def test_fast_http_error_is_categorized(status, category):
    [frame] = await _run_fast(_ScribeFast(status=status, body={"code": 124, "message": "nope"}))

    assert isinstance(frame, ErrorFrame)
    assert str(status) in frame.error
    assert frame.category == category


@pytest.mark.asyncio
async def test_fast_closes_only_a_session_it_created():
    async with aiohttp.ClientSession() as injected:
        service = ZoomScribeFastSTTService(api_key="test-key", aiohttp_session=injected)
        await service.cleanup()
        assert not injected.closed

    service = ZoomScribeFastSTTService(api_key="test-key")
    service._session = aiohttp.ClientSession()
    created = service._session
    await service.cleanup()
    assert created.closed
