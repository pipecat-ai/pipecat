#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock

import pytest
from ringglabs.stt import AckEvent, ErrorEvent, PongEvent, TranscriptEvent
from ringglabs.stt.errors import TransportError

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ringg.stt import RinggSTTParams, RinggSTTService


def _make_transcript(transcription: str, is_final: bool, language: str = "en") -> TranscriptEvent:
    return TranscriptEvent(
        type="transcript",
        transcription=transcription,
        is_final=is_final,
        language=language,
        request_id=None,
        segment_idx=None,
        segments=None,
        compute_latency_ms=None,
        audio_duration_sec=None,
        transcribed_audio_duration_sec=None,
        processing_time_ms=None,
        raw={},
    )


class _FakeSession:
    """Async-iterable stand-in for ``ringglabs`` ``AsyncStreamSession``."""

    def __init__(self, events=None):
        self._events = events or []
        self.start_speaking = AsyncMock()
        self.stop_speaking = AsyncMock()
        self.send_audio = AsyncMock()

    async def events(self):
        for e in self._events:
            yield e


def _build_service(session: _FakeSession | None) -> RinggSTTService:
    """Build a RinggSTTService without invoking ``__init__`` (skips SDK handshake)."""
    service = RinggSTTService.__new__(RinggSTTService)
    service._session = session
    service._params = RinggSTTParams()
    service._user_id = ""
    service._name = "RinggSTTService"
    return service


@pytest.mark.asyncio
async def test_final_transcript_emits_transcription_frame(monkeypatch):
    session = _FakeSession([_make_transcript("hello world", is_final=True, language="en")])
    service = _build_service(session)

    pushed: list[Frame] = []
    traced: list[tuple] = []

    async def fake_push_frame(frame):
        pushed.append(frame)

    async def fake_handle_transcription(transcript, is_final, language=None):
        traced.append((transcript, is_final, language))

    async def fake_stop_metrics():
        pass

    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "_handle_transcription", fake_handle_transcription)
    monkeypatch.setattr(service, "stop_processing_metrics", fake_stop_metrics)

    await service._receive_task_handler()

    finals = [f for f in pushed if isinstance(f, TranscriptionFrame)]
    assert len(finals) == 1
    assert finals[0].text == "hello world"
    assert traced == [("hello world", True, "en")]


@pytest.mark.asyncio
async def test_interim_transcript_emits_interim_frame(monkeypatch):
    session = _FakeSession([_make_transcript("hel", is_final=False)])
    service = _build_service(session)

    pushed: list[Frame] = []

    async def fake_push_frame(frame):
        pushed.append(frame)

    async def fake_stop_metrics():
        pass

    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "stop_processing_metrics", fake_stop_metrics)

    await service._receive_task_handler()

    assert any(isinstance(f, InterimTranscriptionFrame) for f in pushed)
    assert not any(isinstance(f, TranscriptionFrame) for f in pushed)


@pytest.mark.asyncio
async def test_empty_transcript_is_skipped(monkeypatch):
    session = _FakeSession([_make_transcript("   ", is_final=True)])
    service = _build_service(session)

    pushed: list[Frame] = []
    monkeypatch.setattr(service, "push_frame", AsyncMock(side_effect=lambda f: pushed.append(f)))
    monkeypatch.setattr(service, "stop_processing_metrics", AsyncMock())
    monkeypatch.setattr(service, "_handle_transcription", AsyncMock())

    await service._receive_task_handler()

    assert pushed == []


@pytest.mark.asyncio
async def test_error_event_pushes_error(monkeypatch):
    error = ErrorEvent(type="error", detail="boom", code="X", status_code=None, raw={})
    session = _FakeSession([error])
    service = _build_service(session)

    push_error_mock = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error_mock)
    monkeypatch.setattr(service, "push_frame", AsyncMock())
    monkeypatch.setattr(service, "stop_processing_metrics", AsyncMock())

    await service._receive_task_handler()

    push_error_mock.assert_awaited_once()
    args, kwargs = push_error_mock.call_args
    assert "boom" in args[0]
    exc = kwargs.get("exception")
    assert isinstance(exc, Exception)
    assert "boom" in str(exc)


@pytest.mark.asyncio
async def test_ack_and_pong_events_are_ignored(monkeypatch):
    session = _FakeSession(
        [
            AckEvent(type="ack", ack_for="open", state="ok", raw={}),
            PongEvent(type="pong", timestamp=0.0, raw={}),
        ]
    )
    service = _build_service(session)

    push_frame_mock = AsyncMock()
    push_error_mock = AsyncMock()
    monkeypatch.setattr(service, "push_frame", push_frame_mock)
    monkeypatch.setattr(service, "push_error", push_error_mock)
    monkeypatch.setattr(service, "stop_processing_metrics", AsyncMock())

    await service._receive_task_handler()

    push_frame_mock.assert_not_awaited()
    push_error_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_frame_forwards_vad_events(monkeypatch):
    session = _FakeSession()
    service = _build_service(session)

    # Bypass the FrameProcessor machinery — we only want to verify our isinstance
    # branches forward to the SDK session.
    async def fake_super_process(*args, **kwargs):
        pass

    monkeypatch.setattr("pipecat.services.stt_service.STTService.process_frame", fake_super_process)

    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    session.start_speaking.assert_awaited_once()
    session.stop_speaking.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_stt_swallows_transport_error(monkeypatch):
    session = _FakeSession()
    session.send_audio = AsyncMock(side_effect=TransportError("closed"))
    service = _build_service(session)

    monkeypatch.setattr(service, "start_processing_metrics", AsyncMock())
    monkeypatch.setattr(service, "stop_processing_metrics", AsyncMock())

    yielded = [f async for f in service.run_stt(b"\x00" * 160)]

    assert yielded == [None]
    session.send_audio.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_stt_skips_when_session_is_none(monkeypatch):
    service = _build_service(session=None)

    monkeypatch.setattr(service, "start_processing_metrics", AsyncMock())
    monkeypatch.setattr(service, "stop_processing_metrics", AsyncMock())

    yielded = [f async for f in service.run_stt(b"\x00" * 160)]

    assert yielded == [None]
