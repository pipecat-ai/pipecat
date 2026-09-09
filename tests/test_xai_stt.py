#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the xAI streaming STT service."""

import asyncio
from urllib.parse import parse_qs, urlparse

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.xai.stt import XAISTTService
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


def _query(service: XAISTTService) -> dict[str, list[str]]:
    """Build the WebSocket URL and return its parsed query parameters."""
    return parse_qs(urlparse(service._build_ws_url()).query)


def _setup_service(service: XAISTTService, monkeypatch, sample_rate: int) -> None:
    """Set the service up with the given input sample rate, without connecting."""

    async def fake_connect():
        pass

    monkeypatch.setattr(service, "_connect", fake_connect)

    async def run():
        await service.setup(frame_processor_setup(TaskManager(), audio_in_sample_rate=sample_rate))

    asyncio.run(run())


def test_sample_rate_inherits_setup_when_omitted(monkeypatch):
    service = XAISTTService(api_key="test-key")

    _setup_service(service, monkeypatch, 8000)

    assert service.sample_rate == 8000
    assert _query(service)["sample_rate"] == ["8000"]


def test_explicit_sample_rate_overrides_setup(monkeypatch):
    service = XAISTTService(api_key="test-key", sample_rate=16000)

    _setup_service(service, monkeypatch, 8000)

    assert service.sample_rate == 16000
    assert _query(service)["sample_rate"] == ["16000"]


def _collect_frames(service: XAISTTService) -> list[Frame]:
    """Record every frame the service pushes, bypassing metrics and tracing."""
    pushed: list[Frame] = []

    async def fake_push_frame(frame, direction=None):
        pushed.append(frame)

    async def noop(*args, **kwargs):
        pass

    service.push_frame = fake_push_frame
    service.emit_stt_usage_metrics = noop
    service._trace_transcription = noop
    return pushed


def _feed(service: XAISTTService, messages: list[dict]) -> None:
    async def run():
        for message in messages:
            await service._handle_message(message)

    asyncio.run(run())


def _partial(text: str, *, is_final: bool = False, speech_final: bool = False) -> dict:
    return {
        "type": "transcript.partial",
        "text": text,
        "is_final": is_final,
        "speech_final": speech_final,
    }


def _describe(frames: list[Frame]) -> list[tuple[str, str]]:
    return [(type(frame).__name__, frame.text) for frame in frames]


def test_utterance_final_is_the_only_transcription_frame():
    """The utterance final restates every chunk final, so it is the one
    TranscriptionFrame for the utterance; chunk finals are interims."""
    service = XAISTTService(api_key="test-key")
    pushed = _collect_frames(service)

    _feed(
        service,
        [
            _partial("my order"),
            _partial("my order number", is_final=True),
            _partial("is four"),
            _partial("is four two one.", is_final=True),
            _partial("my order number is four two one.", is_final=True, speech_final=True),
        ],
    )

    assert _describe(pushed) == [
        ("InterimTranscriptionFrame", "my order"),
        ("InterimTranscriptionFrame", "my order number"),
        ("InterimTranscriptionFrame", "is four"),
        ("InterimTranscriptionFrame", "is four two one."),
        ("TranscriptionFrame", "my order number is four two one."),
    ]
    assert pushed[-1].finalized is True


def test_each_utterance_gets_its_own_transcription_frame():
    """Utterance finals are scoped to one utterance, not cumulative across
    the session, so back-to-back utterances yield one frame each."""
    service = XAISTTService(api_key="test-key")
    pushed = _collect_frames(service)

    _feed(
        service,
        [
            _partial("My order number is", is_final=False),
            _partial("My order number is four two one.", is_final=True),
            _partial("My order number is four two one.", is_final=True, speech_final=True),
            _partial("And I would like to know when it will ship.", is_final=True),
            _partial(
                "And I would like to know when it will ship.", is_final=True, speech_final=True
            ),
        ],
    )

    finals = [frame for frame in pushed if isinstance(frame, TranscriptionFrame)]
    assert [frame.text for frame in finals] == [
        "My order number is four two one.",
        "And I would like to know when it will ship.",
    ]


def test_empty_finals_are_ignored():
    """The server sends an empty chunk final and an empty transcript.done
    after the stream ends; neither produces a frame."""
    service = XAISTTService(api_key="test-key")
    pushed = _collect_frames(service)

    _feed(
        service,
        [
            _partial("", is_final=True),
            {"type": "transcript.done", "text": "", "duration": 5.6},
        ],
    )

    assert pushed == []


def test_transcript_done_with_text_is_a_transcription_frame():
    service = XAISTTService(api_key="test-key")
    pushed = _collect_frames(service)

    _feed(service, [{"type": "transcript.done", "text": "trailing words", "duration": 1.0}])

    assert _describe(pushed) == [("TranscriptionFrame", "trailing words")]
    assert pushed[0].finalized is True
