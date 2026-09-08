#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the xAI streaming STT service connection parameters."""

import asyncio
from urllib.parse import parse_qs, urlparse

from pipecat.frames.frames import TranscriptionFrame
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


def _collect_finals(service: XAISTTService) -> list[str]:
    """Record the text of every TranscriptionFrame the service pushes."""
    pushed: list[str] = []

    async def fake_push_frame(frame, direction=None):
        if isinstance(frame, TranscriptionFrame):
            pushed.append(frame.text)

    service.push_frame = fake_push_frame

    async def noop(*args, **kwargs):
        pass

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


def test_cumulative_speech_final_is_not_emitted_a_second_time():
    """xAI restates the whole utterance at speech_final. Emitting that as
    another TranscriptionFrame hands every consumer that accumulates frame
    text the utterance twice."""
    service = XAISTTService(api_key="test-key")
    pushed = _collect_finals(service)

    _feed(service, [
        _partial("my order"),
        _partial("my order number", is_final=True),
        _partial("is four"),
        _partial("is four two one.", is_final=True),
        # The restatement, re-punctuated: "one." above appears as "one"
        # here, so a raw equality or startswith check does not see it.
        _partial("my order number is four two one.", is_final=True, speech_final=True),
    ])

    assert pushed == ["my order number", "is four two one."]


def test_a_speech_final_with_new_words_keeps_them():
    # Only the already-emitted prefix is dropped, never trailing content.
    service = XAISTTService(api_key="test-key")
    pushed = _collect_finals(service)

    _feed(service, [
        _partial("hello", is_final=True),
        _partial("hello there friend", is_final=True, speech_final=True),
    ])

    assert pushed == ["hello", "there friend"]


def test_a_speech_final_that_is_the_whole_utterance_is_emitted():
    # No segment finals came first, so nothing has been said yet.
    service = XAISTTService(api_key="test-key")
    pushed = _collect_finals(service)

    _feed(service, [_partial("hello there", is_final=True, speech_final=True)])

    assert pushed == ["hello there"]


def test_unrelated_speech_final_text_passes_through_whole():
    # Not a restatement of what was emitted: pass it through rather than
    # guess at a split point.
    service = XAISTTService(api_key="test-key")
    pushed = _collect_finals(service)

    _feed(service, [
        _partial("first thing", is_final=True),
        _partial("something else entirely", is_final=True, speech_final=True),
    ])

    assert pushed == ["first thing", "something else entirely"]


def test_state_is_cleared_between_utterances():
    """A completed utterance must not subtract its text from the next one:
    "yes" precedes "yes I do", and carrying the prefix across would strip
    the second utterance down to " I do"."""
    service = XAISTTService(api_key="test-key")
    pushed = _collect_finals(service)

    _feed(service, [
        _partial("yes", is_final=True),
        _partial("yes", is_final=True, speech_final=True),
        # Second utterance, starting with the same word.
        _partial("yes I do", is_final=True),
        _partial("yes I do.", is_final=True, speech_final=True),
    ])

    assert pushed == ["yes", "yes I do"]


def test_an_unrecognized_restatement_is_never_truncated():
    """An utterance can end without a speech_final -- an endpointing
    timeout, a dropped connection -- leaving no boundary to reset on. The
    de-duplication then cannot recognize the next restatement, and must pass
    it through whole: duplicated text is visible and recoverable, silently
    dropped text is neither."""
    service = XAISTTService(api_key="test-key")
    pushed = _collect_finals(service)

    _feed(service, [
        # First utterance ends after a segment final, with no speech_final.
        _partial("yes", is_final=True),
        # Second utterance: its restatement no longer matches the prefix.
        _partial("I do", is_final=True),
        _partial("I do.", is_final=True, speech_final=True),
    ])

    assert pushed == ["yes", "I do", "I do."]
    assert "".join(pushed).count("I do") == 2, "content duplicated, not lost"
