#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for writing off a TTS service that stops producing audio.

A provider can accept every request and return no audio at all — an unknown
voice ID, say — without reporting an error. TTSService counts the contexts that
complete in silence and, past a configurable limit, reports itself unable to do
its job so the pipeline worker and any ServiceSwitcher can act on it.
"""

import asyncio
from collections.abc import AsyncGenerator, Sequence

import pytest

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000

# Long enough for a context to be dequeued and played out, short enough to keep
# the tests quick: a silent context is only complete once it times out.
_STOP_FRAME_TIMEOUT_S = 0.1


class MockTTSService(TTSService):
    """HTTP-style TTS service that returns audio only for chosen utterances.

    Every other request is accepted and answered with nothing, like a provider
    given a voice it doesn't know.

    Args:
        speaking_utterances: 1-based positions of the utterances that produce
            audio. Empty means the service never speaks.
    """

    def __init__(self, speaking_utterances: set[int] | None = None, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            stop_frame_timeout_s=_STOP_FRAME_TIMEOUT_S,
            **kwargs,
        )
        self._speaking_utterances = speaking_utterances or set()
        self._utterances = 0

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self._utterances += 1
        if self._utterances in self._speaking_utterances:
            yield TTSAudioRawFrame(
                audio=_FAKE_AUDIO,
                sample_rate=_SAMPLE_RATE,
                num_channels=1,
                context_id=context_id,
            )


def _speak(*texts: str) -> list[Frame]:
    """Speak each text in its own context, waiting for each one to complete."""
    frames: list[Frame] = []
    for text in texts:
        frames.append(TTSSpeakFrame(text))
        frames.append(SleepFrame(sleep=_STOP_FRAME_TIMEOUT_S * 3))
    return frames


def _errors(up: Sequence[Frame]) -> list[ErrorFrame]:
    return [frame for frame in up if isinstance(frame, ErrorFrame)]


@pytest.mark.asyncio
async def test_silence_under_the_limit_leaves_the_service_usable():
    tts = MockTTSService(max_consecutive_zero_audio_contexts=3)

    _, up = await asyncio.wait_for(
        run_test(tts, frames_to_send=_speak("one", "two")),
        timeout=5.0,
    )

    assert tts.is_usable
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_consecutive_silent_contexts_write_off_the_service():
    tts = MockTTSService(max_consecutive_zero_audio_contexts=2)

    _, up = await asyncio.wait_for(
        run_test(tts, frames_to_send=_speak("one", "two")),
        timeout=5.0,
    )

    assert not tts.is_usable

    errors = _errors(up)
    assert len(errors) == 1
    # The processor is already written off by the time the error is seen, which
    # is what tells application code the error is not a transient one.
    assert errors[0].processor is tts
    assert not errors[0].processor.is_usable


@pytest.mark.asyncio
async def test_audio_resets_the_count():
    # Silence either side of an utterance that does produce audio: without the
    # reset, the two silent ones together would reach the limit.
    tts = MockTTSService(speaking_utterances={2}, max_consecutive_zero_audio_contexts=2)

    _, up = await asyncio.wait_for(
        run_test(tts, frames_to_send=_speak("silent", "spoken", "silent again")),
        timeout=5.0,
    )

    assert tts.is_usable
    assert tts._consecutive_zero_audio_contexts == 1
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_zero_disables_the_check():
    tts = MockTTSService(max_consecutive_zero_audio_contexts=0)

    _, up = await asyncio.wait_for(
        run_test(tts, frames_to_send=_speak("one", "two", "three", "four")),
        timeout=5.0,
    )

    assert tts.is_usable
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_the_service_is_written_off_once():
    tts = MockTTSService(max_consecutive_zero_audio_contexts=1)

    _, up = await asyncio.wait_for(
        run_test(tts, frames_to_send=_speak("one", "two", "three")),
        timeout=5.0,
    )

    # An unusable service is no longer given work, so its silent contexts say
    # nothing new and are not reported again.
    assert len(_errors(up)) == 1


@pytest.mark.asyncio
async def test_an_interrupted_context_is_not_counted():
    tts = MockTTSService(max_consecutive_zero_audio_contexts=1)

    frames: list[Frame] = [
        TTSSpeakFrame("interrupted"),
        # Interrupt while the context is still waiting for audio.
        SleepFrame(sleep=_STOP_FRAME_TIMEOUT_S / 2),
        InterruptionFrame(),
        SleepFrame(sleep=_STOP_FRAME_TIMEOUT_S * 3),
    ]

    _, up = await asyncio.wait_for(
        run_test(tts, frames_to_send=frames),
        timeout=5.0,
    )

    assert tts.is_usable
    assert tts._consecutive_zero_audio_contexts == 0
    assert _errors(up) == []


@pytest.mark.asyncio
async def test_becoming_usable_again_clears_the_count():
    tts = MockTTSService(max_consecutive_zero_audio_contexts=2)

    await asyncio.wait_for(
        run_test(tts, frames_to_send=_speak("one")),
        timeout=5.0,
    )

    assert tts._consecutive_zero_audio_contexts == 1

    await tts.set_usable(True)

    assert tts._consecutive_zero_audio_contexts == 0
