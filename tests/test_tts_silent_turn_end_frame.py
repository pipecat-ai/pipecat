#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that every LLM turn ends downstream, including the ones that say nothing.

With ``push_text_frames=False`` the ``LLMFullResponseEndFrame`` is not forwarded
when it arrives: ``process_frame`` holds it in
``_pending_llm_response_end_frames`` so ``_maybe_reset_word_timestamps`` can
re-push it, with the PTS of the last word frame, at the end of that turn's audio
context.

A tool-only or empty LLM response never reaches ``run_tts``, so it opens no audio
context and that method never runs for it. Without the flush in
``on_turn_context_completed`` the held frame is dropped, and any processor
downstream of the TTS — including ``LLMAssistantAggregator``, which uses the end
frame as a turn boundary — sees the turn start and never sees it end.
"""

from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000


class MockTTSService(TTSService):
    """HTTP-style TTS service that speaks whatever text reaches it."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            settings=TTSSettings(model=None, voice=None, language=None),
            **kwargs,
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        yield TTSAudioRawFrame(_FAKE_AUDIO, _SAMPLE_RATE, 1)


async def _run(frames) -> list[Frame]:
    received_down, _ = await run_test(
        MockTTSService(), frames_to_send=frames, expected_down_frames=None
    )
    return list(received_down)


def _end_frames(frames) -> list[LLMFullResponseEndFrame]:
    return [f for f in frames if isinstance(f, LLMFullResponseEndFrame)]


@pytest.mark.asyncio
async def test_audible_turn_ends_downstream():
    """The existing path: the end frame rides out with the turn's audio."""
    received = await _run(
        [
            LLMFullResponseStartFrame(),
            TextFrame("Hello there."),
            LLMFullResponseEndFrame(),
            SleepFrame(1.0),
        ]
    )

    assert len(_end_frames(received)) == 1


@pytest.mark.asyncio
async def test_tool_only_turn_ends_downstream():
    """A turn with no text opens no audio context, so nothing else re-pushes it."""
    received = await _run([LLMFullResponseStartFrame(), LLMFullResponseEndFrame(), SleepFrame(1.0)])

    assert len(_end_frames(received)) == 1


@pytest.mark.asyncio
async def test_a_silent_turn_between_audible_ones_ends_exactly_once():
    """Each turn is delivered once, and the silent one does not borrow its
    neighbour's audio context to get out."""
    received = await _run(
        [
            LLMFullResponseStartFrame(),
            TextFrame("First."),
            LLMFullResponseEndFrame(),
            SleepFrame(0.5),
            LLMFullResponseStartFrame(),
            LLMFullResponseEndFrame(),
            SleepFrame(0.5),
            LLMFullResponseStartFrame(),
            TextFrame("Third."),
            LLMFullResponseEndFrame(),
            SleepFrame(1.0),
        ]
    )

    assert len(_end_frames(received)) == 3


@pytest.mark.asyncio
async def test_a_tts_speak_frame_does_not_emit_a_turn_end():
    """TTSSpeakFrame completes a turn context too, under a fresh context ID that
    is never a key in the held-frame dict. It must not synthesise a turn end."""
    received = await _run([TTSSpeakFrame("Standalone."), SleepFrame(1.0)])

    assert _end_frames(received) == []


@pytest.mark.asyncio
async def test_the_original_end_frame_is_delivered_not_a_replacement():
    """Observers dedup by frame id, which is why the held frame is re-pushed
    rather than a fresh one being emitted."""
    end_frame = LLMFullResponseEndFrame()

    received = await _run([LLMFullResponseStartFrame(), end_frame, SleepFrame(1.0)])

    delivered = _end_frames(received)
    assert len(delivered) == 1
    assert delivered[0].id == end_frame.id
