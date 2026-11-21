#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that every LLM reply ends downstream, including the ones that say nothing.

A tool-only or empty reply reaches ``TTSService`` as an ``LLMFullResponseStartFrame``
and an ``LLMFullResponseEndFrame`` with no text between them, so it opens no audio
context. Its end frame must still reach downstream processors, such as
``LLMAssistantAggregator``, which use it as the end of the bot's turn.
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
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000

# Long enough for a context to be dequeued and played out, short enough to keep
# the tests quick: a context is only complete once it times out.
_STOP_FRAME_TIMEOUT_S = 0.1


class MockTTSService(TTSService):
    """HTTP-style TTS service without word timestamps that speaks whatever reaches it."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            stop_frame_timeout_s=_STOP_FRAME_TIMEOUT_S,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


async def _run(frames: list[Frame]) -> list[Frame]:
    down, _ = await run_test(MockTTSService(), frames_to_send=frames, expected_down_frames=None)
    return list(down)


def _end_frames(frames: list[Frame]) -> list[LLMFullResponseEndFrame]:
    return [f for f in frames if isinstance(f, LLMFullResponseEndFrame)]


@pytest.mark.asyncio
async def test_a_reply_without_text_ends_downstream():
    """The reply's own end frame is pushed, so observers that track frame IDs see it."""
    end_frame = LLMFullResponseEndFrame()

    down = await _run([LLMFullResponseStartFrame(), end_frame, SleepFrame(sleep=0.3)])

    assert _end_frames(down) == [end_frame]


@pytest.mark.asyncio
async def test_a_reply_without_text_between_spoken_ones_ends_once():
    """Each reply ends exactly once, whether or not it was spoken."""
    down = await _run(
        [
            LLMFullResponseStartFrame(),
            TextFrame("First."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.3),
            LLMFullResponseStartFrame(),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.3),
            LLMFullResponseStartFrame(),
            TextFrame("Third."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.3),
        ]
    )

    boundaries = (LLMFullResponseStartFrame, LLMFullResponseEndFrame)
    assert [type(f) for f in down if isinstance(f, boundaries)] == list(boundaries) * 3


@pytest.mark.asyncio
async def test_a_speak_frame_does_not_end_a_reply():
    """A TTSSpeakFrame outside an LLM reply pushes no LLMFullResponseEndFrame."""
    down = await _run([TTSSpeakFrame("Standalone."), SleepFrame(sleep=0.3)])

    assert _end_frames(down) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
