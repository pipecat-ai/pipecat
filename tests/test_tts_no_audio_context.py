#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that TTS text never rendered as audio is not written to the LLM context.

Covers the failure mode of a websocket TTS whose socket is open but dead (e.g. a
TCP connection killed by a keepalive timeout): ``run_tts()`` writes to the socket
and returns cleanly, but no audio ever arrives on the receive loop. The audio
context drains via the stop-frame timeout and force-complete emits the un-rendered
remainder. That remainder must not reach the LLM context — the user never heard it.

See https://github.com/pipecat-ai/pipecat/issues/5305.
"""

from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000

# Outlast the audio-context stop-frame timeout (default 3.0s) so the context
# drains and force-complete runs, as it does on a live dropped socket.
_DRAIN_SLEEP = 5.0


class _SilentWebsocketTTS(TTSService):
    """A websocket-shaped TTS whose socket is open but dead.

    Same shape as a word-timestamp websocket service: ``push_text_frames=False``,
    ``push_start_frame=True``, and ``run_tts()`` only writes to the socket and
    yields ``None`` — audio would arrive later on a separate receive loop. Here
    that loop never fires, which is what a keepalive-timeout kill looks like
    from the service's point of view: ``send()`` succeeded.
    """

    def __init__(self, **kwargs):
        super().__init__(
            push_text_frames=False,
            push_start_frame=True,
            pause_frame_processing=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        await self.start_ttfb_metrics()
        yield None


class _HalfDeadWebsocketTTS(_SilentWebsocketTTS):
    """Speaks the first sentence normally, then the socket goes dead."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self._calls += 1
        if self._calls == 1:
            # The base class created the audio context (push_start_frame=True);
            # deliver one audio chunk the way a websocket receive loop would.
            await self.append_to_audio_context(
                context_id, TTSAudioRawFrame(_FAKE_AUDIO, _SAMPLE_RATE, 1)
            )
        yield None


@pytest.mark.asyncio
async def test_unspoken_text_is_not_written_to_context():
    """A turn that produced zero audio must not add an assistant message to context."""
    context = LLMContext()
    pipeline = Pipeline([_SilentWebsocketTTS(), LLMAssistantAggregator(context)])

    down, up = await run_test(
        pipeline,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("What do you usually do in your room to relax?"),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=_DRAIN_SLEEP),
        ],
    )

    assert not any(isinstance(f, (TTSAudioRawFrame, BotStartedSpeakingFrame)) for f in down)
    messages = context.get_messages()
    assert not any(m.get("role") == "assistant" for m in messages), (
        f"text that was never spoken reached the LLM context: {messages}"
    )


@pytest.mark.asyncio
async def test_unspoken_remainder_frame_is_marked_no_context():
    """The force-completed remainder TTSTextFrame carries append_to_context=False."""
    tts = _SilentWebsocketTTS()

    down, _ = await run_test(
        tts,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello there."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=_DRAIN_SLEEP),
        ],
    )

    word_frames = [f for f in down if isinstance(f, TTSTextFrame)]
    assert word_frames, "force-complete should still emit the remainder TTSTextFrame"
    assert all(not f.append_to_context for f in word_frames), (
        "remainder text of a zero-audio context must not be marked for the LLM context"
    )


@pytest.mark.asyncio
async def test_partial_audio_context_still_appends():
    """A context that rendered any audio keeps today's force-complete behavior.

    Distinguishing 'audio played but word events were dropped' (remainder was
    probably spoken) from 'stream died mid-utterance' (remainder was not) is not
    possible from frame flow alone, so a context with audio keeps appending its
    force-completed remainder to the LLM context.
    """
    context = LLMContext()
    pipeline = Pipeline([_HalfDeadWebsocketTTS(), LLMAssistantAggregator(context)])

    await run_test(
        pipeline,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("First sentence. Second sentence."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=_DRAIN_SLEEP),
        ],
    )

    messages = context.get_messages()
    assert any(m.get("role") == "assistant" for m in messages)
