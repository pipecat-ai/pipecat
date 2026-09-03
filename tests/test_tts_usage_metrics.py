#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that TTS usage metrics cover every text a service accepted for synthesis.

In ``TextAggregationMode.TOKEN`` the per-call metric is skipped and text
accumulates instead, so these check that each way an utterance can end settles
the accumulator, and that each utterance reports its own text.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    CancelFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.metrics.metrics import TTSUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.tests.utils import SleepFrame, run_test

_SAMPLE_RATE = 24000
_FAKE_AUDIO = b"\x00\x01" * 160
_GREETING = "Thanks for calling, how can I help?"
_RESPONSE = "Let me look that up. Here we go"
_PARAMS = PipelineParams(enable_metrics=True, enable_usage_metrics=True)


class _MockUsageTTSService(TTSService):
    """Websocket-style TTS that records the text it sends for synthesis.

    Audio arrives out of band, as it does for a service holding a persistent
    connection, so ``run_tts`` hands the text off and returns.
    """

    def __init__(self, **kwargs):
        super().__init__(push_start_frame=True, sample_rate=_SAMPLE_RATE, **kwargs)
        self.sent: list[str] = []

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.sent.append(text)
        await self.start_tts_usage_metrics(text)

        async def deliver():
            await asyncio.sleep(0.01)
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(_FAKE_AUDIO, _SAMPLE_RATE, 1, context_id=context_id),
            )

        self.create_task(deliver(), name=f"mock_usage_deliver_{context_id}")
        if False:
            yield

    async def flush_audio(self, context_id: str | None = None):
        # Without this the audio context never closes and tests that complete a
        # turn wait out the pause timeout.
        ctx = context_id or self.get_active_audio_context_id()
        if not ctx or not self.audio_context_available(ctx):
            return

        async def close():
            await asyncio.sleep(0.02)
            await self.append_to_audio_context(ctx, TTSStoppedFrame(context_id=ctx))
            await self.remove_audio_context(ctx)

        self.create_task(close(), name=f"mock_usage_close_{ctx}")


def _usage_values(frames) -> list[int]:
    return [
        data.value
        for frame in frames
        if isinstance(frame, MetricsFrame)
        for data in frame.data
        if isinstance(data, TTSUsageMetricsData)
    ]


async def _run(frames_to_send, **kwargs):
    tts = _MockUsageTTSService(text_aggregation_mode=TextAggregationMode.TOKEN)
    down, _ = await run_test(tts, frames_to_send=frames_to_send, pipeline_params=_PARAMS, **kwargs)
    return sum(len(text) for text in tts.sent), _usage_values(down)


@pytest.mark.asyncio
async def test_speak_frame_interrupted_reports_usage():
    """A TTSSpeakFrame has no terminator of its own, so its usage must not wait for one."""
    sent, usage = await _run([TTSSpeakFrame(_GREETING), SleepFrame(sleep=0.3), InterruptionFrame()])
    assert usage == [len(_GREETING)]
    assert sum(usage) == sent


@pytest.mark.asyncio
async def test_interrupted_llm_response_reports_usage():
    """Text sent before the caller barged in was still synthesized, so it is still usage."""
    sent, usage = await _run(
        [
            LLMFullResponseStartFrame(),
            TextFrame(_RESPONSE),
            SleepFrame(sleep=0.3),
            InterruptionFrame(),
        ]
    )
    assert usage == [len(_RESPONSE)]
    assert sum(usage) == sent


@pytest.mark.asyncio
async def test_cancel_settles_an_in_flight_llm_turn():
    """A cancelled session sees no EndFrame, so only cancel() can settle the accumulator."""
    sent, usage = await _run(
        [
            LLMFullResponseStartFrame(),
            TextFrame(_RESPONSE),
            SleepFrame(sleep=0.3),
            CancelFrame(),
        ],
        send_end_frame=False,
    )
    assert usage == [len(_RESPONSE)]
    assert sum(usage) == sent


@pytest.mark.asyncio
async def test_speak_frame_reports_separately_from_the_llm_turn():
    """A TTSSpeakFrame reports its own text, whether it precedes a turn or interrupts one."""
    sent, usage = await _run(
        [
            LLMFullResponseStartFrame(),
            TextFrame("Hello"),
            SleepFrame(sleep=0.3),
            TTSSpeakFrame("Wait"),
            SleepFrame(sleep=0.3),
            TextFrame(" there"),
            LLMFullResponseEndFrame(),
        ]
    )
    assert usage == [len("Wait"), len("Hello there")]
    assert sum(usage) == sent
