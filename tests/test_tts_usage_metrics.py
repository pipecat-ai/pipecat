#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    CancelFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TTSSpeakFrame,
)
from pipecat.metrics.metrics import TTSUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.tests.utils import SleepFrame, run_test

GREETING = "Thanks for calling. How can I help?"


class FakeTTSService(TTSService):
    """TTS service that streams tokens and reports usage as real services do."""

    def __init__(self):
        super().__init__(
            text_aggregation_mode=TextAggregationMode.TOKEN,
            settings=TTSSettings(model=None, voice=None, language=None),
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        await self.start_tts_usage_metrics(text)
        yield None


async def _reported_characters(frames: list[Frame]) -> list[int]:
    received_down, _ = await run_test(
        FakeTTSService(),
        frames_to_send=frames,
        pipeline_params=PipelineParams(enable_usage_metrics=True),
        send_end_frame=False,
    )
    return [
        data.value
        for frame in received_down
        if isinstance(frame, MetricsFrame)
        for data in frame.data
        if isinstance(data, TTSUsageMetricsData)
    ]


@pytest.mark.asyncio
async def test_interrupted_text_usage_is_reported_once():
    characters = await _reported_characters(
        [
            TTSSpeakFrame(GREETING),
            SleepFrame(sleep=0.1),
            InterruptionFrame(),
            SleepFrame(sleep=0.1),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Sure, "),
            SleepFrame(sleep=0.1),
            InterruptionFrame(),
            SleepFrame(sleep=0.1),
            CancelFrame(),
        ]
    )

    assert characters == [len(GREETING), len("Sure, ")]


@pytest.mark.asyncio
async def test_cancelled_text_usage_is_reported():
    characters = await _reported_characters(
        [TTSSpeakFrame(GREETING), SleepFrame(sleep=0.1), CancelFrame()]
    )

    assert characters == [len(GREETING)]
