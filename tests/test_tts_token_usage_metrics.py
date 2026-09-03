#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""TOKEN-mode TTS usage metrics flush on interruption as well as on LLM/end frames."""

from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.metrics.metrics import TTSUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.tests.utils import run_test

_SAMPLE_RATE = 16000
_FAKE_AUDIO = b"\x00\x01" * 320


pytestmark = pytest.mark.asyncio


class _TokenTTSService(TTSService):
    def __init__(self, **kwargs):
        super().__init__(
            text_aggregation_mode=TextAggregationMode.TOKEN,
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=_SAMPLE_RATE,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield TTSAudioRawFrame(
            audio=_FAKE_AUDIO,
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            context_id=context_id,
        )


def _tts_usage_values(frames: list[Frame]) -> list[int]:
    values: list[int] = []
    for frame in frames:
        if isinstance(frame, MetricsFrame):
            for data in frame.data:
                if isinstance(data, TTSUsageMetricsData):
                    values.append(data.value)
    return values


async def test_ttsspeak_usage_flushed_on_interruption():
    text = "Thanks for calling, I'm an AI assistant."
    tts = _TokenTTSService()
    down, _up = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(text=text, append_to_context=False), InterruptionFrame()],
        pipeline_params=PipelineParams(enable_usage_metrics=True),
        start_timeout=5.0,
    )
    assert _tts_usage_values(down) == [len(text)]


async def test_llm_token_usage_flushed_on_response_end():
    text = "Hello there"
    tts = _TokenTTSService()
    down, _up = await run_test(
        tts,
        frames_to_send=[
            LLMFullResponseStartFrame(),
            TextFrame(text=text),
            LLMFullResponseEndFrame(),
        ],
        pipeline_params=PipelineParams(enable_usage_metrics=True),
        start_timeout=5.0,
    )
    assert _tts_usage_values(down) == [len(text)]


async def test_interruption_without_streamed_text_emits_no_tts_usage():
    tts = _TokenTTSService()
    down, _up = await run_test(
        tts,
        frames_to_send=[InterruptionFrame()],
        pipeline_params=PipelineParams(enable_usage_metrics=True),
        start_timeout=5.0,
    )
    assert _tts_usage_values(down) == []
