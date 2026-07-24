#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest
from openai.types.audio import Transcription

from pipecat.frames.frames import (
    InputAudioRawFrame,
    MetricsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.tests.utils import run_test

SAMPLE_RATE = 16000


@pytest.mark.asyncio
async def test_segment_emits_usage_and_transcription(monkeypatch):
    service = OpenAISTTService(api_key="test-key")

    async def fake_transcribe(audio: bytes) -> Transcription:
        return Transcription(text="hello world")

    monkeypatch.setattr(service, "_transcribe", fake_transcribe)

    pcm = b"\x01\x02" * SAMPLE_RATE  # 1s of 16-bit mono audio
    received_down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=pcm, sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ],
        pipeline_params=PipelineParams(enable_usage_metrics=True),
    )

    transcripts = [f for f in received_down if isinstance(f, TranscriptionFrame)]
    assert len(transcripts) == 1
    assert transcripts[0].text == "hello world"
    assert transcripts[0].finalized is True

    usage_indexes = [
        i
        for i, f in enumerate(received_down)
        if isinstance(f, MetricsFrame) and any(isinstance(d, STTUsageMetricsData) for d in f.data)
    ]
    assert len(usage_indexes) == 1
    usage_frame = received_down[usage_indexes[0]]
    usage = next(d for d in usage_frame.data if isinstance(d, STTUsageMetricsData))
    assert usage.value.audio_seconds == pytest.approx(len(pcm) / (SAMPLE_RATE * 2))

    # Usage precedes the transcript so tracing attaches it to the span the
    # finalized TranscriptionFrame closes.
    assert usage_indexes[0] < received_down.index(transcripts[0])
