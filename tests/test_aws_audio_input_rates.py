#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the rate the AWS speech services send the user's audio at."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from websockets.protocol import State

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.services.aws import stt as aws_stt
from pipecat.services.aws.stt import AWSTranscribeSTTService


def _tone(rate: int, samples: int) -> bytes:
    t = np.arange(samples) / rate
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16).tobytes()


async def _resampled(audio: bytes, in_rate: int, out_rate: int) -> bytes:
    return await create_stream_resampler().resample(audio, in_rate, out_rate)


async def _transcribe_sends(monkeypatch, sample_rate: int, audio: bytes) -> list[bytes]:
    sent: list[bytes] = []
    monkeypatch.setattr(aws_stt, "build_event_message", lambda chunk: sent.append(chunk) or b"")
    service = AWSTranscribeSTTService(api_key="secret", aws_access_key_id="key")
    service._sample_rate = sample_rate
    service._websocket = MagicMock(state=State.OPEN, send=AsyncMock())
    async for _ in service.run_stt(audio):
        pass
    return sent


@pytest.mark.asyncio
async def test_transcribe_resamples_a_rate_it_does_not_take_to_16_khz(monkeypatch):
    clip = _tone(24000, 4800)

    sent = await _transcribe_sends(monkeypatch, 24000, clip)

    assert sent == [await _resampled(clip, 24000, 16000)]


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [8000, 16000])
async def test_transcribe_sends_8_and_16_khz_as_is(monkeypatch, rate):
    clip = _tone(rate, rate // 10)

    sent = await _transcribe_sends(monkeypatch, rate, clip)

    assert sent == [clip]


@pytest.mark.asyncio
async def test_nova_sonic_sends_user_audio_at_the_declared_input_rate():
    nova_sonic = pytest.importorskip("pipecat.services.aws.nova_sonic.llm")
    service = nova_sonic.AWSNovaSonicLLMService(
        secret_access_key="test", access_key_id="test", region="us-east-1"
    )
    service._send_user_audio_event = AsyncMock()
    clip = _tone(24000, 4800)

    await service._handle_input_audio_frame(InputAudioRawFrame(clip, 24000, 1))

    service._send_user_audio_event.assert_awaited_once_with(await _resampled(clip, 24000, 16000))
