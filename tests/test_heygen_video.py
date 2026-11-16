#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the audio HeyGen receives from the video service and the transport."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import pipecat.services.heygen.video as heygen_video
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import OutputAudioRawFrame, TTSAudioRawFrame
from pipecat.services.heygen.client import HEY_GEN_SAMPLE_RATE
from pipecat.services.heygen.video import HeyGenVideoService
from pipecat.transports.heygen.transport import HeyGenOutputTransport, HeyGenParams


def _tone(rate: int, samples: int) -> bytes:
    t = np.arange(samples) / rate
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16).tobytes()


async def _resampled(clip: bytes, in_rate: int) -> bytes:
    """Resample a whole clip in one go, the audio HeyGen should receive."""
    resampler = create_stream_resampler(clear_after_secs=None)
    return await resampler.resample(clip, in_rate, HEY_GEN_SAMPLE_RATE) + await resampler.flush()


def _fake_client() -> tuple[MagicMock, list[bytes]]:
    spoken: list[bytes] = []
    client = MagicMock()
    client.out_sample_rate = HEY_GEN_SAMPLE_RATE
    client.agent_speak = AsyncMock(side_effect=lambda audio, event_id: spoken.append(audio))
    client.agent_speak_end = AsyncMock()
    client.interrupt = AsyncMock()
    client.start_agent_listening = AsyncMock()
    return client, spoken


def _video_service() -> tuple[HeyGenVideoService, MagicMock, list[bytes]]:
    service = HeyGenVideoService(api_key="key", session=MagicMock())
    client, spoken = _fake_client()
    service._client = client
    service._first_chunk_size = 4800
    service._chunk_size = 4800
    service._queue = asyncio.Queue()
    return service, client, spoken


@pytest.mark.asyncio
async def test_a_pause_in_a_reply_leaves_no_gap_in_the_avatar_audio(monkeypatch):
    # Long enough that the pause below doesn't end the utterance.
    monkeypatch.setattr(heygen_video, "AVATAR_VAD_STOP_SECS", 1.0)
    service, client, spoken = _video_service()
    clip = _tone(16000, 8000)
    half = len(clip) // 2
    task = asyncio.create_task(service._send_task_handler())

    await service._queue.put(TTSAudioRawFrame(clip[:half], 16000, 1))
    await asyncio.sleep(0.3)
    await service._queue.put(TTSAudioRawFrame(clip[half:], 16000, 1))
    while not client.agent_speak_end.await_count:
        await asyncio.sleep(0.05)
    task.cancel()

    assert b"".join(spoken) == await _resampled(clip, 16000)


@pytest.mark.asyncio
async def test_an_interruption_drops_the_audio_the_video_resampler_holds():
    service, _, _ = _video_service()
    service._cancel_send_task = AsyncMock()
    service._create_send_task = AsyncMock()
    await service._resampler.resample(_tone(16000, 1600), 16000, HEY_GEN_SAMPLE_RATE)

    await service._handle_user_started_speaking()

    assert await service._resampler.flush() == b""


@pytest.mark.asyncio
async def test_the_transport_sends_the_resampler_tail_before_ending_the_speech():
    client, spoken = _fake_client()
    transport = HeyGenOutputTransport(client, HeyGenParams())
    clip = _tone(16000, 1600)
    transport._event_id = "event"
    await transport.write_audio_frame(OutputAudioRawFrame(clip, 16000, 1))

    await transport._send_resampler_tail("event")

    assert b"".join(spoken) == await _resampled(clip, 16000)
