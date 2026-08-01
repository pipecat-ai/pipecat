#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Tavus transport."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.transports.tavus.transport import TavusOutputTransport, TavusParams


def _make_output_transport(**params_kwargs) -> tuple[TavusOutputTransport, MagicMock]:
    client = MagicMock()
    client.queue_tts_frame = AsyncMock(return_value=True)
    client.send_realtime_audio_frame = AsyncMock(return_value=True)
    client.out_sample_rate = 24000
    params = TavusParams(**params_kwargs)
    return TavusOutputTransport(client, params), client


def test_audio_is_sent_faster_than_realtime_by_default():
    assert TavusParams().audio_out_faster_than_realtime is True


@pytest.mark.asyncio
async def test_default_params_send_audio_through_the_queue():
    """By default audio is queued for the send task rather than paced to playback time."""
    transport, client = _make_output_transport()

    frame = OutputAudioRawFrame(audio=b"\x00" * 960, sample_rate=24000, num_channels=1)
    assert await transport.write_audio_frame(frame) is True

    client.queue_tts_frame.assert_awaited_once_with(frame)
    client.send_realtime_audio_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_opting_out_paces_audio_to_playback_time():
    """With the flag off, each frame is sent immediately and the caller paces itself."""
    transport, client = _make_output_transport(audio_out_faster_than_realtime=False)

    frame = OutputAudioRawFrame(audio=b"\x00" * 960, sample_rate=24000, num_channels=1)
    assert await transport.write_audio_frame(frame) is True

    client.send_realtime_audio_frame.assert_awaited_once_with(frame)
    client.queue_tts_frame.assert_not_awaited()
