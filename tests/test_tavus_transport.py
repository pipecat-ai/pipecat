#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Tavus transport."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.transports.tavus.transport import (
    TavusOutputTransport,
    TavusParams,
    TavusTransportClient,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


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


@pytest.mark.asyncio
async def test_concurrent_setup_builds_a_single_daily_client(monkeypatch):
    """The input and output transports share one client, and both set it up.

    They are set up concurrently, so a client built per caller would leave the
    losing one orphaned, with its callback tasks running and nobody to clean
    them up.
    """
    import pipecat.transports.tavus.transport as tavus

    built = []

    def fake_daily_client(*args, **kwargs):
        client = MagicMock()
        client.setup = AsyncMock()
        built.append(client)
        return client

    monkeypatch.setattr(tavus, "DailyTransportClient", fake_daily_client)

    client = TavusTransportClient(
        bot_name="Pipecat",
        callbacks=MagicMock(),
        api_key="test-key",
        replica_id="replica",
        session=MagicMock(),
    )

    conversations = []

    async def fake_initialize():
        await asyncio.sleep(0.01)  # the real one calls the Tavus API
        conversations.append("conversation")
        client._conversation_id = f"conversation-{len(conversations)}"
        return "https://example.daily.co/room"

    monkeypatch.setattr(client, "_initialize", fake_initialize)

    setup = frame_processor_setup(TaskManager())
    await asyncio.gather(client.setup(setup), client.setup(setup))

    assert len(conversations) == 1, "a Tavus conversation was created per caller"
    assert len(built) == 1, f"{len(built)} Daily clients built, so one is orphaned"
