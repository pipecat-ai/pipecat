#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Playback context attribution for bot start events."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterruptionFrame,
    SpeechOutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.frame_queue import FrameQueue


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", [None, "speaker"])
@pytest.mark.parametrize("first_samples", [3, 160])
async def test_context_follows_playback_and_survives_starvation(destination, first_samples):
    """Queued contexts and flushed audio retain their owner across starvation stops."""
    params = TransportParams(audio_out_enabled=True)
    transport = BaseOutputTransport(params)
    transport.push_frame = AsyncMock()
    sender = BaseOutputTransport.MediaSender(
        transport, destination=destination, sample_rate=16000, audio_chunk_size=320, params=params
    )
    sender._audio_queue = asyncio.Queue()
    sender._audio_buffer = bytearray()
    transport._media_senders[destination] = sender
    try:
        for owner, samples in (("first", first_samples), ("second", 160)):
            for frame in (
                TTSStartedFrame(context_id=owner),
                TTSAudioRawFrame(
                    audio=b"\x01\x00" * samples,
                    sample_rate=16000,
                    num_channels=1,
                    context_id=owner,
                ),
                TTSStoppedFrame(context_id=owner),
            ):
                frame.transport_destination = destination
                await transport.process_frame(frame, FrameDirection.DOWNSTREAM)

        assert sender._tts_context_id is None
        resumed = False
        while not sender._audio_queue.empty():
            frame = sender._audio_queue.get_nowait()
            await sender._handle_frame(frame)
            if isinstance(frame, TTSAudioRawFrame) and not resumed:
                await sender._bot_stopped_speaking()
                await sender._handle_frame(frame)
                resumed = True

        assert sender._tts_context_id is None
        await sender._handle_frame(
            SpeechOutputAudioRawFrame(audio=b"\xff\x7f" * 160, sample_rate=16000, num_channels=1)
        )
        starts = [
            call.args[0]
            for call in transport.push_frame.call_args_list
            if isinstance(call.args[0], BotStartedSpeakingFrame)
        ]
        assert [frame.context_id for frame in starts] == ["first"] * 4 + ["second"] * 2 + [None] * 2
        assert all(frame.transport_destination == destination for frame in starts)
    finally:
        await sender.cleanup()


@pytest.mark.asyncio
async def test_interruption_clears_playback_context():
    """Speech after an interruption cannot inherit the interrupted TTS context."""
    transport = BaseOutputTransport(TransportParams(audio_out_enabled=True))
    transport.push_frame = AsyncMock()
    sender = BaseOutputTransport.MediaSender(
        transport,
        destination=None,
        sample_rate=16000,
        audio_chunk_size=320,
        params=transport._params,
    )
    sender._create_audio_task = lambda: None
    sender._create_video_task = lambda: None
    sender._create_clock_task = lambda: None
    sender._audio_queue = FrameQueue()
    try:
        await sender._handle_frame(TTSStartedFrame(context_id="interrupted"))
        await sender.handle_interruptions(InterruptionFrame())
        await sender._bot_started_speaking()
        assert all(call.args[0].context_id is None for call in transport.push_frame.call_args_list)
    finally:
        await sender.cleanup()
