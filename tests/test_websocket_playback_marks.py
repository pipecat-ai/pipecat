#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Playback marks over a FastAPI websocket: sent through the serializer, pushed on its echo."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import CancelFrame, PlaybackMarkFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketOutputTransport,
    FastAPIWebsocketParams,
)
from pipecat.utils.asyncio.task_manager import TaskManager

STREAM_SID = "MZ123"


async def _make_output(serializer, connected: bool = True) -> FastAPIWebsocketOutputTransport:
    client = MagicMock(is_closing=False, is_connected=connected)
    client.setup = AsyncMock()
    client.send = AsyncMock()
    client.disconnect = AsyncMock()
    params = FastAPIWebsocketParams(audio_out_enabled=True, serializer=serializer)
    output = FastAPIWebsocketOutputTransport(MagicMock(), client, params)
    output.push_frame = AsyncMock()
    await output.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
        )
    )
    start = StartFrame(audio_out_sample_rate=8000)
    await output.process_frame(start, FrameDirection.DOWNSTREAM)
    return output


def _pushed_marks(output) -> list[PlaybackMarkFrame]:
    return [
        call.args[0]
        for call in output.push_frame.call_args_list
        if isinstance(call.args[0], PlaybackMarkFrame)
    ]


async def _until(condition, timeout: float = 1):
    async def wait():
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout)


class TestWebsocketPlaybackMarks(unittest.IsolatedAsyncioTestCase):
    async def test_a_twilio_mark_goes_on_when_twilio_echoes_it(self):
        serializer = TwilioFrameSerializer(
            STREAM_SID, params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
        )
        output = await _make_output(serializer)
        try:
            mark = PlaybackMarkFrame()
            await output.process_frame(mark, FrameDirection.DOWNSTREAM)

            await _until(lambda: output._client.send.await_count == 1)
            sent = json.loads(output._client.send.call_args.args[0])
            self.assertEqual(sent["mark"], {"name": mark.name})
            await asyncio.sleep(0.05)
            self.assertEqual(_pushed_marks(output), [])

            echo = {"event": "mark", "streamSid": STREAM_SID, "mark": {"name": mark.name}}
            await serializer.deserialize(json.dumps(echo))

            await _until(lambda: _pushed_marks(output) == [mark])
        finally:
            await output.cancel(CancelFrame())

    async def test_a_mark_goes_on_when_written_if_the_protocol_cant_report_playback(self):
        output = await _make_output(ProtobufFrameSerializer())
        try:
            mark = PlaybackMarkFrame()
            await output.process_frame(mark, FrameDirection.DOWNSTREAM)

            await _until(lambda: _pushed_marks(output) == [mark])
        finally:
            await output.cancel(CancelFrame())

    async def test_a_mark_goes_on_if_it_cant_be_sent(self):
        serializer = TwilioFrameSerializer(
            STREAM_SID, params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
        )
        output = await _make_output(serializer, connected=False)
        try:
            mark = PlaybackMarkFrame()
            await output.process_frame(mark, FrameDirection.DOWNSTREAM)

            await _until(lambda: _pushed_marks(output) == [mark])
            self.assertEqual(serializer._playback_marks, {})
        finally:
            await output.cancel(CancelFrame())
