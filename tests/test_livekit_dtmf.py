# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Native LiveKit DTMF sequences and observable send failures."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    OutputAudioRawFrame,
    OutputDTMFFrame,
    OutputDTMFUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.livekit.transport import (
    LiveKitOutputTransport,
    LiveKitParams,
    LiveKitTransportClient,
)
from pipecat.utils.asyncio.task_manager import TaskManager


class TestLiveKitDTMF(unittest.IsolatedAsyncioTestCase):
    async def test_every_digit_is_sent_in_order(self):
        for frame_type in (OutputDTMFFrame, OutputDTMFUrgentFrame):
            sent = []

            async def send(digit, sent=sent):
                sent.append(digit)

            transport = SimpleNamespace(_client=SimpleNamespace(send_dtmf=send))
            frame = frame_type(buttons=[KeypadEntry.ONE, KeypadEntry.TWO, KeypadEntry.POUND])
            await LiveKitOutputTransport._write_dtmf_native(transport, frame)
            self.assertEqual(sent, ["1", "2", "#"])

    async def test_failure_stops_remaining_digits_and_is_observable(self):
        sent = []

        async def send(digit):
            sent.append(digit)
            if digit == "2":
                raise RuntimeError("gateway unavailable")

        transport = SimpleNamespace(_client=SimpleNamespace(send_dtmf=send), push_error=AsyncMock())
        frame = OutputDTMFFrame(buttons=[KeypadEntry.ONE, KeypadEntry.TWO, KeypadEntry.POUND])
        await LiveKitOutputTransport._write_dtmf_native(transport, frame)
        self.assertEqual(sent, ["1", "2"])
        transport.push_error.assert_awaited_once_with("Unable to send DTMF: gateway unavailable")

    async def test_failed_queued_dtmf_does_not_stop_later_audio(self):
        client = MagicMock()
        client.setup = AsyncMock()
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()
        client.send_dtmf = AsyncMock(side_effect=RuntimeError("gateway unavailable"))
        transport = LiveKitOutputTransport(
            MagicMock(), client, LiveKitParams(audio_out_enabled=True)
        )
        transport.push_frame = AsyncMock()
        written = asyncio.Event()

        async def write(frame):
            written.set()
            return True

        transport.write_audio_frame = AsyncMock(side_effect=write)

        await transport.setup(
            FrameProcessorSetup(
                clock=SystemClock(),
                task_manager=TaskManager(),
                pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
            )
        )
        await transport.process_frame(
            StartFrame(audio_out_sample_rate=16000), FrameDirection.DOWNSTREAM
        )
        try:
            await transport.process_frame(
                OutputDTMFFrame(buttons=[KeypadEntry.ONE]), FrameDirection.DOWNSTREAM
            )
            await transport.process_frame(
                OutputAudioRawFrame(
                    audio=b"\x00" * transport.audio_chunk_size,
                    sample_rate=transport.sample_rate,
                    num_channels=1,
                ),
                FrameDirection.DOWNSTREAM,
            )
            await asyncio.wait_for(written.wait(), timeout=1)
        finally:
            await transport.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)

        client.send_dtmf.assert_awaited_once_with("1")
        errors = [
            call.args[0]
            for call in transport.push_frame.await_args_list
            if isinstance(call.args[0], ErrorFrame)
        ]
        self.assertEqual([e.error for e in errors], ["Unable to send DTMF: gateway unavailable"])

    async def test_client_publishes_correct_code(self):
        publish = AsyncMock()
        client = SimpleNamespace(
            _connected=True,
            room=SimpleNamespace(local_participant=SimpleNamespace(publish_dtmf=publish)),
        )
        await LiveKitTransportClient.send_dtmf(client, "#")
        publish.assert_awaited_once_with(code=11, digit="#")

    async def test_client_does_not_swallow_gateway_failure(self):
        client = SimpleNamespace(
            _connected=True,
            room=SimpleNamespace(
                local_participant=SimpleNamespace(publish_dtmf=AsyncMock(side_effect=OSError))
            ),
        )
        with self.assertRaises(OSError):
            await LiveKitTransportClient.send_dtmf(client, "1")

    async def test_client_requires_connection_and_valid_digit(self):
        with self.assertRaises(RuntimeError):
            await LiveKitTransportClient.send_dtmf(SimpleNamespace(_connected=False), "1")
        with self.assertRaises(ValueError):
            await LiveKitTransportClient.send_dtmf(SimpleNamespace(_connected=True), "x")
