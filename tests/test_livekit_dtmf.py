# Copyright (c) 2024-2026, Daily
# SPDX-License-Identifier: BSD 2-Clause License

"""Native LiveKit DTMF sequences and observable send failures."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import OutputDTMFFrame, OutputDTMFUrgentFrame
from pipecat.transports.livekit.transport import LiveKitOutputTransport, LiveKitTransportClient


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

        transport = SimpleNamespace(_client=SimpleNamespace(send_dtmf=send))
        frame = OutputDTMFFrame(buttons=[KeypadEntry.ONE, KeypadEntry.TWO, KeypadEntry.POUND])
        with self.assertRaises(RuntimeError):
            await LiveKitOutputTransport._write_dtmf_native(transport, frame)
        self.assertEqual(sent, ["1", "2"])

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
