#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TelnyxFrameSerializer OutputTransportMessageFrame passthrough.

Regression test: serialize() had no case for OutputTransportMessageFrame/
OutputTransportMessageUrgentFrame, unlike every other telephony serializer
(Twilio, Plivo, Exotel, Genesys, Vonage), so a custom app message sent over
a Telnyx-backed pipeline was silently dropped instead of reaching the wire.
"""

import json
import unittest

from pipecat.frames.frames import OutputTransportMessageFrame, OutputTransportMessageUrgentFrame
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from tests.frame_processor_helpers import frame_processor_setup

STREAM_ID = "stream123"
SAMPLE_RATE = 8000


class TestTelnyxOutputTransportMessagePassthrough(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = TelnyxFrameSerializer(
            stream_id=STREAM_ID,
            outbound_encoding="PCMU",
            inbound_encoding="PCMU",
            params=TelnyxFrameSerializer.InputParams(auto_hang_up=False),
        )
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )

    async def test_output_transport_message_frame_is_passed_through(self):
        payload = {"foo": "bar"}
        result = await self.serializer.serialize(OutputTransportMessageFrame(message=payload))
        self.assertEqual(json.loads(result), payload)

    async def test_output_transport_message_urgent_frame_is_passed_through(self):
        payload = {"foo": "urgent"}
        result = await self.serializer.serialize(OutputTransportMessageUrgentFrame(message=payload))
        self.assertEqual(json.loads(result), payload)


if __name__ == "__main__":
    unittest.main()
