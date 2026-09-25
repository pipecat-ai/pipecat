#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TelnyxFrameSerializer OutputTransportMessageFrame passthrough and
deserialize() robustness.

Regression test (serialize): serialize() had no case for OutputTransportMessageFrame/
OutputTransportMessageUrgentFrame, unlike every other telephony serializer
(Twilio, Plivo, Exotel, Genesys, Vonage), so a custom app message sent over
a Telnyx-backed pipeline was silently dropped instead of reaching the wire.

Regression tests (deserialize): a malformed or unexpectedly-shaped WebSocket message
from Telnyx used to raise (JSONDecodeError/KeyError) straight out of deserialize(),
which the transport's receive loop treats as fatal -- silently ending the
call. Plivo/Vonage/Genesys already guard this; Telnyx should behave the same.
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


class TestTelnyxDeserializeRobustness(unittest.IsolatedAsyncioTestCase):
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

    async def test_malformed_json_is_ignored_not_raised(self):
        frame = await self.serializer.deserialize("not json")
        self.assertIsNone(frame)

    async def test_media_event_missing_payload_is_ignored_not_raised(self):
        frame = await self.serializer.deserialize(json.dumps({"event": "media", "media": {}}))
        self.assertIsNone(frame)

    async def test_message_missing_event_key_is_ignored_not_raised(self):
        frame = await self.serializer.deserialize(json.dumps({"foo": "bar"}))
        self.assertIsNone(frame)


if __name__ == "__main__":
    unittest.main()
