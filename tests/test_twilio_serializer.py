#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TwilioFrameSerializer deserialize() robustness.

Regression tests: a malformed or unexpectedly-shaped WebSocket message from
Twilio used to raise (JSONDecodeError/KeyError) straight out of deserialize(),
which the transport's receive loop treats as fatal -- silently ending the
call. Plivo/Vonage/Genesys already guard this; Twilio should behave the same.
"""

import json
import unittest

from pipecat.serializers.twilio import TwilioFrameSerializer
from tests.frame_processor_helpers import frame_processor_setup

STREAM_SID = "stream123"
SAMPLE_RATE = 8000


class TestTwilioDeserializeRobustness(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = TwilioFrameSerializer(
            stream_sid=STREAM_SID, params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
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
