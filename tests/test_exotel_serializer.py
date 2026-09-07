#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ExotelFrameSerializer wire format."""

import json
import unittest

from pipecat.frames.frames import InterruptionFrame, OutputAudioRawFrame
from pipecat.serializers.exotel import ExotelFrameSerializer
from tests.frame_processor_helpers import frame_processor_setup

STREAM_SID = "stream123"
SAMPLE_RATE = 8000


class TestExotelSerializeWireFormat(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = ExotelFrameSerializer(stream_sid=STREAM_SID, call_sid="call123")
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )

    async def test_media_event(self):
        frame = OutputAudioRawFrame(
            audio=b"\x00\x00" * 160, sample_rate=SAMPLE_RATE, num_channels=1
        )
        message = json.loads(await self.serializer.serialize(frame))
        self.assertEqual(message["event"], "media")
        self.assertEqual(message["stream_sid"], STREAM_SID)
        self.assertIn("payload", message["media"])

    async def test_clear_event(self):
        message = json.loads(await self.serializer.serialize(InterruptionFrame()))
        self.assertEqual(message, {"event": "clear", "stream_sid": STREAM_SID})


class TestExotelDeserialize(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = ExotelFrameSerializer(stream_sid=STREAM_SID)
        await self.serializer.setup(
            frame_processor_setup(
                audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE
            )
        )

    async def test_media_round_trip(self):
        audio = b"\x01\x02" * 160
        outbound = await self.serializer.serialize(
            OutputAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1)
        )
        frame = await self.serializer.deserialize(outbound)
        self.assertEqual(frame.audio, audio)
        self.assertEqual(frame.sample_rate, SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
