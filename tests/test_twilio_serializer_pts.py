#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TwilioFrameSerializer capture-timestamp (pts) stamping.

Twilio media messages carry a ``timestamp`` field (ms since stream start).
``deserialize`` stamps it onto the frame's ``pts`` in nanoseconds so
downstream processors (e.g. ``AudioBufferProcessor``) can position the audio
by capture time instead of arrival time.
"""

import base64
import json
import unittest

from pipecat.frames.frames import InputAudioRawFrame, StartFrame
from pipecat.serializers.twilio import TwilioFrameSerializer

# One 200 ms chunk of 8 kHz u-law (0x10 decodes to a non-zero PCM value).
# Large enough that the stream resampler's warm-up latency still leaves
# output for the very first chunk.
PAYLOAD = base64.b64encode(b"\x10" * 1600).decode("utf-8")


def media_message(**media_overrides) -> str:
    media = {"track": "inbound", "chunk": "1", "timestamp": "5000", "payload": PAYLOAD}
    media.update(media_overrides)
    # A key set to None means "remove the field entirely".
    media = {k: v for k, v in media.items() if v is not None}
    return json.dumps({"event": "media", "media": media, "streamSid": "MZstream"})


class TestTwilioMediaTimestampPts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = TwilioFrameSerializer(
            "MZstream", params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
        )
        await self.serializer.setup(StartFrame(audio_in_sample_rate=16000))

    async def test_timestamp_stamped_as_pts_nanoseconds(self):
        frame = await self.serializer.deserialize(media_message(timestamp="5000"))
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.pts, 5000 * 1_000_000)

    async def test_missing_timestamp_leaves_pts_none(self):
        frame = await self.serializer.deserialize(media_message(timestamp=None))
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertIsNone(frame.pts)

    async def test_malformed_timestamp_leaves_pts_none(self):
        frame = await self.serializer.deserialize(media_message(timestamp="not-a-number"))
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertIsNone(frame.pts)

    async def test_integer_timestamp_accepted(self):
        # Twilio documents the field as a string, but accept a raw number too.
        frame = await self.serializer.deserialize(media_message(timestamp=160))
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.pts, 160 * 1_000_000)


if __name__ == "__main__":
    unittest.main()
