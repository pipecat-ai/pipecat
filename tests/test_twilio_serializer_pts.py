#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TwilioFrameSerializer capture-timestamp handling.

Twilio media messages carry a ``timestamp`` field (ms since stream start).
``deserialize`` records it in ``frame.metadata["audio_capture_time_ns"]`` in
nanoseconds so downstream processors (e.g. ``AudioBufferProcessor``) can
position the audio by capture time instead of arrival time. It is recorded in
metadata rather than on ``frame.pts`` on purpose: ``pts`` is a general
presentation timestamp other transports set in their own units, so a dedicated
key keeps the capture-time signal unambiguous.
"""

import base64
import json
import unittest

from pipecat.frames.frames import InputAudioRawFrame, StartFrame
from pipecat.serializers.twilio import TwilioFrameSerializer

_METADATA_KEY = "audio_capture_time_ns"

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


class TestTwilioMediaCaptureTime(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.serializer = TwilioFrameSerializer(
            "MZstream", params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
        )
        await self.serializer.setup(StartFrame(audio_in_sample_rate=16000))

    async def test_timestamp_recorded_as_capture_time_ns(self):
        frame = await self.serializer.deserialize(media_message(timestamp="5000"))
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.metadata.get(_METADATA_KEY), 5000 * 1_000_000)

    async def test_capture_time_is_not_written_to_pts(self):
        # pts must stay untouched: it is a general presentation timestamp and
        # positioning must key off the metadata, never pts.
        frame = await self.serializer.deserialize(media_message(timestamp="5000"))
        self.assertIsNone(frame.pts)

    async def test_missing_timestamp_leaves_metadata_unset(self):
        frame = await self.serializer.deserialize(media_message(timestamp=None))
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertIsNone(frame.metadata.get(_METADATA_KEY))

    async def test_malformed_timestamp_leaves_metadata_unset(self):
        frame = await self.serializer.deserialize(media_message(timestamp="not-a-number"))
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertIsNone(frame.metadata.get(_METADATA_KEY))

    async def test_integer_timestamp_accepted(self):
        # Twilio documents the field as a string, but accept a raw number too.
        frame = await self.serializer.deserialize(media_message(timestamp=160))
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.metadata.get(_METADATA_KEY), 160 * 1_000_000)


if __name__ == "__main__":
    unittest.main()
