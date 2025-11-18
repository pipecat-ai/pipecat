#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    OutputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.serializers.protobuf import ProtobufFrameSerializer


class TestProtobufFrameSerializer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.serializer = ProtobufFrameSerializer()

    async def test_roundtrip(self):
        text_frame = TextFrame(text="hello world")
        frame = await self.serializer.deserialize(await self.serializer.serialize(text_frame))
        self.assertEqual(text_frame, frame)

        transcription_frame = TranscriptionFrame(
            text="Hello there!", user_id="123", timestamp="2021-01-01"
        )
        frame = await self.serializer.deserialize(
            await self.serializer.serialize(transcription_frame)
        )
        self.assertEqual(frame, transcription_frame)

        audio_frame = OutputAudioRawFrame(audio=b"1234567890", sample_rate=16000, num_channels=1)
        frame = await self.serializer.deserialize(await self.serializer.serialize(audio_frame))
        self.assertEqual(frame.audio, audio_frame.audio)
        self.assertEqual(frame.sample_rate, audio_frame.sample_rate)
        self.assertEqual(frame.num_channels, audio_frame.num_channels)
