#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import pipecat.frames.protobufs.frames_pb2 as frame_protos
from pipecat.frames.frames import (
    InterruptionFrame,
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
        self.assertEqual(frame.text, text_frame.text)

        transcription_frame = TranscriptionFrame(
            text="Hello there!", user_id="123", timestamp="2021-01-01"
        )
        frame = await self.serializer.deserialize(
            await self.serializer.serialize(transcription_frame)
        )
        self.assertEqual(frame.text, transcription_frame.text)
        self.assertEqual(frame.user_id, transcription_frame.user_id)
        self.assertEqual(frame.timestamp, transcription_frame.timestamp)

        audio_frame = OutputAudioRawFrame(audio=b"1234567890", sample_rate=16000, num_channels=1)
        frame = await self.serializer.deserialize(await self.serializer.serialize(audio_frame))
        self.assertEqual(frame.audio, audio_frame.audio)
        self.assertEqual(frame.sample_rate, audio_frame.sample_rate)
        self.assertEqual(frame.num_channels, audio_frame.num_channels)

    async def test_interruption_frame_roundtrip(self):
        interruption_frame = InterruptionFrame()
        frame = await self.serializer.deserialize(
            await self.serializer.serialize(interruption_frame)
        )
        self.assertIsInstance(frame, InterruptionFrame)

    async def test_audio_timestamp_serialization(self):
        for pts in (None, 0, 123456789):
            with self.subTest(pts=pts):
                frame = OutputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
                frame.pts = pts

                data = await self.serializer.serialize(frame)
                wire_frame = frame_protos.Frame.FromString(data)

                self.assertEqual(wire_frame.audio.HasField("pts"), pts is not None)
                if pts is not None:
                    self.assertEqual(wire_frame.audio.pts, pts)

    async def test_audio_timestamp_deserialization(self):
        for pts in (None, 0, 123456789):
            with self.subTest(pts=pts):
                wire_frame = frame_protos.Frame()
                wire_frame.audio.audio = b"\x00\x00"
                wire_frame.audio.sample_rate = 16000
                wire_frame.audio.num_channels = 1
                if pts is not None:
                    wire_frame.audio.pts = pts

                frame = await self.serializer.deserialize(wire_frame.SerializeToString())

                self.assertEqual(frame.pts, pts)


if __name__ == "__main__":
    unittest.main()
