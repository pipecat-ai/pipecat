import unittest

from pipecat.frames.frames import AudioRawFrame, TextFrame, TranscriptionFrame
from pipecat.serializers.protobuf import ProtobufFrameSerializer


class TestProtobufFrameSerializer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.serializer = ProtobufFrameSerializer()

    @unittest.skip("FIXME: This test is failing")
    async def test_roundtrip(self):
        text_frame = TextFrame(text="hello world")
        frame = self.serializer.deserialize(self.serializer.serialize(text_frame))
        self.assertEqual(frame, TextFrame(text="hello world"))

        transcription_frame = TranscriptionFrame(
            text="Hello there!", participantId="123", timestamp="2021-01-01"
        )
        frame = self.serializer.deserialize(self.serializer.serialize(transcription_frame))
        self.assertEqual(frame, transcription_frame)

        audio_frame = AudioRawFrame(data=b"1234567890")
        frame = self.serializer.deserialize(self.serializer.serialize(audio_frame))
        self.assertEqual(frame, audio_frame)


if __name__ == "__main__":
    unittest.main()
