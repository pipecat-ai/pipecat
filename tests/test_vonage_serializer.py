#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import unittest

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.serializers.vonage import VonageAudioTransport, VonageFrameSerializer


class TestVonageFrameSerializer(unittest.IsolatedAsyncioTestCase):
    async def test_default_binary_audio_transport(self):
        serializer = VonageFrameSerializer()
        await serializer.setup(StartFrame(audio_in_sample_rate=16000))

        audio = b"\x01\x00\x02\x00"
        frame = OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)

        payload = await serializer.serialize(frame)

        self.assertEqual(payload, audio)

    async def test_json_audio_transport_serializes_base64_audio(self):
        serializer = VonageFrameSerializer(
            VonageFrameSerializer.InputParams(
                audio_transport=VonageAudioTransport.JSON,
                static_fields={"kind": "media"},
            )
        )
        await serializer.setup(StartFrame(audio_in_sample_rate=16000))

        audio = b"\x01\x00\x02\x00"
        frame = OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)

        payload = await serializer.serialize(frame)

        self.assertIsInstance(payload, str)
        message = json.loads(payload)
        self.assertEqual(message["kind"], "media")
        self.assertEqual(base64.b64decode(message["audio"]), audio)

    async def test_json_audio_transport_deserializes_base64_audio(self):
        serializer = VonageFrameSerializer(
            VonageFrameSerializer.InputParams(audio_transport=VonageAudioTransport.JSON)
        )
        await serializer.setup(StartFrame(audio_in_sample_rate=16000))

        audio = b"\x01\x00\x02\x00"
        payload = json.dumps({"audio": base64.b64encode(audio).decode("ascii")})

        frame = await serializer.deserialize(payload)

        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.audio, audio)
        self.assertEqual(frame.sample_rate, 16000)
        self.assertEqual(frame.num_channels, 1)

    async def test_json_audio_transport_uses_custom_audio_fields(self):
        serializer = VonageFrameSerializer(
            VonageFrameSerializer.InputParams(
                audio_transport="json",
                audio_field="outbound_audio",
                receive_audio_field="inbound_audio",
            )
        )
        await serializer.setup(StartFrame(audio_in_sample_rate=16000))

        audio = b"\x01\x00\x02\x00"
        output_frame = OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)

        payload = await serializer.serialize(output_frame)
        message = json.loads(payload)
        self.assertIn("outbound_audio", message)
        self.assertNotIn("audio", message)

        input_payload = json.dumps({"inbound_audio": base64.b64encode(audio).decode("ascii")})
        input_frame = await serializer.deserialize(input_payload)

        self.assertIsInstance(input_frame, InputAudioRawFrame)
        self.assertEqual(input_frame.audio, audio)


if __name__ == "__main__":
    unittest.main()
