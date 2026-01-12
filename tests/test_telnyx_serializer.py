#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import unittest

import numpy as np

from pipecat.frames.frames import AudioRawFrame, InputAudioRawFrame, StartFrame
from pipecat.serializers.telnyx import TelnyxFrameSerializer


class TestTelnyxL16Codec(unittest.IsolatedAsyncioTestCase):
    """Test L16 codec support in TelnyxFrameSerializer."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        self.serializer = TelnyxFrameSerializer(
            stream_id="test-stream",
            call_control_id="test-call",
            api_key="test-key",
            outbound_encoding="L16",
            inbound_encoding="L16",
            params=TelnyxFrameSerializer.InputParams(
                telnyx_sample_rate=16000,
                sample_rate=16000,
                inbound_encoding="L16",
                outbound_encoding="L16",
            ),
        )
        # Setup with a StartFrame to initialize sample rates
        await self.serializer.setup(StartFrame(audio_in_sample_rate=16000))

    async def test_l16_serialize_byte_order(self):
        """Test that serialize converts PCM to big-endian L16."""
        # Create test audio: simple sine wave samples in little-endian
        samples = np.array([0x0102, 0x0304, 0x0506, 0x0708], dtype="<i2")
        audio_data = samples.tobytes()

        frame = AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)
        result = await self.serializer.serialize(frame)

        self.assertIsNotNone(result)
        message = json.loads(result)
        self.assertEqual(message["event"], "media")

        # Decode the payload and verify big-endian byte order
        payload = base64.b64decode(message["media"]["payload"])
        result_samples = np.frombuffer(payload, dtype=">i2")

        # Values should be the same, just in different byte order
        np.testing.assert_array_equal(result_samples, samples)

    async def test_l16_deserialize_byte_order(self):
        """Test that deserialize converts big-endian L16 to PCM."""
        # Create big-endian test data
        samples = np.array([0x0102, 0x0304, 0x0506, 0x0708], dtype=">i2")
        payload = base64.b64encode(samples.tobytes()).decode("utf-8")

        message = json.dumps({"event": "media", "media": {"payload": payload}})
        frame = await self.serializer.deserialize(message)

        self.assertIsInstance(frame, InputAudioRawFrame)

        # Result should be little-endian
        result_samples = np.frombuffer(frame.audio, dtype="<i2")
        np.testing.assert_array_equal(result_samples, samples)

    async def test_l16_roundtrip(self):
        """Test that L16 serialize/deserialize is lossless."""
        # Create test audio
        original_samples = np.array([100, -200, 300, -400, 500], dtype="<i2")
        audio_data = original_samples.tobytes()

        # Serialize (PCM -> L16)
        frame = AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)
        serialized = await self.serializer.serialize(frame)

        # Extract payload from serialized message
        message = json.loads(serialized)
        payload = message["media"]["payload"]

        # Deserialize (L16 -> PCM)
        incoming = json.dumps({"event": "media", "media": {"payload": payload}})
        result_frame = await self.serializer.deserialize(incoming)

        # Should get back the original samples
        result_samples = np.frombuffer(result_frame.audio, dtype="<i2")
        np.testing.assert_array_equal(result_samples, original_samples)

    async def test_l16_odd_length_rejected(self):
        """Test that L16 deserialize rejects odd-length data."""
        # Create invalid odd-length payload (3 bytes instead of even)
        payload = base64.b64encode(b"\x01\x02\x03").decode("utf-8")
        message = json.dumps({"event": "media", "media": {"payload": payload}})

        result = await self.serializer.deserialize(message)
        self.assertIsNone(result)


class TestTelnyxSerializerBackwardCompatibility(unittest.IsolatedAsyncioTestCase):
    """Test that existing PCMU/PCMA functionality still works."""

    async def test_pcmu_still_works(self):
        """Test PCMU encoding still functions."""
        serializer = TelnyxFrameSerializer(
            stream_id="test-stream",
            call_control_id="test-call",
            api_key="test-key",
            outbound_encoding="PCMU",
            inbound_encoding="PCMU",
        )
        await serializer.setup(StartFrame(audio_in_sample_rate=8000))

        # Create simple audio
        samples = np.zeros(160, dtype="<i2")
        audio_data = samples.tobytes()

        frame = AudioRawFrame(audio=audio_data, sample_rate=8000, num_channels=1)
        result = await serializer.serialize(frame)

        self.assertIsNotNone(result)
        message = json.loads(result)
        self.assertEqual(message["event"], "media")

    async def test_unsupported_encoding_raises(self):
        """Test that unsupported encodings raise ValueError."""
        serializer = TelnyxFrameSerializer(
            stream_id="test-stream",
            call_control_id="test-call",
            api_key="test-key",
            outbound_encoding="INVALID",
            inbound_encoding="PCMU",
        )
        await serializer.setup(StartFrame(audio_in_sample_rate=8000))

        payload = base64.b64encode(b"\x00\x00").decode("utf-8")
        message = json.dumps({"event": "media", "media": {"payload": payload}})

        with self.assertRaises(ValueError) as context:
            await serializer.deserialize(message)
        self.assertIn("Unsupported encoding", str(context.exception))
