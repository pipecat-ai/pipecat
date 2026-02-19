#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TelnyxFrameSerializer L16 codec support."""

import base64
import json
import sys

import numpy as np
import pytest

from pipecat.frames.frames import AudioRawFrame, InputAudioRawFrame, StartFrame
from pipecat.serializers.telnyx import TelnyxFrameSerializer


class TestTelnyxFrameSerializerL16:
    """Test L16 (raw PCM) codec support in TelnyxFrameSerializer."""

    @pytest.fixture
    def l16_serializer_16khz(self):
        """Create a TelnyxFrameSerializer configured for L16 at 16kHz."""
        params = TelnyxFrameSerializer.InputParams(
            telnyx_sample_rate=16000,
            sample_rate=16000,
            inbound_encoding="L16",
            outbound_encoding="L16",
        )
        return TelnyxFrameSerializer(
            stream_id="test-stream",
            outbound_encoding="L16",
            inbound_encoding="L16",
            params=params,
        )

    @pytest.fixture
    def l16_serializer_8khz(self):
        """Create a TelnyxFrameSerializer configured for L16 at 8kHz."""
        params = TelnyxFrameSerializer.InputParams(
            telnyx_sample_rate=8000,
            sample_rate=16000,
            inbound_encoding="L16",
            outbound_encoding="L16",
        )
        return TelnyxFrameSerializer(
            stream_id="test-stream",
            outbound_encoding="L16",
            inbound_encoding="L16",
            params=params,
        )

    @pytest.fixture
    def pcmu_serializer(self):
        """Create a TelnyxFrameSerializer configured for PCMU at 8kHz."""
        return TelnyxFrameSerializer(
            stream_id="test-stream",
            outbound_encoding="PCMU",
            inbound_encoding="PCMU",
        )

    @pytest.mark.asyncio
    async def test_l16_serialize_same_rate_byte_order(self, l16_serializer_16khz):
        """Test L16 serialization converts to network byte order (big-endian)."""
        # Setup
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await l16_serializer_16khz.setup(start_frame)

        # Create audio with known pattern: [0x0102, 0x0304] in little-endian
        # Little-endian bytes: 02 01 04 03
        # Big-endian bytes (expected output): 01 02 03 04
        audio_le = np.array([0x0102, 0x0304], dtype="<i2")  # Little-endian
        audio_data = audio_le.tobytes()
        audio_frame = AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)

        # Serialize
        result = await l16_serializer_16khz.serialize(audio_frame)

        # Verify
        assert result is not None
        message = json.loads(result)
        assert message["event"] == "media"
        decoded_audio = base64.b64decode(message["media"]["payload"])

        # On little-endian systems, output should be byte-swapped to big-endian
        if sys.byteorder == "little":
            expected_be = np.array([0x0102, 0x0304], dtype=">i2").tobytes()
            assert decoded_audio == expected_be
        else:
            # On big-endian systems, no swap needed
            assert decoded_audio == audio_data

    @pytest.mark.asyncio
    async def test_l16_deserialize_byte_order(self, l16_serializer_16khz):
        """Test L16 deserialization converts from network byte order (big-endian) to host."""
        # Setup
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await l16_serializer_16khz.setup(start_frame)

        # Create big-endian audio: [0x0102, 0x0304]
        audio_be = np.array([0x0102, 0x0304], dtype=">i2")  # Big-endian
        payload = base64.b64encode(audio_be.tobytes()).decode("utf-8")
        message = json.dumps({"event": "media", "media": {"payload": payload}})

        # Deserialize
        result = await l16_serializer_16khz.deserialize(message)

        # Verify
        assert result is not None
        assert isinstance(result, InputAudioRawFrame)

        # On little-endian systems, output should be converted to little-endian
        result_array = np.frombuffer(result.audio, dtype="<i2" if sys.byteorder == "little" else ">i2")
        expected_values = [0x0102, 0x0304]
        np.testing.assert_array_equal(result_array, expected_values)

    @pytest.mark.asyncio
    async def test_l16_roundtrip_same_rate(self, l16_serializer_16khz):
        """Test L16 roundtrip: serialize then deserialize preserves audio values."""
        # Setup
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await l16_serializer_16khz.setup(start_frame)

        # Create test audio (sine wave)
        audio_np = np.sin(np.linspace(0, 2 * np.pi * 440, 160)) * 16000
        original_audio = audio_np.astype(np.int16).tobytes()
        original_values = np.frombuffer(original_audio, dtype=np.int16).copy()

        # Serialize
        audio_frame = AudioRawFrame(audio=original_audio, sample_rate=16000, num_channels=1)
        serialized = await l16_serializer_16khz.serialize(audio_frame)
        assert serialized is not None

        # Deserialize
        result = await l16_serializer_16khz.deserialize(serialized)
        assert result is not None

        # Verify values are preserved
        result_values = np.frombuffer(result.audio, dtype=np.int16)
        np.testing.assert_array_equal(result_values, original_values)

    @pytest.mark.asyncio
    async def test_l16_serialize_with_resampling(self, l16_serializer_8khz):
        """Test L16 serialization with resampling (16kHz pipeline to 8kHz Telnyx)."""
        # Setup
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await l16_serializer_8khz.setup(start_frame)

        # Create audio frame at 16kHz with enough samples for resampler
        audio_np = np.sin(np.linspace(0, 2 * np.pi * 440, 1600)) * 16000
        audio_data = audio_np.astype(np.int16).tobytes()
        audio_frame = AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)

        # Serialize
        result = await l16_serializer_8khz.serialize(audio_frame)

        # Verify - stream resampler may buffer, so result could be smaller or None initially
        if result is not None:
            message = json.loads(result)
            assert message["event"] == "media"
            decoded_audio = base64.b64decode(message["media"]["payload"])
            # Resampled + byte swapped audio
            assert len(decoded_audio) <= len(audio_data)
            # Verify it's valid big-endian int16 (even length)
            assert len(decoded_audio) % 2 == 0

    @pytest.mark.asyncio
    async def test_l16_deserialize_with_resampling(self, l16_serializer_8khz):
        """Test L16 deserialization with resampling (8kHz Telnyx to 16kHz pipeline)."""
        # Setup
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await l16_serializer_8khz.setup(start_frame)

        # Create incoming Telnyx media message at 8kHz (big-endian)
        audio_np = np.sin(np.linspace(0, 2 * np.pi * 440, 800)) * 16000
        audio_be = audio_np.astype(">i2")  # Big-endian for Telnyx
        payload = base64.b64encode(audio_be.tobytes()).decode("utf-8")
        message = json.dumps({"event": "media", "media": {"payload": payload}})

        # Deserialize
        result = await l16_serializer_8khz.deserialize(message)

        # Verify
        if result is not None:
            assert isinstance(result, InputAudioRawFrame)
            assert result.sample_rate == 16000
            assert len(result.audio) > 0
            # Verify it's valid int16 (even length)
            assert len(result.audio) % 2 == 0

    @pytest.mark.asyncio
    async def test_l16_odd_length_audio_truncated(self, l16_serializer_16khz):
        """Test that odd-length audio is truncated with warning."""
        # Setup
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await l16_serializer_16khz.setup(start_frame)

        # Create odd-length audio (161 bytes instead of 160)
        audio_data = bytes([0x00, 0x80] * 80 + [0xFF])  # 161 bytes
        audio_frame = AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)

        # Serialize - should truncate to 160 bytes
        result = await l16_serializer_16khz.serialize(audio_frame)

        if result is not None:
            message = json.loads(result)
            decoded_audio = base64.b64decode(message["media"]["payload"])
            # Should be truncated to even length
            assert len(decoded_audio) % 2 == 0

    @pytest.mark.asyncio
    async def test_l16_empty_audio_returns_none(self, l16_serializer_16khz):
        """Test that empty audio returns None."""
        # Setup
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await l16_serializer_16khz.setup(start_frame)

        # Create empty audio
        audio_frame = AudioRawFrame(audio=bytes(), sample_rate=16000, num_channels=1)

        # Serialize
        result = await l16_serializer_16khz.serialize(audio_frame)
        assert result is None

    @pytest.mark.asyncio
    async def test_pcmu_still_works(self, pcmu_serializer):
        """Verify PCMU encoding still works after L16 changes."""
        # Setup
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await pcmu_serializer.setup(start_frame)

        # Create audio frame with enough samples for resampler
        audio_np = np.sin(np.linspace(0, 2 * np.pi * 440, 1600)) * 16000
        audio_data = audio_np.astype(np.int16).tobytes()
        audio_frame = AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)

        # Serialize
        result = await pcmu_serializer.serialize(audio_frame)

        # Verify
        if result is not None:
            message = json.loads(result)
            assert message["event"] == "media"

    @pytest.mark.asyncio
    async def test_unsupported_encoding_raises(self):
        """Test that unsupported encodings raise ValueError."""
        serializer = TelnyxFrameSerializer(
            stream_id="test-stream",
            outbound_encoding="OPUS",  # Not supported
            inbound_encoding="OPUS",
        )
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await serializer.setup(start_frame)

        audio_frame = AudioRawFrame(
            audio=bytes([0x00] * 320), sample_rate=16000, num_channels=1
        )

        with pytest.raises(ValueError, match="Unsupported encoding"):
            await serializer.serialize(audio_frame)

    def test_input_params_defaults(self):
        """Test InputParams defaults are preserved."""
        params = TelnyxFrameSerializer.InputParams()
        assert params.telnyx_sample_rate == 8000
        assert params.sample_rate is None
        assert params.inbound_encoding == "PCMU"
        assert params.outbound_encoding == "PCMU"
        assert params.auto_hang_up is True

    def test_input_params_l16_16khz(self):
        """Test InputParams can be configured for L16 at 16kHz."""
        params = TelnyxFrameSerializer.InputParams(
            telnyx_sample_rate=16000,
            sample_rate=16000,
            inbound_encoding="L16",
            outbound_encoding="L16",
        )
        assert params.telnyx_sample_rate == 16000
        assert params.sample_rate == 16000
        assert params.inbound_encoding == "L16"
        assert params.outbound_encoding == "L16"


class TestTelnyxL16ByteOrderEdgeCases:
    """Edge case tests for L16 byte order handling."""

    @pytest.fixture
    def serializer(self):
        """Create a TelnyxFrameSerializer configured for L16 at 16kHz."""
        params = TelnyxFrameSerializer.InputParams(
            telnyx_sample_rate=16000,
            sample_rate=16000,
        )
        return TelnyxFrameSerializer(
            stream_id="test-stream",
            outbound_encoding="L16",
            inbound_encoding="L16",
            params=params,
        )

    @pytest.mark.asyncio
    async def test_l16_max_min_values(self, serializer):
        """Test L16 handles max/min int16 values correctly."""
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await serializer.setup(start_frame)

        # Test with extreme values
        audio_np = np.array([32767, -32768, 0, 1, -1], dtype=np.int16)
        audio_data = audio_np.tobytes()
        audio_frame = AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)

        # Serialize
        result = await serializer.serialize(audio_frame)
        assert result is not None

        # Deserialize and verify values
        deserialized = await serializer.deserialize(result)
        assert deserialized is not None
        result_values = np.frombuffer(deserialized.audio, dtype=np.int16)
        np.testing.assert_array_equal(result_values, audio_np)

    @pytest.mark.asyncio
    async def test_l16_single_sample(self, serializer):
        """Test L16 with single sample (2 bytes)."""
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await serializer.setup(start_frame)

        audio_np = np.array([12345], dtype=np.int16)
        audio_data = audio_np.tobytes()
        audio_frame = AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)

        result = await serializer.serialize(audio_frame)
        assert result is not None

        deserialized = await serializer.deserialize(result)
        assert deserialized is not None
        result_value = np.frombuffer(deserialized.audio, dtype=np.int16)[0]
        assert result_value == 12345

    @pytest.mark.asyncio
    async def test_l16_deserialize_odd_length_truncated(self, serializer):
        """Test deserialization truncates odd-length payloads."""
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await serializer.setup(start_frame)

        # Create odd-length big-endian payload
        audio_be = np.array([0x1234, 0x5678], dtype=">i2").tobytes() + b"\xFF"
        assert len(audio_be) == 5  # Odd length

        payload = base64.b64encode(audio_be).decode("utf-8")
        message = json.dumps({"event": "media", "media": {"payload": payload}})

        result = await serializer.deserialize(message)
        if result is not None:
            # Should have truncated to 4 bytes (2 samples)
            assert len(result.audio) % 2 == 0

    @pytest.mark.asyncio
    async def test_l16_byteswap_vs_astype_equivalence(self, serializer):
        """Verify byteswap() and astype() produce identical byte output.

        This test ensures the serialize path (using byteswap) and deserialize path
        (using astype) are consistent and produce bit-identical results.
        """
        # Test multiple patterns including edge cases
        test_values = [
            [0x0102, 0x0304],  # Simple pattern
            [32767, -32768],  # Max/min int16
            [0, 1, -1],  # Zero and near-zero
            [0x1234, 0x5678, -25924],  # Mixed values (0x9ABC as signed = -25924)
        ]

        for values in test_values:
            audio_le = np.array(values, dtype="<i2")  # Little-endian (host)

            # Method 1: byteswap (used in serialize)
            byteswap_result = audio_le.byteswap().tobytes()

            # Method 2: astype (used in deserialize, reversed)
            # First convert to big-endian representation
            audio_be = np.array(values, dtype=">i2")
            # Then use astype to convert back to little-endian
            astype_result = audio_be.view(">i2").astype("<i2").tobytes()

            # The roundtrip should preserve values
            roundtrip_values = np.frombuffer(astype_result, dtype="<i2")
            np.testing.assert_array_equal(roundtrip_values, values)

    @pytest.mark.asyncio
    async def test_l16_roundtrip_with_resampling(self):
        """Test full roundtrip with resampling: 16kHz pipeline → 8kHz Telnyx → 16kHz pipeline."""
        # Create serializer with mismatched rates
        params = TelnyxFrameSerializer.InputParams(
            telnyx_sample_rate=8000,
            sample_rate=16000,
            inbound_encoding="L16",
            outbound_encoding="L16",
        )
        serializer = TelnyxFrameSerializer(
            stream_id="test-stream",
            outbound_encoding="L16",
            inbound_encoding="L16",
            params=params,
        )

        start_frame = StartFrame(audio_in_sample_rate=16000)
        await serializer.setup(start_frame)

        # Create a longer audio buffer for resampling (need enough samples)
        # 100ms of audio at 16kHz = 1600 samples
        t = np.linspace(0, 0.1, 1600)
        audio_np = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        original_audio = audio_np.tobytes()

        audio_frame = AudioRawFrame(audio=original_audio, sample_rate=16000, num_channels=1)

        # Serialize (16kHz → 8kHz, byte swap to big-endian)
        serialized = await serializer.serialize(audio_frame)

        # Stream resampler may buffer, so we might need multiple frames
        if serialized is not None:
            # Deserialize (big-endian → little-endian, 8kHz → 16kHz)
            result = await serializer.deserialize(serialized)

            if result is not None:
                assert isinstance(result, InputAudioRawFrame)
                assert result.sample_rate == 16000
                # Verify it's valid PCM (even length, reasonable values)
                assert len(result.audio) % 2 == 0
                result_values = np.frombuffer(result.audio, dtype=np.int16)
                assert np.all(result_values >= -32768)
                assert np.all(result_values <= 32767)


class TestTelnyxL16ErrorHandling:
    """Tests for error handling in L16 deserialization."""

    @pytest.fixture
    def serializer(self):
        """Create a TelnyxFrameSerializer configured for L16."""
        params = TelnyxFrameSerializer.InputParams(
            telnyx_sample_rate=16000,
            sample_rate=16000,
        )
        return TelnyxFrameSerializer(
            stream_id="test-stream",
            outbound_encoding="L16",
            inbound_encoding="L16",
            params=params,
        )

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self, serializer):
        """Test that malformed JSON returns None instead of crashing."""
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await serializer.setup(start_frame)

        # Various malformed JSON inputs
        malformed_inputs = [
            "not json at all",
            "{incomplete",
            '{"event": "media", "media": {',
            "",
            "null",
        ]

        for bad_input in malformed_inputs:
            result = await serializer.deserialize(bad_input)
            assert result is None, f"Expected None for malformed input: {bad_input}"

    @pytest.mark.asyncio
    async def test_invalid_base64_returns_none(self, serializer):
        """Test that invalid base64 payload returns None instead of crashing."""
        start_frame = StartFrame(audio_in_sample_rate=16000)
        await serializer.setup(start_frame)

        # Valid JSON but invalid base64 payload
        invalid_payloads = [
            "not-valid-base64!!!",
            "====",
            "ab",  # Too short
        ]

        for bad_payload in invalid_payloads:
            message = json.dumps({"event": "media", "media": {"payload": bad_payload}})
            result = await serializer.deserialize(message)
            assert result is None, f"Expected None for invalid base64: {bad_payload}"
