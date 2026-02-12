#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the ACS frame serializer."""

import base64
import json
import math
import struct

import pytest

# Skip the entire module if the azure package is not installed.
pytest.importorskip("azure.communication.callautomation")

from azure.core.exceptions import ResourceNotFoundError  # noqa: E402

from pipecat.audio.dtmf.types import KeypadEntry  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.serializers.acs import ACSFrameSerializer  # noqa: E402


def make_sine_pcm(num_samples: int = 8192, sample_rate: int = 16000) -> bytes:
    """Generate a 440Hz sine wave as raw int16 PCM bytes.

    Silence produces empty output from the streaming resampler — a real signal is required.
    """
    samples = [
        int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(num_samples)
    ]
    return struct.pack(f"<{num_samples}h", *samples)


def expected_resampled_length(input_samples: int, in_rate: int, out_rate: int) -> int:
    """Expected byte length after resampling, allowing ±30% for streaming buffer rounding."""
    return int(input_samples * out_rate / in_rate) * 2


@pytest.fixture
def serializer(mocked_call_automation_client, mocked_call_connection_id):
    serializer = ACSFrameSerializer(
        call_automation_client=mocked_call_automation_client,
        call_connection_id=mocked_call_connection_id,
    )
    serializer._sample_rate = 16000
    return serializer


class TestSetup:
    """Test suite testing setup correctly sets all sample rates."""

    @pytest.mark.asyncio
    async def test_setup_affects_deserialized_audio_sample_rate(self, serializer):
        """The sample rate set by setup() must appear on deserialized InputAudioRawFrame."""
        await serializer.setup(StartFrame(audio_in_sample_rate=16000))

        # Use audio at _acs_sample_rate (24k) so the resampler has something to convert
        pcm_24k = make_sine_pcm(num_samples=8192, sample_rate=24000)
        message = json.dumps(
            {
                "kind": "AudioData",
                "audioData": {"data": base64.b64encode(pcm_24k).decode("ascii")},
            }
        )
        result = await serializer.deserialize(message)

        assert isinstance(result, InputAudioRawFrame)
        assert result.sample_rate == 16000

    @pytest.mark.asyncio
    async def test_params_sample_rate_overrides_start_frame(
        self, mocked_call_automation_client, mocked_call_connection_id
    ):
        """InputParams.sample_rate takes priority over StartFrame.audio_in_sample_rate."""
        params = ACSFrameSerializer.InputParams(acs_sample_rate=24000, sample_rate=8000)
        s = ACSFrameSerializer(
            call_automation_client=mocked_call_automation_client,
            call_connection_id=mocked_call_connection_id,
            params=params,
        )
        await s.setup(StartFrame(audio_in_sample_rate=16000))

        pcm_24k = make_sine_pcm(num_samples=8192, sample_rate=24000)
        message = json.dumps(
            {
                "kind": "AudioData",
                "audioData": {"data": base64.b64encode(pcm_24k).decode("ascii")},
            }
        )
        result = await s.deserialize(message)

        assert isinstance(result, InputAudioRawFrame)
        assert result.sample_rate == 8000

    @pytest.mark.asyncio
    async def test_custom_acs_sample_rate_affects_serialized_audio_length(
        self, mocked_call_automation_client, mocked_call_connection_id
    ):
        """Serializer configured with 16k ACS rate should produce different output length than 24k."""
        params_16k = ACSFrameSerializer.InputParams(acs_sample_rate=16000)
        params_24k = ACSFrameSerializer.InputParams(acs_sample_rate=24000)

        s_16k = ACSFrameSerializer(
            call_automation_client=mocked_call_automation_client,
            call_connection_id=mocked_call_connection_id,
            params=params_16k,
        )
        s_24k = ACSFrameSerializer(
            call_automation_client=mocked_call_automation_client,
            call_connection_id=mocked_call_connection_id,
            params=params_24k,
        )

        await s_16k.setup(StartFrame(audio_in_sample_rate=16000))
        await s_24k.setup(StartFrame(audio_in_sample_rate=16000))

        pcm = make_sine_pcm(num_samples=8192, sample_rate=16000)
        frame = OutputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)

        result_16k = json.loads(await s_16k.serialize(frame))
        result_24k = json.loads(await s_24k.serialize(frame))

        audio_16k = base64.b64decode(result_16k["AudioData"]["data"])
        audio_24k = base64.b64decode(result_24k["AudioData"]["data"])

        # Upsampling to 24k should produce more bytes than passthrough at 16k
        assert len(audio_24k) > len(audio_16k)


class TestSerialize:
    """Test suite testing all functionalities of serialize."""

    @pytest.mark.asyncio
    async def test_audio_frame_structure_and_resampling(self, serializer):
        """Audio serialized from 16k pipeline to 24k ACS should produce a larger payload."""
        raw_pcm_16k = make_sine_pcm(num_samples=8192, sample_rate=16000)
        frame = OutputAudioRawFrame(audio=raw_pcm_16k, sample_rate=16000, num_channels=1)

        result = await serializer.serialize(frame)

        payload = json.loads(result)
        assert payload["Kind"] == "AudioData"
        assert payload["StopAudio"] is None

        # 16k → 24k: output should be ~1.5x the input byte count
        resampled = base64.b64decode(payload["AudioData"]["data"])
        expected = expected_resampled_length(input_samples=8192, in_rate=16000, out_rate=24000)
        assert abs(len(resampled) - expected) < expected * 0.3

    @pytest.mark.asyncio
    async def test_interruption_frame(self, serializer):
        result = await serializer.serialize(InterruptionFrame())

        payload = json.loads(result)
        assert payload["Kind"] == "StopAudio"
        assert payload["AudioData"] is None
        assert payload["StopAudio"] == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("frame", [EndFrame(), CancelFrame()])
    async def test_end_cancel_frame_hangs_up_and_returns_none(
        self, serializer, mocked_call_connection_client, frame
    ):
        result = await serializer.serialize(frame)

        assert result is None
        mocked_call_connection_client.hang_up.assert_called_once_with(is_for_everyone=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("frame", [EndFrame(), CancelFrame()])
    async def test_hang_up_already_disconnected_does_not_raise(
        self, serializer, mocked_call_connection_client, frame
    ):
        mocked_call_connection_client.hang_up.side_effect = ResourceNotFoundError("gone")

        result = await serializer.serialize(frame)

        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("frame", [EndFrame(), CancelFrame()])
    async def test_hang_up_without_connection_id_does_not_raise(self, serializer, frame):
        serializer._call_connection_id = None

        result = await serializer.serialize(frame)

        assert result is None

    @pytest.mark.asyncio
    async def test_unhandled_frame_returns_none(self, serializer):
        assert await serializer.serialize(StartFrame()) is None


class TestDeserialize:
    """Test suite testing all functionalities of deserialize."""

    @pytest.mark.asyncio
    async def test_audio_frame_structure_and_resampling(self, serializer):
        """Audio arriving from ACS at 24k should be resampled to the 16k pipeline rate."""
        pcm_24k = make_sine_pcm(num_samples=8192, sample_rate=24000)
        message = json.dumps(
            {
                "kind": "AudioData",
                "audioData": {"data": base64.b64encode(pcm_24k).decode("ascii")},
            }
        )

        result = await serializer.deserialize(message)

        assert isinstance(result, InputAudioRawFrame)
        assert result.sample_rate == 16000
        assert result.num_channels == ACSFrameSerializer.NUM_CHANNELS
        # 24k → 16k: output should be ~2/3 the input byte count
        expected = expected_resampled_length(input_samples=8192, in_rate=24000, out_rate=16000)
        assert abs(len(result.audio) - expected) < expected * 0.3

    @pytest.mark.asyncio
    @pytest.mark.parametrize("digit", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "#"])
    async def test_dtmf_frame(self, serializer, digit):
        message = json.dumps({"kind": "DtmfData", "dtmfData": {"data": digit}})

        result = await serializer.deserialize(message)

        assert isinstance(result, InputDTMFFrame)
        assert result.button == KeypadEntry(digit)

    @pytest.mark.asyncio
    async def test_invalid_dtmf_digit_returns_none(self, serializer):
        message = json.dumps({"kind": "DtmfData", "dtmfData": {"data": "X"}})

        assert await serializer.deserialize(message) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "message",
        [
            json.dumps({"kind": "SomeUnknownEvent", "data": {}}),
            json.dumps({"kind": "StopAudio"}),
            json.dumps({"data": "no kind key"}),
        ],
    )
    async def test_unhandled_messages_return_none(self, serializer, message):
        assert await serializer.deserialize(message) is None
