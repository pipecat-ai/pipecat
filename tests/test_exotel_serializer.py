#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Exotel Media Streams serializer."""

import base64
import json
from unittest.mock import AsyncMock

import pytest

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
)
from pipecat.serializers.exotel import ExotelFrameSerializer


@pytest.fixture
def serializer() -> ExotelFrameSerializer:
    """Create an Exotel serializer for tests."""
    return ExotelFrameSerializer(stream_sid="stream-123", call_sid="call-456")


@pytest.mark.asyncio
@pytest.mark.parametrize("frame", [EndFrame(), CancelFrame()])
async def test_terminal_frames_are_handled(
    serializer: ExotelFrameSerializer, frame: EndFrame | CancelFrame
):
    """Terminal frames should use the Exotel lifecycle hook without emitting data."""
    assert await serializer.serialize(frame) is None


@pytest.mark.asyncio
async def test_audio_serialization_is_unchanged(serializer: ExotelFrameSerializer):
    """Audio should be resampled and encoded as an Exotel media event."""
    audio = b"\x01\x02\x03\x04"
    serializer._output_resampler.resample = AsyncMock(return_value=audio)

    result = await serializer.serialize(
        OutputAudioRawFrame(audio=audio, sample_rate=8000, num_channels=1)
    )

    assert json.loads(result) == {
        "event": "media",
        "streamSid": "stream-123",
        "media": {"payload": base64.b64encode(audio).decode("ascii")},
    }


@pytest.mark.asyncio
async def test_dtmf_deserialization_is_unchanged(serializer: ExotelFrameSerializer):
    """Exotel DTMF events should produce input DTMF frames."""
    result = await serializer.deserialize(json.dumps({"event": "dtmf", "dtmf": {"digit": "5"}}))

    assert isinstance(result, InputDTMFFrame)
    assert result.button == KeypadEntry.FIVE


@pytest.mark.asyncio
async def test_interruption_serialization_is_unchanged(serializer: ExotelFrameSerializer):
    """Interruptions should clear queued Exotel audio."""
    result = await serializer.serialize(InterruptionFrame())

    assert json.loads(result) == {"event": "clear", "streamSid": "stream-123"}


@pytest.mark.asyncio
async def test_transport_message_serialization_is_unchanged(serializer: ExotelFrameSerializer):
    """Transport messages should be passed through as JSON."""
    result = await serializer.serialize(OutputTransportMessageFrame(message={"event": "mark"}))

    assert json.loads(result) == {"event": "mark"}
