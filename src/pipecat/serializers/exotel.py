#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    KeypadEntry,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class ExotelFrameSerializer(FrameSerializer):
    """Serializer for Exotel Media Streams WebSocket protocol.

    This serializer handles converting between Pipecat frames and Exotel's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    Ref Doc for events - https://support.exotel.com/support/solutions/articles/3000108630-working-with-the-stream-and-voicebot-applet
    """

    class InputParams(BaseModel):
        """Configuration parameters for ExotelFrameSerializer.

        Attributes:
            exotel_sample_rate: Sample rate used by Exotel, defaults to 8000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
        """

        exotel_sample_rate: int = 8000
        sample_rate: Optional[int] = None

    def __init__(
        self, stream_sid: str, call_sid: Optional[str] = None, params: Optional[InputParams] = None
    ):
        """Initialize the ExotelFrameSerializer.

        Args:
            stream_sid: The Exotel Media Stream SID.
            call_sid: The associated Exotel Call SID (optional, not used in this implementation).
            params: Configuration parameters.
        """
        self._stream_sid = stream_sid
        self._call_sid = call_sid
        self._params = params or ExotelFrameSerializer.InputParams()

        self._exotel_sample_rate = self._params.exotel_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._resampler = create_default_resampler()

    @property
    def type(self) -> FrameSerializerType:
        """Gets the serializer type.

        Returns:
            The serializer type, either TEXT or BINARY.
        """
        return FrameSerializerType.TEXT

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Exotel WebSocket format.

        Handles conversion of various frame types to Exotel WebSocket messages.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
        """
        if isinstance(frame, StartInterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Exotel outputs PCM audio, but we need to resample to match requested sample_rate
            serialized_data = await self._resampler.resample(
                data, frame.sample_rate, self._exotel_sample_rate
            )
            payload = base64.b64encode(serialized_data).decode("ascii")

            answer = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Exotel WebSocket data to Pipecat frames.

        Handles conversion of Exotel media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Exotel.

        Returns:
            A Pipecat frame corresponding to the Exotel event, or None if unhandled.
        """
        message = json.loads(data)

        if message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            deserialized_data = await self._resampler.resample(
                payload,
                self._exotel_sample_rate,
                self._sample_rate,
            )

            # Input: Exotel takes PCM data, so just resample to match sample_rate
            audio_frame = InputAudioRawFrame(
                audio=deserialized_data,
                num_channels=1,  # Assuming mono audio from Exotel
                sample_rate=self._sample_rate,  # Use the configured pipeline input rate
            )
            return audio_frame
        elif message["event"] == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                # Handle case where string doesn't match any enum value
                logger.info(f"Invalid DTMF digit: {digit}")
                return None

        return None
