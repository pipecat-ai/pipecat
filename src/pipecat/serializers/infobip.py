"""Infobip Media Streams WebSocket protocol serializer for Pipecat."""

import asyncio
import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import infobip_resampler
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    KeypadEntry,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class InfobipFrameSerializer(FrameSerializer):
    """Serializer for Infobip Media Streams WebSocket protocol.

    This serializer handles converting between Pipecat frames and Infobip's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.

    When auto_hang_up is enabled (default), the serializer will automatically terminate
    the Infobip call when an EndFrame or CancelFrame is processed, but requires Infobip
    credentials to be provided.
    """

    class InputParams(BaseModel):
        """Configuration parameters for InfobipFrameSerializer.

        Parameters:
            infobip_sample_rate: Sample rate used by Infobip (8000 or 16000 Hz).
            sample_rate: Optional override for pipeline input sample rate.
        """

        infobip_sample_rate: int = 16000
        sample_rate: Optional[int] = None

    def __init__(self, params: Optional[InputParams] = None):
        """Initialize the InfobipFrameSerializer.

        Args:
            params: Optional configuration parameters for the serializer.
        """
        self._sample_rate = 0
        self._infobip_sample_rate = self._params.infobip_sample_rate
        self._params = params or InfobipFrameSerializer.InputParams()
        self._input_resampler = infobip_resampler
        self._output_resampler = infobip_resampler

    @property
    def type(self) -> FrameSerializerType:
        """Gets the serializer type.

        Returns:
            The serializer type, either TEXT or BINARY.
        """
        return FrameSerializerType.BINARY

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to Infobip WebSocket format.

        Handles conversion of various frame types to Infobip WebSocket messages.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Serialized data as string or bytes, or None if the frame isn't handled.
        """
        if isinstance(frame, StartInterruptionFrame):
            return None
        elif isinstance(frame, TTSAudioRawFrame):
            data = frame.audio
            # We can also use the Resampy resampler here for better quality audio but for performance reasons we're using the scipy resampler
            resampled_data = await self._input_resampler(
                data, frame.sample_rate, self._infobip_sample_rate
            )
            if not resampled_data:
                return None
            await asyncio.sleep(0.025)
            return resampled_data
        elif isinstance(frame, EndFrame):
            logger.info("EndFrame received - call ending")
            return None

        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            if isinstance(frame.message, dict):
                result = json.dumps(frame.message)
                return result
            return frame.message
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes Infobip WebSocket data to Pipecat frames.

        Handles conversion of Infobip media events to appropriate Pipecat frames.

        Args:
            data: The raw WebSocket data from Infobip.

        Returns:
            A Pipecat frame corresponding to the Infobip event, or None if unhandled.
        """
        if isinstance(data, str):
            try:
                message = json.loads(data)
                event_type = message.get("event")
                if event_type == "websocket:dtmf":
                    digit = message.get("digit")
                    try:
                        result = InputDTMFFrame(KeypadEntry(digit))
                        return result
                    except ValueError:
                        return None
                elif event_type == "websocket:connected":
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"deserialize: JSONDecodeError {e}")
                return None
        elif isinstance(data, bytes):
            if not data:
                return None
            # We can also use the Resampy resampler here for better quality audio but for performance reasons we're using the scipy resampler
            resampled_data = await self._output_resampler(
                data, self._sample_rate, self._infobip_sample_rate
            )
            if resampled_data is not None and len(resampled_data) > 0:
                audio_frame = InputAudioRawFrame(
                    audio=resampled_data,
                    num_channels=1,
                    sample_rate=self._sample_rate,
                )
                return audio_frame
            return None
        return None
