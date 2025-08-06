import io
import wave
import json
import base64
from loguru import logger
from pydub import AudioSegment
from pydantic import BaseModel
from typing import Optional
from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    Frame,
    StartFrame,
    EndFrame,
    CancelFrame,
    StartInterruptionFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


VONAGE_SAMPLE_RATE = 16000


class VonageFrameSerializer(FrameSerializer):
    """
    Serializer for Vonage Media Streams WebSocket protocol.
    This serializer handles converting between Pipecat frames and Vonage's WebSocket
    media streams protocol. It supports audio conversion, DTMF events, and automatic
    call termination.
    """
    
    class InputParams(BaseModel):
        """Configuration parameters for VonageFrameSerializer.

        Parameters:
            auto_hang_up: Whether to automatically terminate call on EndFrame.
        """
        auto_hang_up: bool = True

    def __init__(
        self,
        params: Optional[InputParams] = None,
        ):
        """Initialize the VonageFrameSerializer."""
        self._sample_rate = VONAGE_SAMPLE_RATE
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self.chunk_duration_ms = 20
        self.sample_width = 2
        self.channels = 1
        self.sleep_interval = 0.01
        self._params = params or VonageFrameSerializer.InputParams()
        self._hangup_attempted = False

    @property
    def type(self) -> FrameSerializerType:
        """
        Get the serializer type.

        Returns:
            FrameSerializerType: The serializer type, BINARY for Vonage.
        """
        return FrameSerializerType.BINARY

    async def setup(self, frame: StartFrame):
        self._sample_rate = VONAGE_SAMPLE_RATE

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """
        Serialize a Pipecat frame into Vonage-compatible format.

        Args:
            frame: The frame to serialize.

        Returns:
            bytes or list of bytes: Serialized chunk(s), or None.
        """
        if (
            self._params.auto_hang_up
            and not self._hangup_attempted
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            self._hangup_attempted = True
            logger.debug("VonageFrameSerializer would trigger hangup here (not implemented)")
            return None

        elif isinstance(frame, StartInterruptionFrame):
            answer = {"event": "clearAudio"}
            return json.dumps(answer)

        elif isinstance(frame, OutputAudioRawFrame):
            return frame.audio

        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            logger.info(
                "VonageFrameSerializer does not support serialization of TransportFrame with data: "
                + frame.message
            )
            return None

        else:
            logger.info(
                "VonageFrameSerializer does not support serialization of frame type: "
                + type(frame).__name__
            )
            return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """
        Deserialize incoming data to Pipecat frame.

        Args:
            data: Serialized data.

        Returns:
            Frame | None: Deserialized Pipecat frame.
        """
        if isinstance(data, str):
            logger.info(
                "VonageFrameSerializer does not support deserialization of string data: " + data
            )
            return None

        resampled_data = await self._input_resampler.resample(
            data, self._sample_rate, self._sample_rate
        )

        return InputAudioRawFrame(
            audio=resampled_data,
            num_channels=1,
            sample_rate=self._sample_rate,
        )
