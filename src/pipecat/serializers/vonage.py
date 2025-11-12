import io
import wave
import json
from loguru import logger
from pydub import AudioSegment
from pydantic import BaseModel
from typing import Optional

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

    async def resample_audio(
        self,
        data: bytes,
        current_rate,
        num_channels,
        sample_width,
        target_rate=VONAGE_SAMPLE_RATE,
    ) -> bytes:
        """
        Resample audio data to 16kHz mono PCM 16-bit.

        Args:
            data: Raw WAV byte data.
            current_rate: Original sample rate.
            num_channels: Original channel count.
            sample_width: Sample width in bytes.
            target_rate: Target sample rate (default: 16000).

        Returns:
            bytes: Resampled raw audio data.
        """
        wf = wave.open(io.BytesIO(data), "rb")
        num_frames = wf.getnframes()
        pcm_data = wf.readframes(num_frames)

        audio = AudioSegment.from_raw(
            io.BytesIO(pcm_data),
            sample_width=sample_width,
            frame_rate=current_rate,
            channels=num_channels,
        )

        resampled_audio = (
            audio.set_channels(num_channels)
            .set_sample_width(sample_width)
            .set_frame_rate(target_rate)
        )
        return resampled_audio.raw_data

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
            resampled_data = await self.resample_audio(
                frame.audio,
                frame.sample_rate,
                self.channels,
                self.sample_width,
                self._sample_rate,
            )

            self.chunk_frames = int(self._sample_rate * self.chunk_duration_ms / 1000)
            self.chunk_size = self.chunk_frames * self.channels * self.sample_width

            chunks = []
            for i in range(0, len(resampled_data), self.chunk_size):
                chunk = resampled_data[i : i + self.chunk_size]
                chunks.append(chunk)

            return chunks

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
