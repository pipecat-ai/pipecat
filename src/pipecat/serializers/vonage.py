import io
from loguru import logger
from pydub import AudioSegment
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    StartFrame,
    Frame,
    InputAudioRawFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    OutputAudioRawFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType

VONAGE_SAMPLE_RATE = 16000

class VonageFrameSerializer(FrameSerializer):
    """Serializer for Vonage Media Streams WebSocket protocol."""

    def __init__(self):
        self._sample_rate = VONAGE_SAMPLE_RATE
        self._input_resampler = create_stream_resampler()
        self.sample_width = 2
        self.channels = 1

    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.BINARY

    async def resample_audio(
        self, data: bytes, current_rate, num_channels, sample_width,
        target_rate=VONAGE_SAMPLE_RATE
    ) -> bytes:
        audio = AudioSegment.from_raw(
            io.BytesIO(data),
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
        if isinstance(frame, OutputAudioRawFrame):
            resampled_data = await self.resample_audio(
                frame.audio,
                frame.sample_rate,
                self.channels,
                self.sample_width,
                self._sample_rate
            )
            return resampled_data  # This must be `bytes`

        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            logger.info("VonageFrameSerializer does not support TransportFrame")
            return None

        else:
            logger.info(f"VonageFrameSerializer does not support frame type: {type(frame).__name__}")
            return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, str):
            logger.info("VonageFrameDeserializer does not support string data")
            return None

        resampled_data = await self._input_resampler.resample(
            data, self._sample_rate, self._sample_rate
        )
        return InputAudioRawFrame(
            audio=resampled_data, num_channels=1, sample_rate=self._sample_rate
        )
