import io
from pydub import AudioSegment

from pipecat.frames.frames import Frame, OutputAudioRawFrame, StartFrame, EndFrame, CancelFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

VONAGE_SAMPLE_RATE = 16000

class ChunkedAudioSenderProcessor(FrameProcessor):
    def __init__(self, chunk_duration_ms=20, sample_rate=VONAGE_SAMPLE_RATE, channels=1, sample_width=2, **kwargs):
        super().__init__(**kwargs)
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width

    async def setup(self, setup):
        await super().setup(setup)

    async def resample_audio(self, data: bytes, current_rate, num_channels, sample_width, target_rate=VONAGE_SAMPLE_RATE) -> bytes:
        """
        Resample raw PCM audio data to target sample rate using PyDub.
        """
        try:
            if data[:4] == b'RIFF':
                # It's WAV
                audio = AudioSegment.from_file(io.BytesIO(data), format="wav")
            else:
                # It's raw PCM
                audio = AudioSegment.from_raw(
                    io.BytesIO(data),
                    sample_width=sample_width,
                    frame_rate=current_rate,
                    channels=num_channels,
                )
        except Exception as e:
            raise ValueError(f"Audio parsing failed: {e}")

        resampled_audio = (
            audio.set_channels(self.channels)
            .set_sample_width(self.sample_width)
            .set_frame_rate(target_rate)
        )
        return resampled_audio.raw_data

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM or not isinstance(frame, OutputAudioRawFrame):
            await self.push_frame(frame, direction)
            return

        resampled_data = await self.resample_audio(
            frame.audio,
            frame.sample_rate,
            frame.num_channels,
            self.sample_width,
            self.sample_rate,
        )

        chunk_frames = int(self.sample_rate * self.chunk_duration_ms / 1000)
        chunk_size = chunk_frames * self.channels * self.sample_width

        for i in range(0, len(resampled_data), chunk_size):
            chunk = resampled_data[i : i + chunk_size]
            if not chunk:
                continue

            chunk_frame = OutputAudioRawFrame(
                audio=chunk,
                sample_rate=self.sample_rate,
                num_channels=self.channels,
            )
            await self.push_frame(chunk_frame, FrameDirection.DOWNSTREAM)
