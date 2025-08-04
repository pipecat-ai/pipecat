import asyncio
from pipecat.frames.frames import Frame, OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # ✅ fixed import


class ChunkedAudioSenderProcessor(FrameProcessor):
    """Processor to split and delay OutputAudioRawFrames to simulate streaming."""

    def __init__(
        self,
        chunk_duration_ms: int = 20,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
        delay: float = 0.01,
        **kwargs,
    ):
        super().__init__(name="ChunkedAudioSenderProcessor", **kwargs)
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.delay = delay

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, OutputAudioRawFrame):
            await self.push_frame(frame, direction)
            return

        chunk_frames = int(self.sample_rate * self.chunk_duration_ms / 1000)
        chunk_size = chunk_frames * self.channels * self.sample_width
        audio = frame.audio

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            out_frame = OutputAudioRawFrame(audio=chunk, sample_rate=self.sample_rate, num_channels=self.channels)
            await self.push_frame(out_frame, direction)
            await asyncio.sleep(self.delay)
