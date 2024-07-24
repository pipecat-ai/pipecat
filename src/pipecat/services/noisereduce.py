
import noisereduce as nr
import numpy as np
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class NoiseReduce(FrameProcessor):
    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            self.reduce_noise(frame)
        await self.push_frame(frame, direction)

    def reduce_noise(self, frame: AudioRawFrame):
        if frame.num_channels != 1:
            raise ValueError(f"Expected 1 channel, got {frame.num_channels}")

        # load data
        data = np.frombuffer(frame.audio, dtype=np.int16)

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        data = data.astype(np.float32) + epsilon

        # perform noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=frame.sample_rate)
        frame.audio = np.clip(reduced_noise, -32768, 32767).astype(np.int16).tobytes()
