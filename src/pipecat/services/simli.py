import asyncio

from pipecat.frames.frames import (
    Frame,
    OutputImageRawFrame,
    TTSAudioRawFrame,
    StartInterruptionFrame,
    EndFrame,
    CancelFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, StartFrame

import numpy as np
from av import AudioFrame
from av.audio.resampler import AudioResampler

from simli import SimliClient, SimliConfig
from loguru import logger


class SimliVideoService(FrameProcessor):
    def __init__(self, simli_config: SimliConfig, use_turn_server=False, latency_interval=0):
        super().__init__()
        self._simli_client = SimliClient(simli_config, use_turn_server, latency_interval)

        self._ready = False
        self._pipecat_resampler: AudioResampler = None
        self._simli_resampler = AudioResampler("s16", 1, 16000)

        self._audio_task: asyncio.Task = None
        self._video_task: asyncio.Task = None

    async def _start_connection(self):
        await self._simli_client.Initialize()
        self._ready = True
        # Create task to consume and process audio and video
        self._audio_task = asyncio.create_task(self._consume_and_process_audio())
        self._video_task = asyncio.create_task(self._consume_and_process_video())

    async def _consume_and_process_audio(self):
        while self._pipecat_resampler is None:
            await asyncio.sleep(0.001)
        async for audio_frame in self._simli_client.getAudioStreamIterator():
            # Process the audio frame
            try:
                resampled_frames = self._pipecat_resampler.resample(audio_frame)
                for resampled_frame in resampled_frames:
                    await self.push_frame(
                        TTSAudioRawFrame(
                            audio=resampled_frame.to_ndarray().tobytes(),
                            sample_rate=self._pipecat_resampler.rate,
                            num_channels=1,
                        ),
                    )
            except Exception as e:
                logger.exception(f"{self} exception: {e}")

    async def _consume_and_process_video(self):
        while self._pipecat_resampler is None:
            await asyncio.sleep(0.001)
        async for video_frame in self._simli_client.getVideoStreamIterator(targetFormat="rgb24"):
            # Process the video frame
            convertedFrame: OutputImageRawFrame = OutputImageRawFrame(
                image=video_frame.to_rgb().to_image().tobytes(),
                size=(video_frame.width, video_frame.height),
                format="RGB",
            )
            convertedFrame.pts = video_frame.pts
            await self.push_frame(
                convertedFrame,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self._start_connection()
        elif isinstance(frame, TTSAudioRawFrame):
            # Send audio frame to Simli
            try:
                if self._ready:
                    oldFrame = AudioFrame.from_ndarray(
                        np.frombuffer(frame.audio, dtype=np.int16)[None, :],
                        layout=frame.num_channels,
                    )
                    oldFrame.sample_rate = frame.sample_rate
                    if self._pipecat_resampler is None:
                        self._pipecat_resampler = AudioResampler(
                            "s16", oldFrame.layout, oldFrame.sample_rate
                        )

                    resampledFrame = self._simli_resampler.resample(oldFrame)
                    for frame in resampledFrame:
                        await self._simli_client.send(frame.to_ndarray().astype(np.int16).tobytes())
                    return
                else:
                    logger.warning(
                        "Simli Connection is not Initialized properly, passing audio to next processor"
                    )
            except Exception as e:
                logger.exception(f"{self} exception: {e}")
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._simli_client.stop()
            self._audio_task.cancel()
            await self._audio_task
            self._video_task.cancel()
            await self._video_task

        elif isinstance(frame, StartInterruptionFrame):
            await self._simli_client.clearBuffer()

        await self.push_frame(frame, direction)
