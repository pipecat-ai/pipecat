#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputImageRawFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, StartFrame
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator

try:
    from av.audio.frame import AudioFrame
    from av.audio.resampler import AudioResampler
    from simli import SimliClient, SimliConfig
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Simli, you need to `pip install pipecat-ai[simli]`.")
    raise Exception(f"Missing module: {e}")


class SimliVideoService(FrameProcessor):
    def __init__(
        self,
        simli_config: SimliConfig,
        use_turn_server: bool = False,
        latency_interval: int = 0,
    ):
        super().__init__()
        self._simli_client = SimliClient(simli_config, use_turn_server, latency_interval)

        self._pipecat_resampler_event = asyncio.Event()
        self._pipecat_resampler: AudioResampler = None
        self._simli_resampler = AudioResampler("s16", "mono", 16000)

        self._initialized = False
        self._audio_task: asyncio.Task = None
        self._video_task: asyncio.Task = None

    async def _start_connection(self):
        if not self._initialized:
            await self._simli_client.Initialize()
            self._initialized = True

        # Create task to consume and process audio and video
        if not self._audio_task:
            self._audio_task = self.create_task(self._consume_and_process_audio())

        if not self._video_task:
            self._video_task = self.create_task(self._consume_and_process_video())

    async def _consume_and_process_audio(self):
        await self._pipecat_resampler_event.wait()
        audio_iterator = self._simli_client.getAudioStreamIterator()
        async for audio_frame in WatchdogAsyncIterator(audio_iterator, manager=self.task_manager):
            resampled_frames = self._pipecat_resampler.resample(audio_frame)
            for resampled_frame in resampled_frames:
                audio_array = resampled_frame.to_ndarray()
                # Only push frame is there is audio (e.g. not silence)
                if audio_array.any():
                    await self.push_frame(
                        TTSAudioRawFrame(
                            audio=audio_array.tobytes(),
                            sample_rate=self._pipecat_resampler.rate,
                            num_channels=1,
                        ),
                    )

    async def _consume_and_process_video(self):
        await self._pipecat_resampler_event.wait()
        video_iterator = self._simli_client.getVideoStreamIterator(targetFormat="rgb24")
        async for video_frame in WatchdogAsyncIterator(video_iterator, manager=self.task_manager):
            # Process the video frame
            convertedFrame: OutputImageRawFrame = OutputImageRawFrame(
                image=video_frame.to_rgb().to_image().tobytes(),
                size=(video_frame.width, video_frame.height),
                format="RGB",
            )
            convertedFrame.pts = video_frame.pts
            await self.push_frame(convertedFrame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._start_connection()
        elif isinstance(frame, TTSAudioRawFrame):
            # Send audio frame to Simli
            try:
                old_frame = AudioFrame.from_ndarray(
                    np.frombuffer(frame.audio, dtype=np.int16)[None, :],
                    layout="mono" if frame.num_channels == 1 else "stereo",
                )
                old_frame.sample_rate = frame.sample_rate

                if self._pipecat_resampler is None:
                    self._pipecat_resampler = AudioResampler(
                        "s16", old_frame.layout, old_frame.sample_rate
                    )
                    self._pipecat_resampler_event.set()

                resampled_frames = self._simli_resampler.resample(old_frame)
                for resampled_frame in resampled_frames:
                    await self._simli_client.send(
                        resampled_frame.to_ndarray().astype(np.int16).tobytes()
                    )
            except Exception as e:
                logger.exception(f"{self} exception: {e}")
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame):
            await self._simli_client.clearBuffer()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _stop(self):
        await self._simli_client.stop()
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None
        if self._video_task:
            await self.cancel_task(self._video_task)
            self._video_task = None
