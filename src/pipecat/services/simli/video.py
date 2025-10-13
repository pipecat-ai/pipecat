#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simli video service for real-time avatar generation."""

import asyncio

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputImageRawFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, StartFrame

try:
    from av.audio.frame import AudioFrame
    from av.audio.resampler import AudioResampler
    from simli import SimliClient, SimliConfig
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Simli, you need to `pip install pipecat-ai[simli]`.")
    raise Exception(f"Missing module: {e}")


class SimliVideoService(FrameProcessor):
    """Simli video service for real-time avatar generation.

    Provides real-time avatar video generation by processing audio frames
    and producing synchronized video output using the Simli API. Handles
    audio resampling, video frame processing, and connection management.
    """

    def __init__(
        self,
        simli_config: SimliConfig,
        use_turn_server: bool = False,
        latency_interval: int = 0,
        simli_url: str = "https://api.simli.ai",
        is_trinity_avatar: bool = False,
    ):
        """Initialize the Simli video service.

        Args:
            simli_config: Configuration object for Simli client settings.
            use_turn_server: Whether to use TURN server for connection. Defaults to False.
            latency_interval: Latency interval setting for sending health checks to check the latency to Simli Servers. Defaults to 0.
            simli_url: URL of the simli servers. Can be changed for custom deployments of enterprise users.
            is_trinity_avatar: boolean to tell simli client that this is a Trinity avatar which reduces latency when using Trinity.

        """
        super().__init__()
        self._initialized = False
        simli_config.maxIdleTime += 5
        simli_config.maxSessionLength += 5
        self._simli_client = SimliClient(
            simli_config,
            use_turn_server,
            latency_interval,
            simliURL=simli_url,
        )

        self._pipecat_resampler: AudioResampler = None
        self._pipecat_resampler_event = asyncio.Event()
        self._simli_resampler = AudioResampler("s16", "mono", 16000)

        self._audio_task: asyncio.Task = None
        self._video_task: asyncio.Task = None
        self._is_trinity_avatar = is_trinity_avatar
        self._previously_interrupted = is_trinity_avatar
        self._audio_buffer = bytearray()

    async def _start_connection(self):
        """Start the connection to Simli service and begin processing tasks."""
        if not self._initialized:
            await self._simli_client.Initialize()
            self._initialized = True

        # Create task to consume and process audio and video
        await self._simli_client.sendSilence()
        self._audio_task = self.create_task(self._consume_and_process_audio())
        self._video_task = self.create_task(self._consume_and_process_video())

    async def _consume_and_process_audio(self):
        """Consume audio frames from Simli and push them downstream."""
        await self._pipecat_resampler_event.wait()
        audio_iterator = self._simli_client.getAudioStreamIterator()
        async for audio_frame in audio_iterator:
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
        """Consume video frames from Simli and convert them to output frames."""
        await self._pipecat_resampler_event.wait()
        video_iterator = self._simli_client.getVideoStreamIterator(targetFormat="rgb24")
        async for video_frame in video_iterator:
            # Process the video frame
            convertedFrame: OutputImageRawFrame = OutputImageRawFrame(
                image=video_frame.to_rgb().to_image().tobytes(),
                size=(video_frame.width, video_frame.height),
                format="RGB",
            )
            convertedFrame.pts = video_frame.pts
            await self.push_frame(convertedFrame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle Simli video generation.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
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
                    audioBytes = resampled_frame.to_ndarray().astype(np.int16).tobytes()
                    if self._previously_interrupted:
                        self._audio_buffer.extend(audioBytes)
                        if len(self._audio_buffer) >= 128000:
                            try:
                                for flushFrame in self._simli_resampler.resample(None):
                                    self._audio_buffer.extend(
                                        flushFrame.to_ndarray().astype(np.int16).tobytes()
                                    )
                            finally:
                                await self._simli_client.playImmediate(self._audio_buffer)
                                self._previously_interrupted = False
                                self._audio_buffer = bytearray()
                    else:
                        await self._simli_client.send(audioBytes)
                return
            except Exception as e:
                logger.exception(f"{self} exception: {e}")
        elif isinstance(frame, TTSStoppedFrame):
            try:
                if self._previously_interrupted and len(self._audio_buffer) > 0:
                    await self._simli_client.playImmediate(self._audio_buffer)
                    self._previously_interrupted = False
                    self._audio_buffer = bytearray()
            except Exception as e:
                logger.exception(f"{self} exception: {e}")
            return
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
        elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
            if not self._previously_interrupted:
                await self._simli_client.clearBuffer()
            self._previously_interrupted = self._is_trinity_avatar

        await self.push_frame(frame, direction)

    async def _stop(self):
        """Stop the Simli client and cancel processing tasks."""
        await self._simli_client.stop()
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None
        if self._video_task:
            await self.cancel_task(self._video_task)
            self._video_task = None
