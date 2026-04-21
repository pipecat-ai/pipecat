#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base input transport implementation for Pipecat.

This module provides the BaseInputTransport class which handles audio and video
input processing.
"""

import asyncio

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    FilterUpdateSettingsFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    StartFrame,
    StopFrame,
    SystemFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams

AUDIO_INPUT_TIMEOUT_SECS = 0.5


class BaseInputTransport(FrameProcessor):
    """Base class for input transport implementations.

    Handles audio and video input processing including Voice Activity Detection,
    turn analysis, audio filtering, and user interaction management. Supports
    interruption handling and provides hooks for transport-specific implementations.
    """

    def __init__(self, params: TransportParams, **kwargs):
        """Initialize the base input transport.

        Args:
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)

        self._params = params

        # Input sample rate. It will be initialized on StartFrame.
        self._sample_rate = 0

        # Track bot speaking state for interruption logic
        self._bot_speaking = False

        # Track user speaking state for interruption logic
        self._user_speaking = False
        # Last time a UserSpeakingFrame was pushed.
        self._user_speaking_frame_time = 0
        # How often a UserSpeakingFrame should be pushed (value should be
        # greater than the audio chunks to have any effect).
        self._user_speaking_frame_period = 0.2

        # Task to process incoming audio (VAD) and push audio frames downstream
        # if passthrough is enabled.
        self._audio_task = None

        # If the transport is stopped with `StopFrame` we might still be
        # receiving frames from the transport but we really don't want to push
        # them downstream until we get another `StartFrame`.
        self._paused = False

    def enable_audio_in_stream_on_start(self, enabled: bool) -> None:
        """Enable or disable audio streaming on transport start.

        Args:
            enabled: Whether to start audio streaming immediately on transport start.
        """
        logger.debug(f"Enabling audio on start. {enabled}")
        self._params.audio_in_stream_on_start = enabled

    async def start_audio_in_streaming(self):
        """Start audio input streaming.

        Override in subclasses to implement transport-specific audio streaming.
        """
        pass

    @property
    def sample_rate(self) -> int:
        """Get the current audio sample rate.

        Returns:
            The sample rate in Hz.
        """
        return self._sample_rate

    async def start(self, frame: StartFrame):
        """Start the input transport and initialize components.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self._paused = False
        self._user_speaking = False

        self._sample_rate = self._params.audio_in_sample_rate or frame.audio_in_sample_rate

        # Start audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.start(self._sample_rate)

    async def stop(self, frame: EndFrame):
        """Stop the input transport and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        # Cancel and wait for the audio input task to finish.
        await self._cancel_audio_task()
        # Stop audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.stop()

    async def pause(self, frame: StopFrame):
        """Pause the input transport temporarily.

        Args:
            frame: The stop frame signaling transport pause.
        """
        self._paused = True
        # Cancel task so we clear the queue
        await self._cancel_audio_task()
        # Retart the task
        self._create_audio_task()

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        # Cancel and wait for the audio input task to finish.
        await self._cancel_audio_task()
        # Stop audio filter.
        if self._params.audio_in_filter:
            await self._params.audio_in_filter.stop()

    async def set_transport_ready(self, frame: StartFrame):
        """Called when the transport is ready to stream.

        Args:
            frame: The start frame containing initialization parameters.
        """
        # Create audio input queue and task if needed.
        self._create_audio_task()

    async def push_video_frame(self, frame: InputImageRawFrame):
        """Push a video frame downstream if video input is enabled.

        Args:
            frame: The input video frame to process.
        """
        if self._params.video_in_enabled and not self._paused:
            await self.push_frame(frame)

    async def push_audio_frame(self, frame: InputAudioRawFrame):
        """Push an audio frame to the processing queue if audio input is enabled.

        Args:
            frame: The input audio frame to process.
        """
        if self._params.audio_in_enabled and not self._paused:
            await self._audio_in_queue.put(frame)

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle transport-specific logic.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Specific system frames
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
            await self.push_frame(frame, direction)
        # All other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames
        elif isinstance(frame, EndFrame):
            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self.push_frame(frame, direction)
            await self.stop(frame)
        elif isinstance(frame, StopFrame):
            await self.push_frame(frame, direction)
            await self.pause(frame)
        elif isinstance(frame, FilterUpdateSettingsFrame) and self._params.audio_in_filter:
            await self._params.audio_in_filter.process_frame(frame)
        # Other frames
        else:
            await self.push_frame(frame, direction)

    #
    # Audio input
    #

    def _create_audio_task(self):
        """Create the audio processing task if audio input is enabled."""
        if not self._audio_task and self._params.audio_in_enabled:
            self._audio_in_queue = asyncio.Queue()
            self._audio_task = self.create_task(self._audio_task_handler())

    async def _cancel_audio_task(self):
        """Cancel and cleanup the audio processing task."""
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None

    async def _audio_task_handler(self):
        """Main audio processing task handler."""
        # Skip timeout handling until the first audio frame arrives (e.g. client
        # not yet connected).
        audio_received = False
        while True:
            try:
                frame: InputAudioRawFrame = await asyncio.wait_for(
                    self._audio_in_queue.get(), timeout=AUDIO_INPUT_TIMEOUT_SECS
                )

                # From now on, timeout should warn if there's no audio.
                audio_received = True

                # Filter audio, if an audio filter is available.
                if self._params.audio_in_filter:
                    frame.audio = await self._params.audio_in_filter.filter(frame.audio)

                # Skip frames with no audio data (e.g. filter is buffering).
                if not frame.audio:
                    self._audio_in_queue.task_done()
                    continue

                # Push audio downstream if passthrough is set.
                if self._params.audio_in_passthrough:
                    await self.push_frame(frame)

                self._audio_in_queue.task_done()
            except TimeoutError:
                if not audio_received:
                    continue
