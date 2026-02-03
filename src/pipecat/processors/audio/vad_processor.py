#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice Activity Detection processor for detecting speech in audio streams.

This module provides a VADProcessor that wraps a VADController to process
audio frames and push VAD-related frames into the pipeline.
"""

from typing import Type

from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.audio.vad.vad_controller import VADController
from pipecat.frames.frames import (
    Frame,
    UserSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class VADProcessor(FrameProcessor):
    """Processes audio frames through voice activity detection.

    This processor wraps a VADController to detect speech in audio streams
    and push VAD frames into the pipeline:

    - ``VADUserStartedSpeakingFrame``: Pushed when speech begins.
    - ``VADUserStoppedSpeakingFrame``: Pushed when speech ends.
    - ``UserSpeakingFrame``: Pushed periodically while speech is detected.

    Example::

        vad_processor = VADProcessor(vad_analyzer=SileroVADAnalyzer())
    """

    def __init__(
        self,
        *,
        vad_analyzer: VADAnalyzer,
        speech_activity_period: float = 0.2,
        **kwargs,
    ):
        """Initialize the VAD processor.

        Args:
            vad_analyzer: The VADAnalyzer instance for processing audio.
            speech_activity_period: Minimum interval in seconds between
                UserSpeakingFrame pushes. Defaults to 0.2.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._vad_controller = VADController(
            vad_analyzer, speech_activity_period=speech_activity_period
        )

        # Push VAD frames when speech events are detected
        @self._vad_controller.event_handler("on_speech_started")
        async def on_speech_started(_controller):
            logger.debug(f"{self}: User started speaking")
            await self.broadcast_frame(VADUserStartedSpeakingFrame)

        @self._vad_controller.event_handler("on_speech_stopped")
        async def on_speech_stopped(_controller):
            logger.debug(f"{self}: User stopped speaking")
            await self.broadcast_frame(VADUserStoppedSpeakingFrame)

        @self._vad_controller.event_handler("on_speech_activity")
        async def on_speech_activity(_controller):
            await self.broadcast_frame(UserSpeakingFrame)

        # Wire up frame pushing from controller to processor
        @self._vad_controller.event_handler("on_push_frame")
        async def on_push_frame(_controller, frame: Frame, direction: FrameDirection):
            await self.push_frame(frame, direction)

        @self._vad_controller.event_handler("on_broadcast_frame")
        async def on_broadcast_frame(_controller, frame_cls: Type[Frame], **kwargs):
            await self.broadcast_frame(frame_cls, **kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame through VAD and forward it.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Let the VAD controller handle the frame
        await self._vad_controller.process_frame(frame)

        # Always forward the frame
        await self.push_frame(frame, direction)
