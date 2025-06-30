#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class SpeechmaticsDebugService(FrameProcessor):
    """Simple Frame logger."""

    def __init__(
        self,
        name: str,
        ignore_types: list[type[Frame]] = [InputAudioRawFrame],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._name = name
        self._ignore_types = ignore_types

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle frames in the pipeline."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        # Skip logging if in the ignore list
        for frame_type in self._ignore_types:
            if isinstance(frame, frame_type):
                return

        # Log for debugging
        logger.debug(f"{self._name}: Processing frame: {frame}")
