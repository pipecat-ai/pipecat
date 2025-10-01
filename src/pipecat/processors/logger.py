#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame logging utilities for debugging and monitoring frame flow in Pipecat pipelines."""

from typing import Optional, Tuple, Type

from loguru import logger

from pipecat.frames.frames import (
    BotSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    UserSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logger.opt(ansi=True)


class FrameLogger(FrameProcessor):
    """A frame processor that logs frame information for debugging purposes.

    This processor intercepts frames passing through the pipeline and logs
    their details with configurable formatting and filtering. Useful for
    debugging frame flow and understanding pipeline behavior.
    """

    def __init__(
        self,
        prefix="Frame",
        color: Optional[str] = None,
        ignored_frame_types: Tuple[Type[Frame], ...] = (
            BotSpeakingFrame,
            UserSpeakingFrame,
            InputAudioRawFrame,
            OutputAudioRawFrame,
        ),
    ):
        """Initialize the frame logger.

        Args:
            prefix: Text prefix to add to log messages. Defaults to "Frame".
            color: ANSI color code for log message formatting. If None, no coloring is applied.
            ignored_frame_types: Tuple of frame types to exclude from logging.
                Defaults to common high-frequency frames like audio and speaking frames.
        """
        super().__init__()
        self._prefix = prefix
        self._color = color
        self._ignored_frame_types = ignored_frame_types

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process and log frame information.

        Args:
            frame: The frame to process and potentially log.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if self._ignored_frame_types and not isinstance(frame, self._ignored_frame_types):
            dir = "<" if direction is FrameDirection.UPSTREAM else ">"
            msg = f"{dir} {self._prefix}: {frame}"
            if self._color:
                msg = f"<{self._color}>{msg}</>"
            logger.debug(msg)

        await self.push_frame(frame, direction)
