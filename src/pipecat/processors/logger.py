#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Optional, Tuple, Type

from loguru import logger

from pipecat.frames.frames import (
    BotSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TransportMessageFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logger.opt(ansi=True)


class FrameLogger(FrameProcessor):
    def __init__(
        self,
        prefix="Frame",
        color: Optional[str] = None,
        ignored_frame_types: Tuple[Type[Frame], ...] = (
            BotSpeakingFrame,
            InputAudioRawFrame,
            OutputAudioRawFrame,
            TransportMessageFrame,
        ),
    ):
        super().__init__()
        self._prefix = prefix
        self._color = color
        self._ignored_frame_types = ignored_frame_types

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._ignored_frame_types and not isinstance(frame, self._ignored_frame_types):
            dir = "<" if direction is FrameDirection.UPSTREAM else ">"
            msg = f"{dir} {self._prefix}: {frame}"
            if self._color:
                msg = f"<{self._color}>{msg}</>"
            logger.debug(msg)

        await self.push_frame(frame, direction)
