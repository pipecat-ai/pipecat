#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import abstractmethod
from typing import AsyncGenerator

from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService


class ImageGenService(AIService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Renders the image. Returns an Image object.
    @abstractmethod
    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            await self.start_processing_metrics()
            await self.process_generator(self.run_image_gen(frame.text))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)
