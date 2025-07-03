#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Image generation service implementation.

Provides base functionality for AI-powered image generation services that convert
text prompts into images.
"""

from abc import abstractmethod
from typing import AsyncGenerator

from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService


class ImageGenService(AIService):
    """Base class for image generation services.

    Processes TextFrames by using their content as prompts for image generation.
    Subclasses must implement the run_image_gen method to provide actual image
    generation functionality using their specific AI service.
    """

    def __init__(self, **kwargs):
        """Initialize the image generation service.

        Args:
            **kwargs: Additional arguments passed to the parent AIService.
        """
        super().__init__(**kwargs)

    # Renders the image. Returns an Image object.
    @abstractmethod
    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate an image from a text prompt.

        This method must be implemented by subclasses to provide actual image
        generation functionality using their specific AI service.

        Args:
            prompt: The text prompt to generate an image from.

        Yields:
            Frame: Frames containing the generated image (typically ImageRawFrame
                or URLImageRawFrame).
        """
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for image generation.

        TextFrames are used as prompts for image generation, while other frames
        are passed through unchanged.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            await self.start_processing_metrics()
            await self.process_generator(self.run_image_gen(frame.text))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)
