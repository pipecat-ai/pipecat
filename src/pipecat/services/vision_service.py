#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vision service implementation.

Provides base classes and implementations for computer vision services that can
analyze images and generate textual descriptions or answers to questions about
visual content.
"""

from abc import abstractmethod
from typing import AsyncGenerator

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService


class VisionService(AIService):
    """Base class for vision services.

    Provides common functionality for vision services that process images and
    generate textual responses. Handles image frame processing and integrates
    with the AI service infrastructure for metrics and lifecycle management.
    """

    def __init__(self, **kwargs):
        """Initialize the vision service.

        Args:
            **kwargs: Additional arguments passed to the parent AIService.
        """
        super().__init__(**kwargs)
        self._describe_text = None

    @abstractmethod
    async def run_vision(self, context: LLMContext) -> AsyncGenerator[Frame, None]:
        """Process the latest image in the context and generate results.

        This method must be implemented by subclasses to provide actual computer
        vision functionality such as image description, object detection, or
        visual question answering.

        Args:
            context: The context to process, containing image data.

        Yields:
            Frame: Frames containing the vision analysis results, typically TextFrame
            objects with descriptions or answers.
        """
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, handling vision image frames for analysis.

        Automatically processes VisionImageRawFrame objects by calling run_vision
        and handles metrics tracking. Other frames are passed through unchanged.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self.start_processing_metrics()
            await self.process_generator(self.run_vision(frame.context))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)
