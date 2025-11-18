#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Async generator processor for frame serialization and streaming."""

import asyncio
from typing import Any, AsyncGenerator

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer


class AsyncGeneratorProcessor(FrameProcessor):
    """A frame processor that serializes frames and provides them via async generator.

    This processor passes frames through unchanged while simultaneously serializing
    them and making the serialized data available through an async generator interface.
    Useful for streaming frame data to external consumers while maintaining the
    normal frame processing pipeline.
    """

    def __init__(self, *, serializer: FrameSerializer, **kwargs):
        """Initialize the async generator processor.

        Args:
            serializer: The frame serializer to use for converting frames to data.
            **kwargs: Additional arguments passed to the parent FrameProcessor.
        """
        super().__init__(**kwargs)
        self._serializer = serializer
        self._data_queue = asyncio.Queue()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames by passing them through and queuing serialized data.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, (CancelFrame, EndFrame)):
            await self._data_queue.put(None)
        else:
            data = await self._serializer.serialize(frame)
            if data:
                await self._data_queue.put(data)

    async def generator(self) -> AsyncGenerator[Any, None]:
        """Generate serialized frame data asynchronously.

        Yields:
            Serialized frame data from the internal queue until a termination
            signal (None) is received.
        """
        running = True
        while running:
            data = await self._data_queue.get()
            running = data is not None
            if data:
                yield data
