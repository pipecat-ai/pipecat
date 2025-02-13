#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
    def __init__(self, *, serializer: FrameSerializer, **kwargs):
        super().__init__(**kwargs)
        self._serializer = serializer
        self._data_queue = asyncio.Queue()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        await self.push_frame(frame, direction)

        if isinstance(frame, (CancelFrame, EndFrame)):
            await self._data_queue.put(None)
        else:
            data = await self._serializer.serialize(frame)
            if data:
                await self._data_queue.put(data)

    async def generator(self) -> AsyncGenerator[Any, None]:
        running = True
        while running:
            data = await self._data_queue.get()
            running = data is not None
            if data:
                yield data
