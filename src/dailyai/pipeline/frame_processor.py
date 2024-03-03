from abc import abstractmethod
from typing import AsyncGenerator

from dailyai.pipeline.frames import ControlQueueFrame, QueueFrame


class FrameProcessor:
    @abstractmethod
    async def process_frame(
        self, frame: QueueFrame
    ) -> AsyncGenerator[QueueFrame, None]:
        if isinstance(frame, ControlQueueFrame):
            yield frame

    @abstractmethod
    async def finalize(self) -> AsyncGenerator[QueueFrame, None]:
        # This is a trick for the interpreter (and linter) to know that this is a generator.
        if False:
            yield QueueFrame()

    @abstractmethod
    async def interrupted(self) -> None:
        pass

