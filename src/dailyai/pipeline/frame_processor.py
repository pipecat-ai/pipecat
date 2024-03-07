from abc import abstractmethod
from typing import AsyncGenerator

from dailyai.pipeline.frames import ControlFrame, Frame


class FrameProcessor:
    """This is the base class for all frame processors. Frame processors consume a frame
    and yield 0 or more frames. Generally frame processors are used as part of a pipeline
    where frames come from a source queue, are processed by a series of frame processors,
    then placed on a sink queue.

    By convention, FrameProcessors should immediately yield any frames they don't process.

    Stateful FrameProcessors should watch for the EndStreamQueueFrame and finalize their
    output, eg. yielding an unfinished sentence if they're aggregating LLM output to full
    sentences. EndStreamQueueFrame is also a chance to clean up any services that need to
    be closed, del'd, etc.
    """

    @abstractmethod
    async def process_frame(
        self, frame: Frame
    ) -> AsyncGenerator[Frame, None]:
        """Process a single frame and yield 0 or more frames."""
        if isinstance(frame, ControlFrame):
            yield frame
        yield frame

    @abstractmethod
    async def interrupted(self) -> None:
        """Handle any cleanup if the pipeline was interrupted."""
        pass
