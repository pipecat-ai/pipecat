import asyncio
from typing import AsyncGenerator, List
from dailyai.pipeline.frame_processor import FrameProcessor

from dailyai.pipeline.frames import EndPipeFrame, EndFrame, Frame


class Pipeline:
    """
    This class manages a pipe of FrameProcessors, and runs them in sequence. The "source"
    and "sink" queues are managed by the caller. You can use this class stand-alone to
    perform specialized processing, or you can use the Transport's run_pipeline method to
    instantiate and run a pipeline with the Transport's sink and source queues.
    """

    def __init__(
        self,
        processors: List[FrameProcessor],
        source: asyncio.Queue | None = None,
        sink: asyncio.Queue[Frame] | None = None,
    ):
        """ Create a new pipeline. By default neither the source nor sink
        queues are set, so you'll need to pass them to this constructor or
        call set_source and set_sink before using the pipeline. Note that
        the transport's run_*_pipeline methods will set the source and sink
        queues on the pipeline for you.
        """
        self.processors = processors
        self.source: asyncio.Queue[Frame] | None = source
        self.sink: asyncio.Queue[Frame] | None = sink

    def set_source(self, source: asyncio.Queue[Frame]):
        """ Set the source queue for this pipeline. Frames from this queue
        will be processed by each frame_processor in the pipeline, or order
        from first to last. """
        self.source = source

    def set_sink(self, sink: asyncio.Queue[Frame]):
        """ Set the sink queue for this pipeline. After the last frame_processor
        has processed a frame, its output will be placed on this queue."""
        self.sink = sink

    async def get_next_source_frame(self) -> AsyncGenerator[Frame, None]:
        """ Convenience function to get the next frame from the source queue. This
        lets us consistently have an AsyncGenerator yield frames, from either the
        source queue or a frame_processor."""
        if self.source is None:
            raise ValueError("Source queue not set")
        yield await self.source.get()

    async def run_pipeline_recursively(
        self, initial_frame: Frame, processors: List[FrameProcessor]
    ) -> AsyncGenerator[Frame, None]:
        if processors:
            async for frame in processors[0].process_frame(initial_frame):
                async for final_frame in self.run_pipeline_recursively(frame, processors[1:]):
                    yield final_frame
        else:
            yield initial_frame

    async def run_pipeline(self):
        """ Run the pipeline. Take each frame from the source queue, pass it to
        the first frame_processor, pass the output of that frame_processor to the
        next in the list, etc. until the last frame_processor has processed the
        resulting frames, then place those frames in the sink queue.

        The source and sink queues must be set before calling this method.

        This method will exit when an EndStreamQueueFrame is placed on the sink queue.
        No more frames will be placed on the sink queue after an EndStreamQueueFrame, even
        if it's not the last frame yielded by the last frame_processor in the pipeline.."""

        if self.source is None or self.sink is None:
            raise ValueError("Source or sink queue not set")

        try:
            while True:
                initial_frame = await self.source.get()
                async for frame in self.run_pipeline_recursively(initial_frame, self.processors):
                    await self.sink.put(frame)

                if isinstance(initial_frame, EndFrame) or isinstance(initial_frame, EndPipeFrame):
                    break
        except asyncio.CancelledError:
            # this means there's been an interruption, do any cleanup necessary here.
            for processor in self.processors:
                await processor.interrupted()
            pass
