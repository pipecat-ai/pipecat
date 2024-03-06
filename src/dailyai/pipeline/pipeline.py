import asyncio
from typing import AsyncGenerator, List
from dailyai.pipeline.frame_processor import FrameProcessor

from dailyai.pipeline.frames import EndPipeFrame, EndFrame, Frame

"""
This class manages a pipe of FrameProcessors, and runs them in sequence. The "source"
and "sink" queues are managed by the caller. You can use this class stand-alone to
perform specialized processing, or you can use the Transport's run_pipeline method to
instantiate and run a pipeline with the Transport's sink and source queues.
"""

class Pipeline:

    def __init__(
        self,
        processors: List[FrameProcessor],
        source: asyncio.Queue | None = None,
        sink: asyncio.Queue[Frame] | None = None,
    ):
        self.processors = processors
        self.source: asyncio.Queue[Frame] | None = source
        self.sink: asyncio.Queue[Frame] | None = sink

    def set_source(self, source: asyncio.Queue[Frame]):
        self.source = source

    def set_sink(self, sink: asyncio.Queue[Frame]):
        self.sink = sink

    async def get_next_source_frame(self) -> AsyncGenerator[Frame, None]:
        if self.source is None:
            raise ValueError("Source queue not set")
        yield await self.source.get()

    async def run_pipeline(self):
        if self.source is None or self.sink is None:
            raise ValueError("Source or sink queue not set")

        try:
            while True:
                frame_generators = [self.get_next_source_frame()]
                for processor in self.processors:
                    next_frame_generators = []
                    for frame_generator in frame_generators:
                        async for frame in frame_generator:
                            next_frame_generators.append(processor.process_frame(frame))
                    frame_generators = next_frame_generators

                for frame_generator in frame_generators:
                    async for frame in frame_generator:
                        await self.sink.put(frame)
                        if isinstance(
                            frame, EndFrame
                        ) or isinstance(
                            frame, EndPipeFrame
                        ):
                            return
        except asyncio.CancelledError:
            # this means there's been an interruption, do any cleanup necessary here.
            for processor in self.processors:
                await processor.interrupted()
            pass
