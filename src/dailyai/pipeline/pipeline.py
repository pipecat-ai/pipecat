import asyncio
from typing import AsyncGenerator, List
from dailyai.pipeline.frame_processor import FrameProcessor

from dailyai.pipeline.frames import EndParallelPipeQueueFrame, EndStreamQueueFrame, QueueFrame


class Pipeline:
    def __init__(
        self,
        source: asyncio.Queue,
        sink: asyncio.Queue[QueueFrame],
        processors: List[FrameProcessor],
    ):
        self.source: asyncio.Queue[QueueFrame] = source
        self.sink: asyncio.Queue[QueueFrame] = sink
        self.processors = processors

    async def get_next_source_frame(self) -> AsyncGenerator[QueueFrame, None]:
        yield await self.source.get()

    async def run_pipeline(self):
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
                        frame, EndStreamQueueFrame
                    ) or isinstance(
                        frame, EndParallelPipeQueueFrame
                    ):
                        return
        except asyncio.CancelledError:
            # this means there's been an interruption, do any cleanup necessary here.
            for processor in self.processors:
                await processor.interrupted()
            pass
