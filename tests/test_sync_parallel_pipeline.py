#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from dataclasses import dataclass

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.sync_parallel_pipeline import FrameOrder, SyncParallelPipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import QueuedFrameProcessor, run_test
from pipecat.workers.runner import WorkerRunner


@dataclass
class TaggedFrame(Frame):
    """A simple tagged frame for testing pipeline ordering."""

    tag: str = ""

    def __str__(self):
        return f"{self.name}(tag: {self.tag})"


class EmitTaggedFrameProcessor(FrameProcessor):
    """Emits a TaggedFrame for every TextFrame it receives.

    Used to produce distinguishable output from different pipelines so tests
    can verify ordering.
    """

    def __init__(self, tag: str, *, delay: float = 0, **kwargs):
        super().__init__(**kwargs)
        self._tag = tag
        self._delay = delay

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            await self.push_frame(TaggedFrame(tag=self._tag))
        else:
            await self.push_frame(frame, direction)


async def end_pipeline(pipeline: SyncParallelPipeline, frames: list[Frame]) -> list[Frame]:
    """Send frames and an EndFrame through the pipeline and return what came out.

    The worker waits for the EndFrame for as long as it takes, so if it doesn't
    come out the run is ended with a CancelFrame pushed from inside.
    """
    head = IdentityFilter()
    received = asyncio.Queue()
    sink = QueuedFrameProcessor(queue=received, queue_direction=FrameDirection.DOWNSTREAM)
    worker = PipelineWorker(Pipeline([head, pipeline, sink]))
    await worker.queue_frames([*frames, EndFrame()])

    runner = WorkerRunner()
    await runner.add_workers(worker)
    run = asyncio.create_task(runner.run())
    done, _ = await asyncio.wait([run], timeout=2.0)
    if not done:
        await head.push_frame(CancelFrame())
        await run

    return [received.get_nowait() for _ in range(received.qsize())]


class TestSyncParallelPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_dedup_multiple_frames(self):
        """Identical frames from multiple paths should be deduplicated."""
        pipeline = SyncParallelPipeline([IdentityFilter()], [IdentityFilter()])

        frames_to_send = [TextFrame(text="one"), TextFrame(text="two")]
        expected_down_frames = [TextFrame, TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_arrival_order(self):
        """With FrameOrder.ARRIVAL, a slow first pipeline's frames should
        arrive after a fast second pipeline's frames."""
        pipeline = SyncParallelPipeline(
            [EmitTaggedFrameProcessor("slow", delay=0.05)],
            [EmitTaggedFrameProcessor("fast")],
            frame_order=FrameOrder.ARRIVAL,
        )

        frames_to_send = [TextFrame(text="one"), TextFrame(text="two")]
        (down_frames, _) = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )

        tags = [f.tag for f in down_frames if isinstance(f, TaggedFrame)]
        assert tags == [
            "fast",
            "slow",
            "fast",
            "slow",
        ], f"Expected fast before slow in each batch, got {tags}"

    async def test_pipeline_order(self):
        """With FrameOrder.PIPELINE and multiple input frames, each batch
        should follow pipeline definition order regardless of processing speed."""
        pipeline = SyncParallelPipeline(
            [EmitTaggedFrameProcessor("slow", delay=0.05)],
            [EmitTaggedFrameProcessor("fast")],
            frame_order=FrameOrder.PIPELINE,
        )

        frames_to_send = [TextFrame(text="one"), TextFrame(text="two")]
        (down_frames, _) = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
        )

        tags = [f.tag for f in down_frames if isinstance(f, TaggedFrame)]
        assert tags == [
            "slow",
            "fast",
            "slow",
            "fast",
        ], f"Expected pipeline definition order (slow, fast) in each batch, got {tags}"

    async def test_end_frame_after_pipeline_output(self):
        """An EndFrame comes out after the output a pipeline pushes when it ends."""
        pipeline = SyncParallelPipeline([SentenceAggregator()])

        down_frames = await end_pipeline(pipeline, [TextFrame(text="Hello")])

        assert [type(f) for f in down_frames] == [TextFrame, EndFrame]

    async def test_end_frame_after_every_pipeline_output(self):
        """An EndFrame comes out once, after the output of every pipeline."""
        for frame_order in FrameOrder:
            pipeline = SyncParallelPipeline(
                [IdentityFilter()], [SentenceAggregator()], frame_order=frame_order
            )

            down_frames = await end_pipeline(pipeline, [TextFrame(text="Hello")])

            assert [type(f) for f in down_frames] == [TextFrame, TextFrame, EndFrame], frame_order

    async def test_default_is_arrival(self):
        """The default frame_order should be ARRIVAL."""
        pipeline = SyncParallelPipeline([IdentityFilter()])
        assert pipeline._frame_order == FrameOrder.ARRIVAL


if __name__ == "__main__":
    unittest.main()
