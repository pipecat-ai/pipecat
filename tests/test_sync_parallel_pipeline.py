#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from dataclasses import dataclass
from weakref import ref

from pipecat.frames.frames import Frame, InputAudioRawFrame, SystemFrame, TextFrame
from pipecat.pipeline.sync_parallel_pipeline import (
    FrameOrder,
    SyncParallelPipeline,
    SyncParallelPipelineSink,
    SyncParallelPipelineSource,
)
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test


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


class CaptureFramesProcessor(FrameProcessor):
    """Records frames while preserving their direction and identity."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen_frame_ids: list[int] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.seen_frame_ids.append(frame.id)
        await self.push_frame(frame, direction)


@dataclass
class SampleSystemFrame(SystemFrame):
    """System frame used to verify branch bookend behavior."""

    pass


class TestSyncParallelPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_system_frames_downstream_are_delivered_once(self):
        """System frames should traverse every branch without being emitted twice."""
        first_branch = CaptureFramesProcessor()
        second_branch = CaptureFramesProcessor()
        pipeline = SyncParallelPipeline([first_branch], [second_branch])

        audio_frames = [
            InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
            for _ in range(3)
        ]
        text_frames = [TextFrame(text="one"), TextFrame(text="two")]
        frames_to_send = audio_frames + text_frames

        down_frames, _ = await run_test(pipeline, frames_to_send=frames_to_send, start_timeout=5.0)

        assert [frame.id for frame in down_frames] == [frame.id for frame in frames_to_send]
        for branch in (first_branch, second_branch):
            for frame in audio_frames:
                assert branch.seen_frame_ids.count(frame.id) == 1

    async def test_bookends_do_not_retain_system_frames(self):
        """Bookends should suppress fanned copies while preserving branch-generated frames."""
        upstream_queue = asyncio.Queue()
        downstream_queue = asyncio.Queue()
        source = SyncParallelPipelineSource(upstream_queue)
        sink = SyncParallelPipelineSink(downstream_queue)

        fanned_upstream = SampleSystemFrame()
        fanned_downstream = SampleSystemFrame()
        source._register_bypassed_system_frame(fanned_upstream)
        sink._register_bypassed_system_frame(fanned_downstream)

        await source.process_frame(fanned_upstream, FrameDirection.UPSTREAM)
        await sink.process_frame(fanned_downstream, FrameDirection.DOWNSTREAM)

        assert upstream_queue.empty()
        assert downstream_queue.empty()

        generated_upstream = SampleSystemFrame()
        generated_downstream = SampleSystemFrame()
        await source.process_frame(generated_upstream, FrameDirection.UPSTREAM)
        await sink.process_frame(generated_downstream, FrameDirection.DOWNSTREAM)

        assert await upstream_queue.get() is generated_upstream
        assert await downstream_queue.get() is generated_downstream

        swallowed_frame = SampleSystemFrame()
        swallowed_frame_ref = ref(swallowed_frame)
        source._register_bypassed_system_frame(swallowed_frame)
        del swallowed_frame

        assert swallowed_frame_ref() is None

    async def test_system_frames_upstream_are_delivered_once(self):
        """Upstream system frames should traverse every branch without being emitted twice."""
        first_branch = CaptureFramesProcessor()
        second_branch = CaptureFramesProcessor()
        pipeline = SyncParallelPipeline([first_branch], [second_branch])

        system_frame = SampleSystemFrame()
        data_frame = TextFrame(text="upstream")
        _, up_frames = await run_test(
            pipeline,
            # Keep the harness's downstream EndFrame from racing the upstream sync.
            frames_to_send=[system_frame, data_frame, SleepFrame(sleep=0.1)],
            frames_to_send_direction=FrameDirection.UPSTREAM,
            start_timeout=5.0,
        )

        assert [frame.id for frame in up_frames] == [system_frame.id, data_frame.id]
        for branch in (first_branch, second_branch):
            assert branch.seen_frame_ids.count(system_frame.id) == 1

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

    async def test_default_is_arrival(self):
        """The default frame_order should be ARRIVAL."""
        pipeline = SyncParallelPipeline([IdentityFilter()])
        assert pipeline._frame_order == FrameOrder.ARRIVAL


if __name__ == "__main__":
    unittest.main()
