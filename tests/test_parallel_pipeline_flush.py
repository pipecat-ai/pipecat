#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Exercise real worker flushes through parallel branches and service filters."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from pipecat.frames.frames import ManuallySwitchServiceFrame, PipelineFlushFrame, TextFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.service_switcher import ServiceSwitcher
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner


@asynccontextmanager
async def running_pipeline(*processors):
    worker = PipelineWorker(
        Pipeline(list(processors)),
        enable_rtvi=False,
        enable_turn_tracking=False,
        idle_timeout_secs=None,
        cancel_timeout_secs=1,
    )
    ready = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def on_started(worker, frame):
        ready.set()

    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        yield worker
    finally:
        try:
            await asyncio.wait_for(worker.cancel(), timeout=2)
            await asyncio.wait_for(asyncio.shield(task), timeout=3)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


class Recorder(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.probes = []
        self.texts = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, PipelineFlushFrame):
            self.probes.append((frame.id, direction, frame.returning))
        elif isinstance(frame, TextFrame):
            self.texts.append(frame.text)
        await self.push_frame(frame, direction)


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["ordinary", "parallel", "nested", "switcher"])
async def test_idle_pipeline_flush(layout):
    if layout == "ordinary":
        middle = IdentityFilter()
    elif layout == "parallel":
        middle = ParallelPipeline([IdentityFilter()], [IdentityFilter()])
    elif layout == "nested":
        middle = ParallelPipeline(
            [ParallelPipeline([IdentityFilter()], [IdentityFilter()])], [IdentityFilter()]
        )
    else:
        middle = ServiceSwitcher(services=[IdentityFilter(), IdentityFilter()])
    before, after = Recorder(), Recorder()
    async with running_pipeline(before, middle, after) as worker:
        assert await asyncio.wait_for(worker.flush_pipeline(timeout=0.1), timeout=2)
        expected = [
            (FrameDirection.DOWNSTREAM, False),
            (FrameDirection.UPSTREAM, False),
            (FrameDirection.DOWNSTREAM, True),
        ]
        for recorder in (before, after):
            assert [
                (direction, returning) for _, direction, returning in recorder.probes
            ] == expected
            assert len({frame_id for frame_id, _, _ in recorder.probes}) == 1


@pytest.mark.asyncio
async def test_flush_after_switching_service():
    first, second = Recorder(), Recorder()
    switcher = ServiceSwitcher(services=[first, second])
    output = Recorder()
    async with running_pipeline(switcher, output) as worker:
        await worker.queue_frame(TextFrame("first"))
        assert await asyncio.wait_for(worker.flush_pipeline(timeout=0.1), timeout=2)
        await worker.queue_frame(ManuallySwitchServiceFrame(service=second))
        await worker.queue_frame(TextFrame("second"))
        assert await asyncio.wait_for(worker.flush_pipeline(timeout=0.1), timeout=2)
        assert first.texts == ["first"]
        assert second.texts == ["second"]
        assert output.texts == ["first", "second"]


class ProbeGate(Recorder):
    def __init__(self, direction, returning):
        super().__init__()
        self.direction = direction
        self.returning = returning
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def process_frame(self, frame, direction):
        if (
            isinstance(frame, PipelineFlushFrame)
            and direction == self.direction
            and frame.returning == self.returning
        ):
            self.entered.set()
            await self.release.wait()
        await super().process_frame(frame, direction)


class DuplicateProbes(Recorder):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, PipelineFlushFrame):
            await self.push_frame(frame, direction)


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["parallel", "nested", "switcher"])
@pytest.mark.parametrize(
    "direction, returning",
    [
        (FrameDirection.DOWNSTREAM, False),
        (FrameDirection.UPSTREAM, False),
        (FrameDirection.DOWNSTREAM, True),
    ],
)
async def test_every_flush_leg_waits_for_each_branch(layout, direction, returning):
    gate = ProbeGate(direction, returning)
    if layout == "switcher":
        # The gated service is inactive; its drain probe still has to complete.
        middle = ServiceSwitcher(services=[DuplicateProbes(), gate])
    else:
        middle = ParallelPipeline([DuplicateProbes()], [gate])
        if layout == "nested":
            middle = ParallelPipeline([middle], [DuplicateProbes()])
    output = Recorder()
    async with running_pipeline(middle, output) as worker:
        flushing = asyncio.create_task(worker.flush_pipeline(timeout=0.5))
        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=2)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(flushing), timeout=0.03)
        finally:
            gate.release.set()
            flushed = await asyncio.wait_for(flushing, timeout=2)
        assert flushed
        assert len(output.probes) == 3


@pytest.mark.asyncio
async def test_concurrent_and_repeated_flushes_keep_separate_trips():
    output = Recorder()
    middle = ParallelPipeline([IdentityFilter()], [IdentityFilter()])
    async with running_pipeline(middle, output) as worker:
        results = await asyncio.wait_for(
            asyncio.gather(*(worker.flush_pipeline(timeout=0.5) for _ in range(3))), timeout=3
        )
        assert results == [True, True, True]
        for _ in range(2):
            assert await asyncio.wait_for(worker.flush_pipeline(timeout=0.5), timeout=2)
        frame_ids = {frame_id for frame_id, _, _ in output.probes}
        assert len(frame_ids) == 5
        for frame_id in frame_ids:
            assert [(d, r) for i, d, r in output.probes if i == frame_id] == [
                (FrameDirection.DOWNSTREAM, False),
                (FrameDirection.UPSTREAM, False),
                (FrameDirection.DOWNSTREAM, True),
            ]


@pytest.mark.asyncio
async def test_swallowed_probe_times_out_without_stalling_later_work():
    class DropFirstProbe(Recorder):
        dropped = False

        async def process_frame(self, frame, direction):
            if isinstance(frame, PipelineFlushFrame) and not self.dropped:
                self.dropped = True
                return
            await super().process_frame(frame, direction)

    middle = ParallelPipeline([IdentityFilter()], [DropFirstProbe()])
    output = Recorder()
    async with running_pipeline(middle, output) as worker:
        assert not await asyncio.wait_for(worker.flush_pipeline(timeout=0.05), timeout=2)
        await worker.queue_frame(TextFrame("after timeout"))
        assert await asyncio.wait_for(worker.flush_pipeline(timeout=0.5), timeout=2)
        assert output.texts == ["after timeout"]
        assert not middle._flush_branches


@pytest.mark.asyncio
async def test_flush_waits_for_work_triggered_upstream():
    entered, release = asyncio.Event(), asyncio.Event()

    class RespondUpstream(Recorder):
        async def process_frame(self, frame, direction):
            if isinstance(frame, TextFrame) and direction == FrameDirection.UPSTREAM:
                entered.set()
                await release.wait()
                await self.push_frame(TextFrame("response"))
            else:
                await super().process_frame(frame, direction)

    class TriggerUpstream(Recorder):
        async def process_frame(self, frame, direction):
            if isinstance(frame, TextFrame) and frame.text == "trigger":
                await self.push_frame(TextFrame("request"), FrameDirection.UPSTREAM)
            else:
                await super().process_frame(frame, direction)

    middle = ParallelPipeline([TriggerUpstream()], [IdentityFilter()])
    output = Recorder()
    async with running_pipeline(RespondUpstream(), middle, output) as worker:
        await worker.queue_frame(TextFrame("trigger"))
        flushing = asyncio.create_task(worker.flush_pipeline(timeout=0.5))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(flushing), timeout=0.03)
        finally:
            release.set()
            flushed = await asyncio.wait_for(flushing, timeout=2)
        assert flushed
        assert output.texts.count("response") == 1
