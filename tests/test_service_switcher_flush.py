#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import InputAudioRawFrame, PipelineFlushFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.service_switcher import ServiceSwitcher
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner


class Passthrough(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.flush_passes = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, PipelineFlushFrame):
            self.flush_passes.append((frame.id, direction, frame.returning))
        await self.push_frame(frame, direction)


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("continuous_audio", [False, True])
@pytest.mark.parametrize("flush_count", [1, 2])
async def test_flush_round_trip_through_active_services(nested, continuous_audio, flush_count):
    active = Passthrough()
    inactive = Passthrough()
    switcher = ServiceSwitcher(services=[active, inactive])
    if nested:
        switcher = ServiceSwitcher(services=[switcher, Passthrough()])
    worker = PipelineWorker(Pipeline([switcher, Passthrough()]), enable_rtvi=False)
    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def on_started(worker, frame):
        started.set()

    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    stop_audio = asyncio.Event()

    async def feed_audio():
        while not stop_audio.is_set():
            await worker.queue_frame(
                InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
            )
            await asyncio.sleep(0.005)

    audio = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        if continuous_audio:
            audio = asyncio.create_task(feed_audio())
        results = await asyncio.gather(
            *(asyncio.wait_for(worker.flush_pipeline(timeout=0.05), 1) for _ in range(flush_count))
        )
        assert all(results)
        flush_ids = {frame_id for frame_id, _, _ in active.flush_passes}
        assert len(flush_ids) == flush_count
        for flush_id in flush_ids:
            assert [
                (direction, returning)
                for frame_id, direction, returning in active.flush_passes
                if frame_id == flush_id
            ] == [
                (FrameDirection.DOWNSTREAM, False),
                (FrameDirection.UPSTREAM, False),
                (FrameDirection.DOWNSTREAM, True),
            ]
        assert inactive.flush_passes == []
    finally:
        stop_audio.set()
        if audio:
            await audio
        await runner.cancel()
        await asyncio.wait_for(running, 5)


@pytest.mark.asyncio
async def test_deduplication_is_per_flush_pass_and_unchanged_for_other_frames():
    switcher = ServiceSwitcher(services=[Passthrough(), Passthrough()])
    frame = PipelineFlushFrame(event=asyncio.Event())
    with patch.object(switcher, "push_frame", new_callable=AsyncMock) as push:
        # A probe arriving while a lifecycle frame is being synchronized is
        # buffered, then delivered when the buffer flushes.
        switcher._synchronizing = True
        await switcher._parallel_push_frame(frame, FrameDirection.DOWNSTREAM)
        assert push.await_count == 0
        await switcher._flush_buffered_frames()
        assert push.await_count == 1
        switcher._synchronizing = False
        frame = PipelineFlushFrame(event=asyncio.Event())

        for direction, returning in [
            (FrameDirection.DOWNSTREAM, False),
            (FrameDirection.UPSTREAM, False),
            (FrameDirection.DOWNSTREAM, True),
        ]:
            frame.returning = returning
            await switcher._parallel_push_frame(frame, direction)
            await switcher._parallel_push_frame(frame, direction)
        assert push.await_count == 4

        text = TextFrame("ordinary output")
        await switcher._parallel_push_frame(text, FrameDirection.DOWNSTREAM)
        await switcher._parallel_push_frame(text, FrameDirection.DOWNSTREAM)
        await switcher._parallel_push_frame(text, FrameDirection.UPSTREAM)
        assert push.await_count == 5
