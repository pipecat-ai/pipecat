#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Runtime observer registration through the public pipeline worker API."""

import asyncio
import unittest

from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, WorkerParams
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.utils.asyncio.task_manager import TaskManager


class RecordingObserver(BaseObserver):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.received = asyncio.Event()
        self.texts = []

    async def on_push_frame(self, data: FramePushed):
        if data.source is self.source and isinstance(data.frame, TextFrame):
            self.texts.append(data.frame.text)
            self.received.set()


class TestRuntimeObserverRegistration(unittest.IsolatedAsyncioTestCase):
    async def test_add_first_observer_after_start(self):
        await self._check_runtime_registration(replace_last=False)

    async def test_add_observer_after_removing_last(self):
        await self._check_runtime_registration(replace_last=True)

    async def _check_runtime_registration(self, *, replace_last: bool):
        source = IdentityFilter()
        original = RecordingObserver(source)
        added = RecordingObserver(source)
        worker = PipelineWorker(
            Pipeline([source]),
            observers=[original] if replace_last else [],
            enable_rtvi=False,
            enable_turn_tracking=False,
            idle_timeout_secs=None,
        )
        started = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(worker, frame):
            started.set()

        run = asyncio.create_task(worker.run(WorkerParams(task_manager=TaskManager())))
        try:
            await asyncio.wait_for(started.wait(), timeout=10)
            if replace_last:
                await worker.queue_frame(TextFrame(text="before replacement"))
                await asyncio.wait_for(original.received.wait(), timeout=2)
                await worker.remove_observer(original)

            worker.add_observer(added)
            await worker.queue_frame(TextFrame(text="after registration"))
            await asyncio.wait_for(added.received.wait(), timeout=2)
            self.assertEqual(added.texts, ["after registration"])
            self.assertEqual(original.texts, ["before replacement"] if replace_last else [])
        finally:
            await worker.queue_frame(EndFrame())
            await asyncio.wait_for(run, timeout=10)
