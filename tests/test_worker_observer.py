#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how WorkerObserver hands pushes to the observers it manages."""

import asyncio
import gc
import unittest

from pipecat.frames.frames import TextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker_observer import WorkerObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.asyncio.task_manager import TaskManager


class RecordingObserver(BaseObserver):
    """Records every push of a text frame it is told about."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pushes: list[tuple[str, bool]] = []

    async def on_push_frame(self, data: FramePushed):
        if isinstance(data.frame, TextFrame):
            self.pushes.append((data.source.name, data.first_push))


class TestWorkerObserverInAPipeline(unittest.IsolatedAsyncioTestCase):
    """A frame is pushed again by every processor that passes it along."""

    async def _run(self, *observers: BaseObserver):
        pipeline = Pipeline(
            [
                IdentityFilter(name="first"),
                IdentityFilter(name="second"),
                IdentityFilter(name="third"),
            ]
        )
        await run_test(
            pipeline,
            frames_to_send=[TextFrame("hello"), SleepFrame(sleep=0.1)],
            expected_down_frames=[TextFrame],
            observers=list(observers),
        )

    async def test_an_observer_that_handles_a_frame_once_is_told_once(self):
        observer = RecordingObserver(observe_every_push=False)
        await self._run(observer)
        self.assertEqual(len(observer.pushes), 1)
        self.assertTrue(observer.pushes[0][1])

    async def test_an_observer_gets_every_hop_by_default(self):
        observer = RecordingObserver()
        await self._run(observer)
        sources = [source for source, _ in observer.pushes]
        for name in ("first", "second", "third"):
            self.assertIn(name, sources)
        # Only the first push is marked as such.
        firsts = [first for _, first in observer.pushes]
        self.assertEqual(firsts, [True] + [False] * (len(firsts) - 1))


class TestWorkerObserverMemory(unittest.IsolatedAsyncioTestCase):
    async def test_a_frame_is_forgotten_once_the_pipeline_lets_go_of_it(self):
        observer = RecordingObserver()
        worker_observer = WorkerObserver(observers=[observer])
        await worker_observer.setup(TaskManager())
        source = IdentityFilter(name="source")

        async def push(frame):
            await worker_observer.on_push_frame(
                FramePushed(
                    source=source,
                    destination=source,
                    frame=frame,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )

        for _ in range(10):
            await push(TextFrame("hello"))
        self.assertEqual(len(worker_observer._frames_pushed), 10)

        # The proxy holds on to the last push it handled, so end with a frame
        # the test keeps and wait for it to be handed over.
        kept = TextFrame("kept")
        await push(kept)
        await asyncio.gather(*(proxy.queue.join() for proxy in worker_observer._proxies.values()))
        gc.collect()

        self.assertEqual(list(worker_observer._frames_pushed.values()), [kept])
        self.assertEqual(len(observer.pushes), 11)

        await worker_observer.cleanup()


if __name__ == "__main__":
    unittest.main()
