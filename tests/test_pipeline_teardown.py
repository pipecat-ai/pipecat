#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A finished session is freed by reference counting alone.

Each test runs a session with the cyclic garbage collector disabled and keeps
only weak references, so any reference cycle left after teardown keeps the
session alive and fails the test.
"""

import asyncio
import gc
import unittest
import weakref

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    StopFrame,
    TextFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.registry.registry import WorkerRegistry
from pipecat.utils.base_object import BaseObject
from pipecat.workers.runner import WorkerRunner


class PassThrough(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class RaisingProcessor(PassThrough):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            raise RuntimeError("processing failed")


class SlowSetupProcessor(PassThrough):
    async def setup(self, setup):
        await super().setup(setup)
        await asyncio.sleep(10)


class EventObserver(BaseObserver):
    def __init__(self):
        super().__init__()
        self._register_event_handler("on_event")


async def _run_session(processors, *, end: str = "end", **worker_kwargs):
    """Run a session to completion and return weak references to its objects.

    The application's handlers close over the worker and the runner, the way
    bots register them.
    """
    observer = EventObserver()
    worker = PipelineWorker(Pipeline(processors), observers=[observer], **worker_kwargs)
    runner = WorkerRunner(handle_sigint=False)

    @worker.event_handler("on_pipeline_finished")
    async def on_pipeline_finished(worker, frame):
        return runner

    @observer.event_handler("on_event")
    async def on_event(observer):
        return worker

    if worker.turn_tracking_observer:

        @worker.turn_tracking_observer.event_handler("on_turn_ended")
        async def on_turn_ended(observer, *args):
            return worker

    await runner.add_workers(worker)
    await worker.queue_frame(TextFrame("hello"))
    if end == "end":
        await worker.queue_frame(EndFrame())
    elif end == "cancel":

        async def cancel_soon():
            await asyncio.sleep(0.05)
            await runner.cancel()

        cancel_task = asyncio.create_task(cancel_soon())
    await runner.run()
    if end == "cancel":
        await cancel_task

    return [weakref.ref(o) for o in (worker, runner, observer, *processors)]


class TestPipelineTeardown(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        gc.collect()
        gc.disable()

    def tearDown(self):
        gc.enable()

    async def assert_freed(self, refs):
        """The caller must have dropped its own references to the session."""
        # Tasks cancelled during teardown finish on the next loop iteration.
        await asyncio.sleep(0)
        alive = [r() for r in refs if r() is not None]
        self.assertEqual(alive, [], "objects kept alive by reference cycles")

    async def test_ended_session_is_freed(self):
        processors = [PassThrough(), PassThrough(), PassThrough()]
        refs = await _run_session(processors)
        del processors
        await self.assert_freed(refs)

    async def test_cancelled_session_is_freed(self):
        processors = [PassThrough(), PassThrough()]
        refs = await _run_session(processors, end="cancel")
        del processors
        await self.assert_freed(refs)

    async def test_parallel_pipeline_session_is_freed(self):
        parallel = ParallelPipeline([PassThrough()], [PassThrough(), PassThrough()])
        branch_refs = [weakref.ref(p) for p in parallel.processors]
        processors = [PassThrough(), parallel, PassThrough()]
        del parallel
        refs = await _run_session(processors)
        del processors
        await self.assert_freed(refs + branch_refs)

    async def test_bridged_session_is_freed(self):
        processors = [PassThrough(), PassThrough()]
        refs = await _run_session(processors, bridged=())
        del processors
        await self.assert_freed(refs)

    async def test_session_with_processor_error_is_freed(self):
        processors = [PassThrough(), RaisingProcessor(), PassThrough()]
        refs = await _run_session(processors)
        del processors
        await self.assert_freed(refs)

    async def test_session_whose_setup_times_out_is_freed(self):
        processors = [PassThrough(), SlowSetupProcessor()]
        refs = await _run_session(processors, end="none", setup_timeout_secs=0.1)
        del processors
        await self.assert_freed(refs)

    async def test_aggregator_pair_does_not_hold_the_worker(self):
        pair = LLMContextAggregatorPair(LLMContext())
        processors = [PassThrough(), pair.user(), PassThrough(), pair.assistant()]
        del pair
        refs = await _run_session(processors)
        del processors
        worker_ref, runner_ref = refs[:2]
        await self.assert_freed([worker_ref, runner_ref])

    async def test_pipeline_runs_again_in_a_new_worker(self):
        """Pipelines and observers can be used again once a worker is done."""
        first, second = PassThrough(), PassThrough()
        pipeline = Pipeline([first, second])
        observer = EventObserver()
        for _ in range(2):
            worker = PipelineWorker(pipeline, observers=[observer])
            runner = WorkerRunner(handle_sigint=False)
            received = []

            @worker.event_handler("on_frame_reached_downstream")
            async def on_frame_reached_downstream(worker, frame, received=received):
                received.append(frame)

            worker.set_reached_downstream_filter((TextFrame,))
            await runner.add_workers(worker)
            await worker.queue_frames([TextFrame("hello"), EndFrame()])
            await runner.run()
            self.assertEqual([f.text for f in received], ["hello"])
            self.assertIs(first.next, second)
            self.assertIs(second.previous, first)

    async def test_stopped_pipeline_stays_set_up(self):
        """A StopFrame keeps the processors running, so they keep their setup."""
        first, second = PassThrough(), PassThrough()
        worker = PipelineWorker(Pipeline([first, second]))
        runner = WorkerRunner(handle_sigint=False)

        await runner.add_workers(worker)
        await worker.queue_frame(StopFrame())
        await runner.run()

        self.assertIs(first.next, second)
        self.assertIs(second.previous, first)
        self.assertIs(first.pipeline_worker, worker)

    async def test_finished_worker_leaves_a_long_lived_runner_bus(self):
        runner = WorkerRunner(handle_sigint=False)
        run_task = asyncio.create_task(runner.run(auto_end=False))
        await asyncio.sleep(0.05)

        worker = PipelineWorker(Pipeline([PassThrough()]))
        await runner.add_workers(worker)
        self.assertIn(worker.name, runner.bus._subscriptions)
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(worker.wait(), timeout=5)
        await asyncio.sleep(0.05)

        self.assertNotIn(worker.name, runner.bus._subscriptions)
        self.assertIn(runner.name, runner.bus._subscriptions)

        await runner.cancel()
        await run_task
        self.assertNotIn(runner.name, runner.bus._subscriptions)


class TestPreviousLink(unittest.TestCase):
    def test_previous_does_not_keep_the_processor_alive(self):
        first, second = PassThrough(), PassThrough()
        first.link(second)
        self.assertIs(second.previous, first)

        del first
        self.assertIsNone(second.previous)


class _Emitter(BaseObject):
    def __init__(self):
        super().__init__()
        self._register_event_handler("on_event", sync=True)

    async def emit(self):
        await self._call_event_handler("on_event")


class TestClearEventHandlers(unittest.IsolatedAsyncioTestCase):
    async def test_drops_handlers_but_keeps_the_event(self):
        emitter = _Emitter()
        calls = []
        emitter.add_event_handler("on_event", lambda obj: calls.append("before"))

        emitter._clear_event_handlers()
        await emitter.emit()
        self.assertEqual(calls, [])

        emitter.add_event_handler("on_event", lambda obj: calls.append("after"))
        await emitter.emit()
        self.assertEqual(calls, ["after"])


class TestRegistryUnwatch(unittest.IsolatedAsyncioTestCase):
    async def test_unwatch_removes_only_the_given_handler(self):
        registry = WorkerRegistry(runner_name="runner")

        async def first(data):
            pass

        async def second(data):
            pass

        await registry.watch("worker", first)
        await registry.watch("worker", second)
        await registry.unwatch("worker", first)
        self.assertEqual(registry._watches["worker"], [second])

        await registry.unwatch("worker", second)
        self.assertNotIn("worker", registry._watches)

    async def test_unwatch_unknown_handler_is_a_noop(self):
        registry = WorkerRegistry(runner_name="runner")

        async def handler(data):
            pass

        await registry.unwatch("worker", handler)
        self.assertNotIn("worker", registry._watches)


if __name__ == "__main__":
    unittest.main()
