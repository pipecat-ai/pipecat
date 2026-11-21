#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sequential setup preserves frame admission without reviving old work."""

import asyncio
import io
import unittest

from loguru import logger

from pipecat.frames.frames import StartFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup
from tests.test_frame_processor_cleanup import _RawTaskManager


class _RecordingProcessor(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.texts = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            self.texts.append(frame.text)


class _HeldProcessor(_RecordingProcessor):
    def __init__(self, release):
        super().__init__()
        self.release = release
        self.entered = asyncio.Event()

    async def process_frame(self, frame, direction):
        if isinstance(frame, TextFrame) and frame.text == "held":
            await FrameProcessor.process_frame(self, frame, direction)
            self.entered.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    pass
            return
        await super().process_frame(frame, direction)


class TestFrameProcessorSetupLifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.processors = []
        self.managers = []
        self.operations = []
        self.release_gates = []

    async def asyncTearDown(self):
        # Release every controlled coroutine before awaiting cancellation or
        # cleanup, including when an initial admission/entered assertion fails.
        for gate in self.release_gates:
            gate.set()
        await asyncio.sleep(0)
        for task in self.operations:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.operations, return_exceptions=True)
        for processor in self.processors:
            await processor.cleanup()
        for manager in self.managers:
            tasks = list(manager.current_tasks())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _gate(self):
        gate = asyncio.Event()
        self.release_gates.append(gate)
        return gate

    def _manager(self):
        manager = TaskManager()
        self.managers.append(manager)
        return manager

    async def _setup(self, processor, manager):
        if processor not in self.processors:
            self.processors.append(processor)
        if manager not in self.managers:
            self.managers.append(manager)
        await processor.setup(frame_processor_setup(manager))

    async def _queue_text(self, processor, text, *, interruptible=True):
        done = asyncio.Event()

        async def callback(*args):
            done.set()

        frame = TextFrame(text)
        frame.interruptible = interruptible
        await processor.queue_frame(frame, callback=callback)
        return done

    async def _start_and_record(self, processor, text):
        await processor.queue_frame(StartFrame())
        done = await self._queue_text(processor, text)
        await asyncio.wait_for(done.wait(), timeout=2)

    async def _held_processor(self, manager):
        processor = _HeldProcessor(self._gate())
        await self._setup(processor, manager)
        await processor.queue_frame(StartFrame())
        await processor.queue_frame(TextFrame("held"))
        await asyncio.wait_for(processor.entered.wait(), timeout=2)
        return processor

    async def _held_event(self, processor, release):
        entered = asyncio.Event()
        processor._register_event_handler("on_lifecycle_hold")

        @processor.event_handler("on_lifecycle_hold")
        async def hold(processor):
            entered.set()
            await release.wait()

        await processor._call_event_handler("on_lifecycle_hold")
        tasks = [task for name, task in processor._event_tasks if name == "on_lifecycle_hold"]
        self.operations.extend(tasks)
        await asyncio.wait_for(entered.wait(), timeout=2)
        return tasks

    async def test_direct_setup_reuses_a_fully_cleaned_processor(self):
        processor = _RecordingProcessor()
        await self._setup(processor, self._manager())
        await self._start_and_record(processor, "first")
        await processor.cleanup()

        second_manager = self._manager()
        await self._setup(processor, second_manager)
        await self._start_and_record(processor, "second")

        self.assertIs(processor.task_manager, second_manager)
        self.assertEqual(processor.texts, ["first", "second"])

    async def test_marker_preserves_initial_and_reused_early_frames(self):
        processor = _RecordingProcessor()
        self.processors.append(processor)
        for text in ("initial early frame", "reused early frame"):
            with self.subTest(text=text):
                processor._setup_started()
                done = await self._queue_text(processor, text)
                await processor.queue_frame(StartFrame())
                await self._setup(processor, self._manager())
                await asyncio.sleep(0)
                self.assertFalse(done.is_set())

                processor._setup_finished()
                processor._setup_finished()
                await asyncio.wait_for(done.wait(), timeout=2)
                await processor.cleanup()

        self.assertEqual(processor.texts, ["initial early frame", "reused early frame"])

    async def test_unsafe_marker_stays_blocked_and_fresh_attempt_discards_old_frames(self):
        old_manager = self._manager()
        processor = await self._held_processor(old_manager)
        stale = await self._queue_text(processor, "old uninterruptible", interruptible=False)
        await asyncio.wait_for(processor.cleanup(), timeout=3)
        old_tasks = list(old_manager.current_tasks())
        self.assertTrue(old_tasks)
        self.operations.extend(old_tasks)

        new_manager = self._manager()
        with self.assertRaises(RuntimeError):
            await self._setup(processor, new_manager)
        self.assertIs(processor.task_manager, old_manager)

        processor._setup_started()
        await processor.queue_frame(StartFrame())
        dropped = await self._queue_text(processor, "unsafe early frame")
        processor.release.set()
        await asyncio.gather(*old_tasks, return_exceptions=True)
        with self.assertRaises(RuntimeError):
            await self._setup(processor, new_manager)
        processor._setup_finished()
        self.assertEqual(old_manager.current_tasks(), [])
        self.assertFalse(stale.is_set())
        self.assertFalse(dropped.is_set())

        processor._setup_started()
        await processor.queue_frame(StartFrame())
        fresh = await self._queue_text(processor, "fresh")
        await self._setup(processor, new_manager)
        processor._setup_finished()
        await asyncio.wait_for(fresh.wait(), timeout=2)

        self.assertEqual(processor.texts, ["fresh"])
        self.assertFalse(stale.is_set())
        self.assertFalse(dropped.is_set())

    async def test_running_cleanup_rejects_setup_without_rebinding_manager(self):
        processor = _RecordingProcessor(enable_direct_mode=True)
        old_manager = self._manager()
        await self._setup(processor, old_manager)
        release = self._gate()
        await self._held_event(processor, release)
        cleanup = asyncio.create_task(processor.cleanup())
        self.operations.append(cleanup)
        await asyncio.sleep(0)
        self.assertFalse(cleanup.done())

        new_manager = self._manager()
        with self.assertRaises(RuntimeError):
            await self._setup(processor, new_manager)
        self.assertIs(processor.task_manager, old_manager)

        release.set()
        await asyncio.wait_for(cleanup, timeout=2)
        await self._setup(processor, new_manager)
        await self._start_and_record(processor, "after cleanup")
        self.assertEqual(processor.texts, ["after cleanup"])

    async def test_cancelled_event_drain_requires_a_successful_cleanup_retry(self):
        processor = _RecordingProcessor(enable_direct_mode=True)
        old_manager = self._manager()
        await self._setup(processor, old_manager)
        release = self._gate()
        event_tasks = await self._held_event(processor, release)
        cleanup = asyncio.create_task(processor.cleanup())
        self.operations.append(cleanup)
        await asyncio.sleep(0)
        cleanup.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cleanup

        new_manager = self._manager()
        with self.assertRaises(RuntimeError):
            await self._setup(processor, new_manager)
        release.set()
        await asyncio.gather(*event_tasks)
        with self.assertRaises(RuntimeError):
            await self._setup(processor, new_manager)
        self.assertIs(processor.task_manager, old_manager)

        await processor.cleanup()
        await self._setup(processor, new_manager)
        await self._start_and_record(processor, "after retry")
        self.assertEqual(processor.texts, ["after retry"])

    async def test_successful_event_self_cleanup_waits_for_its_current_handler(self):
        processor = _RecordingProcessor(enable_direct_mode=True)
        old_manager = self._manager()
        await self._setup(processor, old_manager)
        release = self._gate()
        cleanup_returned = asyncio.Event()
        processor._register_event_handler("on_self_cleanup")

        @processor.event_handler("on_self_cleanup")
        async def self_cleanup(processor):
            await processor.cleanup()
            cleanup_returned.set()
            await release.wait()

        await processor._call_event_handler("on_self_cleanup")
        event_tasks = [task for name, task in processor._event_tasks if name == "on_self_cleanup"]
        self.operations.extend(event_tasks)
        await asyncio.wait_for(cleanup_returned.wait(), timeout=2)
        new_manager = self._manager()
        with self.assertRaises(RuntimeError):
            await self._setup(processor, new_manager)
        self.assertIs(processor.task_manager, old_manager)

        release.set()
        await asyncio.gather(*event_tasks)
        await self._setup(processor, new_manager)
        await self._start_and_record(processor, "after handler")
        self.assertEqual(processor.texts, ["after handler"])

    async def test_unrelated_shared_manager_task_does_not_block_reuse(self):
        processor = _RecordingProcessor()
        manager = self._manager()
        await self._setup(processor, manager)
        release = self._gate()
        foreign = manager.create_task(release.wait(), name="another processor's task")
        self.operations.append(foreign)
        await self._start_and_record(processor, "first")
        await processor.cleanup()
        self.assertFalse(foreign.done())

        await self._setup(processor, manager)
        await self._start_and_record(processor, "second")
        self.assertFalse(foreign.done())
        self.assertEqual(processor.texts, ["first", "second"])

    async def test_cleanup_invalidates_a_prepared_setup_attempt(self):
        processor = _RecordingProcessor()
        await self._setup(processor, self._manager())
        await self._start_and_record(processor, "old")
        await processor.cleanup()

        processor._setup_started()
        await processor.queue_frame(StartFrame())
        abandoned = await self._queue_text(processor, "abandoned")
        await processor.cleanup()
        manager = self._manager()
        with self.assertRaises(RuntimeError):
            await self._setup(processor, manager)
        processor._setup_finished()
        self.assertFalse(abandoned.is_set())

        processor._setup_started()
        await processor.queue_frame(StartFrame())
        fresh = await self._queue_text(processor, "new")
        await self._setup(processor, manager)
        processor._setup_finished()
        await asyncio.wait_for(fresh.wait(), timeout=2)

        self.assertEqual(processor.texts, ["old", "new"])
        self.assertFalse(abandoned.is_set())

    async def test_done_observer_is_consumed_once_before_rebinding(self):
        old_manager = _RawTaskManager(
            completion_error=RuntimeError("previous lifecycle completion")
        )
        processor = await self._held_processor(old_manager)
        matches = [
            task
            for task in old_manager.current_tasks()
            if "__process_frame_task_handler" in task.get_name()
        ]
        self.assertEqual(len(matches), 1)
        old_task = matches[0]
        self.operations.append(old_task)
        prepared = asyncio.Event()
        marker_errors = []

        def prepare_next_setup(task):
            try:
                processor._setup_started()
            except BaseException as error:
                marker_errors.append(error)
            finally:
                prepared.set()

        # Register before cleanup installs its completion observer, so this
        # marker sees a done task whose result callback has not yet run.
        old_task.add_done_callback(prepare_next_setup)
        output = io.StringIO()
        handler_id = logger.add(output, level="ERROR")
        try:
            await asyncio.wait_for(processor.cleanup(), timeout=3)
            self.assertFalse(old_task.done())
            processor.release.set()
            await asyncio.wait_for(prepared.wait(), timeout=2)
            self.assertEqual(marker_errors, [])

            manager = self._manager()
            await processor.queue_frame(StartFrame())
            fresh = await self._queue_text(processor, "after observer")
            await self._setup(processor, manager)
            processor._setup_finished()
            await asyncio.wait_for(fresh.wait(), timeout=2)
            self.assertEqual(processor.texts, ["after observer"])
            self.assertEqual(output.getvalue().count("previous lifecycle completion"), 1)
            self.assertIs(processor.task_manager, manager)
        finally:
            logger.remove(handler_id)


if __name__ == "__main__":
    unittest.main()
