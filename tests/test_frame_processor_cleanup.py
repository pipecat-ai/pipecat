#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Final processor teardown retains cancellation-resistant task ownership."""

import asyncio
import inspect
import io
import unittest

from loguru import logger

from pipecat.frames.frames import InterruptionFrame, StartFrame, SystemFrame, TextFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup


class _RecordingProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(enable_direct_mode=True)
        self.texts = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            self.texts.append(frame.text)


class _ResistantProcessor(FrameProcessor):
    def __init__(self, *, system=False):
        super().__init__()
        self.system = system
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.requests = 0
        self.processed = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            return
        self.processed.append(frame)
        if isinstance(frame, SystemFrame) != self.system:
            return
        self.entered.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.requests += 1
                self.cancelled.set()
        await self.push_frame(TextFrame("late output"), direction)


class _RawTaskManager(TaskManager):
    """A supported custom manager without the default exception-swallowing wrapper."""

    def __init__(self, completion_error=None, completion_task="__process_frame_task_handler"):
        super().__init__()
        self.live = set()
        self.created = []
        self.completion_error = completion_error
        self.completion_task = completion_task

    def create_task(self, coroutine, name, context=None):
        coroutine_name = coroutine.cr_code.co_name
        self.created.append(coroutine_name)

        async def run():
            await coroutine
            if self.completion_error is not None and coroutine_name == self.completion_task:
                raise self.completion_error

        task = self.get_event_loop().create_task(run(), name=name, context=context)

        def close_unstarted(task):
            if inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED:
                coroutine.close()

        task.add_done_callback(close_unstarted)
        self.live.add(task)
        task.add_done_callback(self.live.discard)
        return task

    def current_tasks(self):
        return list(self.live)


class TestFrameProcessorCleanup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = TaskManager()
        self.processors = []
        self.operations = []

    async def asyncTearDown(self):
        for processor in self.processors:
            if isinstance(processor, _ResistantProcessor):
                processor.release.set()
                input_release = getattr(processor, "input_release", None)
                if input_release is not None:
                    input_release.set()
        await asyncio.sleep(0)
        for operation in self.operations:
            if not operation.done():
                operation.cancel()
        await asyncio.gather(*self.operations, return_exceptions=True)
        for task in self.manager.current_tasks():
            task.cancel()
        await asyncio.gather(*self.manager.current_tasks(), return_exceptions=True)
        for processor in self.processors:
            await processor.cleanup()

    async def _setup(self, processor, **kwargs):
        self.processors.append(processor)
        await processor.setup(frame_processor_setup(self.manager, **kwargs))
        return processor

    async def _start_resistant(self, *, system=False, callback=None):
        processor = await self._setup(_ResistantProcessor(system=system))
        await processor.queue_frame(StartFrame())
        frame = SystemFrame() if system else TextFrame("held")
        await processor.queue_frame(frame, callback=callback)
        await asyncio.wait_for(processor.entered.wait(), timeout=2)
        return processor

    async def test_process_timeout_retains_owner_and_fences_late_work(self):
        callbacks = []
        after_events = []

        async def callback(*args):
            callbacks.append(args)

        processor = await self._start_resistant(callback=callback)
        sink = await self._setup(_RecordingProcessor())
        processor.link(sink)

        @processor.event_handler("on_after_process_frame")
        async def after_process(processor, frame):
            after_events.append(frame)

        await processor.queue_frame(TextFrame("queued before cleanup"))
        cleanup = asyncio.create_task(processor.cleanup())
        self.operations.append(cleanup)
        await asyncio.wait({cleanup}, timeout=1.4)
        self.assertTrue(cleanup.done())
        await cleanup
        owned = self.manager.current_tasks()
        self.assertEqual(len(owned), 1)
        self.assertFalse(owned[0].done())
        self.assertGreaterEqual(processor.requests, 1)

        await processor.queue_frame(StartFrame())
        await processor.queue_frame(TextFrame("queued after cleanup"))
        processor._setup_finished()
        self.assertEqual(self.manager.current_tasks(), owned)
        processor.release.set()
        await asyncio.wait(owned, timeout=1)
        await asyncio.sleep(0)
        self.assertEqual(self.manager.current_tasks(), [])
        self.assertEqual(len(processor.processed), 1)
        self.assertEqual(sink.texts, [])
        self.assertEqual(callbacks, [])
        self.assertEqual(after_events, [])

    async def test_system_task_has_a_bounded_final_wait(self):
        processor = await self._start_resistant(system=True)
        cleanup = asyncio.create_task(processor.cleanup())
        self.operations.append(cleanup)
        await asyncio.wait({cleanup}, timeout=3.4)
        self.assertTrue(cleanup.done())
        await cleanup
        owned = self.manager.current_tasks()
        self.assertEqual(len(owned), 1)
        processor.release.set()
        await asyncio.wait(owned, timeout=1)
        await asyncio.sleep(0)
        self.assertEqual(self.manager.current_tasks(), [])

    async def test_caller_cancellation_propagates_and_cleanup_can_be_retried(self):
        processor = await self._start_resistant()
        cleanup = asyncio.create_task(processor.cleanup())
        self.operations.append(cleanup)
        await asyncio.wait_for(processor.cancelled.wait(), timeout=1)
        cleanup.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(cleanup, timeout=0.5)
        self.assertTrue(self.manager.current_tasks())
        await processor.cleanup()
        owned = self.manager.current_tasks()
        self.assertEqual(len(owned), 1)
        processor.release.set()
        await asyncio.wait(owned, timeout=1)
        await asyncio.sleep(0)
        await processor.cleanup()
        self.assertEqual(self.manager.current_tasks(), [])

    async def test_two_cleanup_callers_retain_the_same_pending_task(self):
        processor = await self._start_resistant()
        operations = [asyncio.create_task(processor.cleanup()) for _ in range(2)]
        self.operations.extend(operations)
        done, pending = await asyncio.wait(operations, timeout=1.4)
        self.assertEqual(pending, set())
        await asyncio.gather(*done)
        self.assertEqual(len(self.manager.current_tasks()), 1)

    async def test_process_task_can_initiate_its_own_cleanup(self):
        finished = asyncio.Event()
        after_events = []
        callbacks = []

        class SelfCleaningProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    await self.cleanup()
                    await self.push_frame(TextFrame("late"))
                    finished.set()

        processor = await self._setup(SelfCleaningProcessor())
        sink = await self._setup(_RecordingProcessor())
        processor.link(sink)

        @processor.event_handler("on_after_process_frame")
        async def after_process(processor, frame):
            if isinstance(frame, TextFrame):
                after_events.append(frame)

        async def callback(*args):
            callbacks.append(args)

        await processor.queue_frame(StartFrame())
        await processor.queue_frame(TextFrame("close"), callback=callback)
        await asyncio.wait_for(finished.wait(), timeout=1)
        await asyncio.sleep(0)
        self.assertEqual(self.manager.current_tasks(), [])
        self.assertEqual(sink.texts, [])
        self.assertEqual(after_events, [])
        self.assertEqual(callbacks, [])

    async def test_cleanup_from_process_hooks_stops_the_next_stage(self):
        for stage in ("before", "body", "callback"):
            with self.subTest(stage=stage):
                body = []
                callbacks = []
                after_events = []

                class CleaningProcessor(FrameProcessor):
                    async def process_frame(self, frame, direction, stage=stage, body=body):
                        body.append(frame)
                        if stage == "body":
                            await self.cleanup()

                processor = await self._setup(CleaningProcessor(enable_direct_mode=True))

                @processor.event_handler("on_before_process_frame")
                async def before(processor, frame, stage=stage):
                    if stage == "before":
                        await processor.cleanup()

                @processor.event_handler("on_after_process_frame")
                async def after(processor, frame, after_events=after_events):
                    after_events.append(frame)

                async def callback(processor, frame, direction, stage=stage, callbacks=callbacks):
                    callbacks.append(frame)
                    if stage == "callback":
                        await processor.cleanup()

                await processor.queue_frame(TextFrame(stage), callback=callback)
                self.assertEqual(len(body), 0 if stage == "before" else 1)
                self.assertEqual(len(callbacks), 1 if stage == "callback" else 0)
                self.assertEqual(after_events, [])

    async def test_input_task_can_initiate_its_own_cleanup(self):
        finished = asyncio.Event()

        class SelfCleaningProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, SystemFrame) and not isinstance(frame, StartFrame):
                    await self.cleanup()
                    finished.set()

        processor = await self._setup(SelfCleaningProcessor())
        await processor.queue_frame(StartFrame())
        await processor.queue_frame(SystemFrame())
        await asyncio.wait_for(finished.wait(), timeout=1)
        await asyncio.sleep(0)
        self.assertEqual(self.manager.current_tasks(), [])

    async def test_final_cleanup_prevents_both_runtime_successors(self):
        class CancelCurrentFrame(SystemFrame):
            pass

        class CancellingProcessor(_ResistantProcessor):
            async def process_frame(self, frame, direction):
                if isinstance(frame, CancelCurrentFrame):
                    await FrameProcessor.process_frame(self, frame, direction)
                    await self._cancel_frame_in_progress(self.processed[0])
                else:
                    await super().process_frame(frame, direction)

        for frame in (InterruptionFrame(), CancelCurrentFrame()):
            with self.subTest(frame=type(frame).__name__):
                self.manager = _RawTaskManager()
                processor = await self._setup(CancellingProcessor())
                await processor.queue_frame(StartFrame())
                await processor.queue_frame(TextFrame("held"))
                await asyncio.wait_for(processor.entered.wait(), timeout=1)
                await processor.queue_frame(frame)
                await asyncio.wait_for(processor.cancelled.wait(), timeout=1)
                cleanup = asyncio.create_task(processor.cleanup())
                self.operations.append(cleanup)
                await asyncio.sleep(0)
                processor.release.set()
                await asyncio.wait_for(cleanup, timeout=1)
                await asyncio.sleep(0)
                self.assertEqual(self.manager.current_tasks(), [])
                self.assertEqual(self.manager.created.count("__process_frame_task_handler"), 1)

    async def test_cleanup_in_before_push_hook_prevents_delivery_and_after_hook(self):
        processor = await self._setup(FrameProcessor(enable_direct_mode=True))
        sink = await self._setup(_RecordingProcessor())
        processor.link(sink)
        after_events = []

        @processor.event_handler("on_before_push_frame")
        async def before(processor, frame):
            await processor.cleanup()

        @processor.event_handler("on_after_push_frame")
        async def after(processor, frame):
            after_events.append(frame)

        await processor.push_frame(TextFrame("discarded"))
        self.assertEqual(sink.texts, [])
        self.assertEqual(after_events, [])

    async def test_cleanup_in_observer_prevents_delivery_and_after_hook(self):
        after_events = []

        class CleaningObserver(BaseObserver):
            async def on_push_frame(self, data):
                await data.source.cleanup()

        observer = CleaningObserver()
        processor = await self._setup(FrameProcessor(enable_direct_mode=True), observer=observer)
        sink = await self._setup(_RecordingProcessor())
        processor.link(sink)

        @processor.event_handler("on_after_push_frame")
        async def after(processor, frame):
            after_events.append(frame)

        await processor.push_frame(TextFrame("discarded"), FrameDirection.DOWNSTREAM)
        self.assertEqual(sink.texts, [])
        self.assertEqual(after_events, [])

    async def test_cleanup_during_neighbor_delivery_stops_the_after_push_event(self):
        processor = await self._setup(FrameProcessor(enable_direct_mode=True))
        after_events = []

        class CleaningSink(_RecordingProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                await processor.cleanup()

        sink = await self._setup(CleaningSink())
        processor.link(sink)

        @processor.event_handler("on_after_push_frame")
        async def after(processor, frame):
            after_events.append(frame)

        await processor.push_frame(TextFrame("already delivered"))
        self.assertEqual(sink.texts, ["already delivered"])
        self.assertEqual(after_events, [])

    async def test_cleanup_owns_a_watcher_while_replacement_is_cancelling_it(self):
        processor = await self._setup(FrameProcessor())
        entered = asyncio.Event()
        cancellation = asyncio.Event()
        release = asyncio.Event()
        replacement_started = asyncio.Event()
        inner_tasks = []

        async def ready():
            inner_tasks.append(asyncio.current_task())
            entered.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellation.set()

        async def replacement_ready():
            replacement_started.set()

        await processor.pause_processing_all_frames_until(ready)
        await asyncio.wait_for(entered.wait(), timeout=1)
        replacing = asyncio.create_task(
            processor.pause_processing_all_frames_until(replacement_ready)
        )
        self.operations.append(replacing)
        try:
            await asyncio.wait_for(cancellation.wait(), timeout=1)
            await processor.cleanup()
            self.assertFalse(replacement_started.is_set())
            self.assertTrue(self.manager.current_tasks())
            self.assertTrue(
                processor._FrameProcessor__pause_watcher_tasks,
                "The retiring watcher must retain a processor owner",
            )
        finally:
            release.set()
            await asyncio.wait({replacing}, timeout=1)
        await replacing
        await asyncio.sleep(0)
        self.assertFalse(replacement_started.is_set())
        self.assertEqual(self.manager.current_tasks(), [])
        self.assertTrue(all(task.done() for task in inner_tasks))

    async def test_late_raw_child_result_is_consumed_once_after_repeated_cleanup(self):
        class FatalCleanup(BaseException):
            pass

        for error in (RuntimeError("late ordinary result"), FatalCleanup("late fatal result")):
            with self.subTest(error=type(error).__name__):
                self.manager = _RawTaskManager(completion_error=error)
                processor = await self._start_resistant()
                loop = asyncio.get_running_loop()
                previous_handler = loop.get_exception_handler()
                contexts = []
                loop.set_exception_handler(
                    lambda loop, context, contexts=contexts: contexts.append(context)
                )
                log = io.StringIO()
                handler_id = logger.add(log, level="ERROR")
                try:
                    await processor.cleanup()
                    await asyncio.gather(processor.cleanup(), processor.cleanup())
                    owned = self.manager.current_tasks()
                    self.assertEqual(len(owned), 1)
                    processor.release.set()
                    await asyncio.wait(owned, timeout=1)
                    await asyncio.sleep(0)
                    self.assertEqual(self.manager.current_tasks(), [])
                    self.assertEqual(log.getvalue().count(str(error)), 1)
                    self.assertEqual(len(contexts), 0 if isinstance(error, Exception) else 1)
                    if contexts:
                        self.assertIs(contexts[0]["exception"], error)
                    await processor.cleanup()
                    self.assertEqual(log.getvalue().count(str(error)), 1)
                finally:
                    logger.remove(handler_id)
                    loop.set_exception_handler(previous_handler)

    async def test_fatal_child_result_keeps_its_propagation_contract(self):
        class FatalCleanup(BaseException):
            pass

        class FailingProcessor(FrameProcessor):
            def __init__(self):
                super().__init__()
                self.entered = asyncio.Event()

            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    self.entered.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        raise FatalCleanup("controlled fatal teardown")

        processor = await self._setup(FailingProcessor())
        await processor.queue_frame(StartFrame())
        await processor.queue_frame(TextFrame("held"))
        await asyncio.wait_for(processor.entered.wait(), timeout=1)
        sink = io.StringIO()
        handler_id = logger.add(sink, level="CRITICAL")
        try:
            with self.assertRaises(FatalCleanup):
                await processor.cleanup()
            self.assertIn("controlled fatal teardown", sink.getvalue())
        finally:
            logger.remove(handler_id)
        self.assertEqual(self.manager.current_tasks(), [])
        self.processors.remove(processor)

    async def test_raw_result_is_observed_after_cleanup_caller_cancellation(self):
        class FatalCleanup(BaseException):
            pass

        for error in (RuntimeError("caller ordinary result"), FatalCleanup("caller fatal result")):
            with self.subTest(error=type(error).__name__):
                self.manager = _RawTaskManager(completion_error=error)
                processor = await self._start_resistant()
                loop = asyncio.get_running_loop()
                previous_handler = loop.get_exception_handler()
                contexts = []
                loop.set_exception_handler(
                    lambda loop, context, contexts=contexts: contexts.append(context)
                )
                log = io.StringIO()
                handler_id = logger.add(log, level="ERROR")
                try:
                    cleanup = asyncio.create_task(processor.cleanup())
                    self.operations.append(cleanup)
                    await asyncio.wait_for(processor.cancelled.wait(), timeout=1)
                    cleanup.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await cleanup
                    owned = self.manager.current_tasks()
                    self.assertEqual(len(owned), 1)
                    processor.release.set()
                    await asyncio.wait(owned, timeout=1)
                    await asyncio.sleep(0)
                    self.assertEqual(self.manager.current_tasks(), [])
                    self.assertEqual(log.getvalue().count(str(error)), 1)
                    self.assertEqual(len(contexts), 0 if isinstance(error, Exception) else 1)
                    await processor.cleanup()
                    self.assertEqual(log.getvalue().count(str(error)), 1)
                finally:
                    logger.remove(handler_id)
                    loop.set_exception_handler(previous_handler)

    async def test_raw_input_and_process_results_are_observed_after_self_cleanup(self):
        class FatalCleanup(BaseException):
            pass

        for frame_type, name in (
            (TextFrame, "__process_frame_task_handler"),
            (SystemFrame, "__input_frame_task_handler"),
        ):
            for error in (RuntimeError("self ordinary result"), FatalCleanup("self fatal result")):
                with self.subTest(task=name, error=type(error).__name__):
                    self.manager = _RawTaskManager(completion_error=error, completion_task=name)
                    finished = asyncio.Event()

                    class SelfCleaningProcessor(FrameProcessor):
                        async def process_frame(
                            self, frame, direction, frame_type=frame_type, finished=finished
                        ):
                            await super().process_frame(frame, direction)
                            if type(frame) is frame_type:
                                await self.cleanup()
                                finished.set()

                    processor = await self._setup(SelfCleaningProcessor())
                    loop = asyncio.get_running_loop()
                    previous_handler = loop.get_exception_handler()
                    contexts = []
                    loop.set_exception_handler(
                        lambda loop, context, contexts=contexts: contexts.append(context)
                    )
                    log = io.StringIO()
                    handler_id = logger.add(log, level="ERROR")
                    try:
                        await processor.queue_frame(StartFrame())
                        frame = TextFrame("close") if frame_type is TextFrame else SystemFrame()
                        await processor.queue_frame(frame)
                        await asyncio.wait_for(finished.wait(), timeout=1)
                        await asyncio.sleep(0)
                        self.assertEqual(self.manager.current_tasks(), [])
                        self.assertEqual(log.getvalue().count(str(error)), 1)
                        self.assertEqual(len(contexts), 0 if isinstance(error, Exception) else 1)
                        await processor.cleanup()
                        self.assertEqual(log.getvalue().count(str(error)), 1)
                    finally:
                        logger.remove(handler_id)
                        loop.set_exception_handler(previous_handler)

    async def test_interrupted_input_phase_keeps_the_unvisited_process_result_owned(self):
        class FatalCleanup(BaseException):
            pass

        class DualProcessor(_ResistantProcessor):
            def __init__(self):
                super().__init__()
                self.input_entered = asyncio.Event()
                self.input_cancelled = asyncio.Event()
                self.input_release = asyncio.Event()

            async def process_frame(self, frame, direction):
                if type(frame) is not SystemFrame:
                    await super().process_frame(frame, direction)
                    return
                self.input_entered.set()
                while not self.input_release.is_set():
                    try:
                        await self.input_release.wait()
                    except asyncio.CancelledError:
                        self.input_cancelled.set()

        for error in (RuntimeError("unvisited ordinary result"), FatalCleanup("unvisited fatal")):
            with self.subTest(error=type(error).__name__):
                self.manager = _RawTaskManager(completion_error=error)
                processor = await self._setup(DualProcessor())
                await processor.queue_frame(StartFrame())
                await processor.queue_frame(TextFrame("held"))
                await asyncio.wait_for(processor.entered.wait(), timeout=1)
                await processor.queue_frame(SystemFrame())
                await asyncio.wait_for(processor.input_entered.wait(), timeout=1)
                contexts = []
                loop = asyncio.get_running_loop()
                previous_handler = loop.get_exception_handler()
                loop.set_exception_handler(
                    lambda loop, context, contexts=contexts: contexts.append(context)
                )
                log = io.StringIO()
                handler_id = logger.add(log, level="ERROR")
                try:
                    cleanup = asyncio.create_task(processor.cleanup())
                    self.operations.append(cleanup)
                    await asyncio.wait_for(processor.input_cancelled.wait(), timeout=1)
                    cleanup.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await cleanup
                    self.assertEqual(processor.requests, 0)
                    owned = self.manager.current_tasks()
                    self.assertEqual(len(owned), 2)
                    processor.input_release.set()
                    processor.release.set()
                    await asyncio.wait(owned, timeout=1)
                    await asyncio.sleep(0)
                    self.assertEqual(self.manager.current_tasks(), [])
                    self.assertEqual(log.getvalue().count(str(error)), 1)
                    self.assertEqual(len(contexts), 0 if isinstance(error, Exception) else 1)
                    await processor.cleanup()
                    self.assertEqual(log.getvalue().count(str(error)), 1)
                finally:
                    processor.input_release.set()
                    processor.release.set()
                    logger.remove(handler_id)
                    loop.set_exception_handler(previous_handler)

    async def test_fatal_done_watcher_keeps_its_pending_sibling_observed(self):
        class FatalCleanup(BaseException):
            pass

        self.manager = _RawTaskManager()
        processor = await self._setup(FrameProcessor())
        entered = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()

        async def fatal():
            entered[0].set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise FatalCleanup("immediate group failure")

        async def pending():
            entered[1].set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            raise RuntimeError("remaining sibling result")

        tasks = [processor.create_task(fatal()), processor.create_task(pending())]
        # This is a focused group-exit control. Public lifecycle and concurrent
        # pause replacement tests separately exercise admission and ownership.
        watchers = processor._FrameProcessor__pause_watcher_tasks
        watchers.update(tasks)
        for task in tasks:
            task.add_done_callback(watchers.discard)
        await asyncio.gather(*(event.wait() for event in entered))
        log = io.StringIO()
        handler_id = logger.add(log, level="ERROR")
        try:
            with self.assertRaises(FatalCleanup):
                await processor.cleanup()
            self.assertFalse(tasks[1].done())
            self.assertIn(tasks[1], self.manager.current_tasks())
            release.set()
            await asyncio.wait(tasks, timeout=1)
            await asyncio.sleep(0)
            self.assertEqual(log.getvalue().count("remaining sibling result"), 1)
            self.assertEqual(self.manager.current_tasks(), [])
        finally:
            release.set()
            logger.remove(handler_id)

    async def test_completed_result_is_not_observed_again_by_a_waiting_retry(self):
        error = RuntimeError("once across a waiting retry")
        self.manager = _RawTaskManager(completion_error=error)
        processor = await self._start_resistant()
        event_entered = asyncio.Event()
        event_release = asyncio.Event()

        @processor.event_handler("on_usable_changed")
        async def event_handler(processor, usable):
            event_entered.set()
            await event_release.wait()

        log = io.StringIO()
        handler_id = logger.add(log, level="ERROR")
        try:
            await processor.cleanup()
            owned = self.manager.current_tasks()
            self.assertEqual(len(owned), 1)
            await processor.set_usable(False)
            await asyncio.wait_for(event_entered.wait(), timeout=1)
            retry = asyncio.create_task(processor.cleanup())
            self.operations.append(retry)
            await asyncio.sleep(0)
            self.assertFalse(retry.done())
            processor.release.set()
            await asyncio.wait(owned, timeout=1)
            await asyncio.sleep(0)
            self.assertEqual(log.getvalue().count(str(error)), 1)
            retry.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await retry
            await asyncio.sleep(0)
            self.assertEqual(log.getvalue().count(str(error)), 1)
            self.assertEqual(self.manager.current_tasks(), [])
        finally:
            event_release.set()
            logger.remove(handler_id)


if __name__ == "__main__":
    unittest.main()
