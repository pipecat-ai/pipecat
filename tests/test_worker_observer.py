#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how WorkerObserver hands pushes to the observers it manages."""

import ast
import asyncio
import gc
import inspect
import io
import textwrap
import unittest

from loguru import logger

from pipecat.frames.frames import (
    BotSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.observers.startup_timing_observer import StartupTimingObserver
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import IdleFrameObserver
from pipecat.pipeline.worker_observer import WorkerObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.tracing.turn_trace_observer import TurnTraceObserver


class RecordingObserver(BaseObserver):
    """Records every push of a text frame it is told about."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pushes: list[tuple[str, bool]] = []

    async def on_push_frame(self, data: FramePushed):
        if isinstance(data.frame, TextFrame):
            self.pushes.append((data.source.name, data.first_push))


class ProcessingRecordingObserver(RecordingObserver):
    """Also records every processor a text frame is processed by."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processed: list[str] = []

    async def on_process_frame(self, data: FrameProcessed):
        if isinstance(data.frame, TextFrame):
            self.processed.append(data.processor.name)


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

    async def test_an_observer_handling_processing_is_told_once_per_processor(self):
        observer = ProcessingRecordingObserver()
        await self._run(observer)
        for name in ("first", "second", "third"):
            self.assertEqual(observer.processed.count(name), 1)


def _processed(frame) -> FrameProcessed:
    return FrameProcessed(
        processor=IdentityFilter(name="processor"),
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


class _BaseMethodWrapper:
    """A bound wrapper exposing the base method as ``__func__``, as wrapt's wrappers do."""

    def __init__(self, processed: list[Frame]):
        self.__func__ = BaseObserver.on_process_frame
        self._processed = processed

    async def __call__(self, data: FrameProcessed):
        self._processed.append(data.frame)


class WorkerObserverTestCase(unittest.IsolatedAsyncioTestCase):
    """Drives a WorkerObserver set up outside a pipeline."""

    async def _worker_observer(self, *observers: BaseObserver) -> WorkerObserver:
        worker_observer = WorkerObserver(observers=list(observers))
        await worker_observer.setup(TaskManager())
        self.addAsyncCleanup(worker_observer.cleanup)
        return worker_observer

    async def _drain(self, worker_observer: WorkerObserver):
        await asyncio.gather(*(proxy.queue.join() for proxy in worker_observer._proxies.values()))


class TestWorkerObserverProcessing(WorkerObserverTestCase):
    """A frame being processed only goes to observers that handle it."""

    async def test_an_observer_not_handling_processing_is_not_queued_it(self):
        handles = ProcessingRecordingObserver()
        ignores = RecordingObserver()
        worker_observer = await self._worker_observer(handles, ignores)

        await worker_observer.on_process_frame(_processed(TextFrame("hello")))

        # Nothing has let the proxies run yet, so what was queued is still there.
        self.assertEqual(worker_observer._proxies[handles].queue.qsize(), 1)
        self.assertEqual(worker_observer._proxies[ignores].queue.qsize(), 0)

    async def test_an_on_process_frame_set_on_the_instance_is_told(self):
        observer = RecordingObserver()
        processed = []

        async def on_process_frame(data: FrameProcessed):
            processed.append(data.frame)

        observer.on_process_frame = on_process_frame
        worker_observer = await self._worker_observer(observer)
        frame = TextFrame("hello")

        await worker_observer.on_process_frame(_processed(frame))
        await self._drain(worker_observer)

        self.assertEqual(processed, [frame])

    async def test_an_on_process_frame_set_on_the_instance_after_setup_is_told(self):
        observer = RecordingObserver()
        worker_observer = await self._worker_observer(observer)
        processed = []

        async def on_process_frame(data: FrameProcessed):
            processed.append(data.frame)

        observer.on_process_frame = on_process_frame
        frame = TextFrame("hello")

        await worker_observer.on_process_frame(_processed(frame))
        await self._drain(worker_observer)

        self.assertEqual(processed, [frame])

    async def test_an_on_process_frame_set_in_the_observers_own_setup_is_told(self):
        processed = []

        class SettingUpObserver(BaseObserver):
            async def setup(self, task_manager):
                await super().setup(task_manager)

                async def on_process_frame(data: FrameProcessed):
                    processed.append(data.frame)

                self.on_process_frame = on_process_frame

        worker_observer = await self._worker_observer(SettingUpObserver())
        frame = TextFrame("hello")

        await worker_observer.on_process_frame(_processed(frame))
        await self._drain(worker_observer)

        self.assertEqual(processed, [frame])

    async def test_an_on_process_frame_wrapping_the_base_one_is_told(self):
        processed = []

        class WrappingObserver(BaseObserver):
            @property
            def on_process_frame(self):
                return _BaseMethodWrapper(processed)

        worker_observer = await self._worker_observer(WrappingObserver())
        frame = TextFrame("hello")

        await worker_observer.on_process_frame(_processed(frame))
        await self._drain(worker_observer)

        self.assertEqual(processed, [frame])

    async def test_lifecycle_events_reach_an_observer_not_handling_processing(self):
        started = asyncio.Event()

        class StartedObserver(BaseObserver):
            async def on_pipeline_started(self):
                started.set()

        worker_observer = await self._worker_observer(StartedObserver())

        await worker_observer.on_pipeline_started()

        await asyncio.wait_for(started.wait(), timeout=1.0)


class TestWorkerObserverMemory(unittest.IsolatedAsyncioTestCase):
    async def test_first_observer_can_be_added_after_setup(self):
        worker_observer = WorkerObserver()
        await worker_observer.setup(TaskManager())
        self.addAsyncCleanup(worker_observer.cleanup)
        observer = RecordingObserver()
        worker_observer.add_observer(observer)
        source = IdentityFilter(name="source")

        await worker_observer.on_push_frame(
            FramePushed(
                source=source,
                destination=source,
                frame=TextFrame("hello"),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        await asyncio.gather(*(proxy.queue.join() for proxy in worker_observer._proxies.values()))
        self.assertEqual(observer.pushes, [("source", True)])

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


class CleanupRecordingObserver(BaseObserver):
    """Records its cleanup and fires an event for every text frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cleaned_up = False
        self._register_event_handler("on_text")

    async def on_push_frame(self, data: FramePushed):
        if isinstance(data.frame, TextFrame):
            await self._call_event_handler("on_text")

    async def cleanup(self):
        await super().cleanup()
        self.cleaned_up = True


class TestWorkerObserverRemoval(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.observer = CleanupRecordingObserver()
        self.worker_observer = WorkerObserver(observers=[self.observer])
        await self.worker_observer.setup(TaskManager())
        self.addAsyncCleanup(self.worker_observer.cleanup)

    async def _push_text(self):
        source = IdentityFilter()
        await self.worker_observer.on_push_frame(
            FramePushed(
                source=source,
                destination=source,
                frame=TextFrame("hello"),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )

    async def test_a_removed_observer_is_cleaned_up(self):
        await self.worker_observer.remove_observer(self.observer)
        self.assertTrue(self.observer.cleaned_up)

    async def test_an_observer_can_be_removed_from_its_own_event_handler(self):
        removed = asyncio.Event()

        @self.observer.event_handler("on_text")
        async def on_text(observer):
            await self.worker_observer.remove_observer(observer)
            removed.set()

        await self._push_text()
        await asyncio.wait_for(removed.wait(), timeout=1.0)
        self.assertTrue(self.observer.cleaned_up)


class FailingCleanupObserver(BaseObserver):
    async def cleanup(self):
        await super().cleanup()
        raise RuntimeError("cleanup failed")


class TestWorkerObserverShutdown(unittest.IsolatedAsyncioTestCase):
    async def test_an_observer_failing_cleanup_does_not_stop_the_others(self):
        later = CleanupRecordingObserver()
        worker_observer = WorkerObserver(observers=[FailingCleanupObserver(), later])
        await worker_observer.setup(TaskManager())

        await worker_observer.cleanup()

        self.assertTrue(later.cleaned_up)


class RaisingObserver(BaseObserver):
    def __init__(self):
        super().__init__(observe_every_push=False)
        self.texts: list[str] = []

    async def on_push_frame(self, data: FramePushed):
        if isinstance(data.frame, TextFrame):
            self.texts.append(data.frame.text)
            if len(self.texts) == 1:
                raise RuntimeError("observer failed")


class TestWorkerObserverErrors(unittest.IsolatedAsyncioTestCase):
    async def test_an_observer_that_raises_keeps_receiving_frames(self):
        observer = RaisingObserver()
        await run_test(
            IdentityFilter(),
            frames_to_send=[TextFrame("one"), TextFrame("two"), SleepFrame(sleep=0.1)],
            expected_down_frames=[TextFrame, TextFrame],
            observers=[observer],
        )
        self.assertEqual(observer.texts, ["one", "two"])


class FrameTypeRecordingObserver(BaseObserver):
    """Records the class of every frame it is told about, pushed or processed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pushes: list[tuple[type[Frame], bool]] = []
        self.processed: list[type[Frame]] = []

    async def on_push_frame(self, data: FramePushed):
        self.pushes.append((type(data.frame), data.first_push))

    async def on_process_frame(self, data: FrameProcessed):
        self.processed.append(type(data.frame))


class AudioCheckFailingObserver(FrameTypeRecordingObserver):
    """Raises when asked whether it observes audio frames."""

    def observes_frame_type(self, frame_type: type[Frame]) -> bool:
        if issubclass(frame_type, InputAudioRawFrame):
            raise RuntimeError("frame type check failed")
        return super().observes_frame_type(frame_type)


def _audio() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)


def _pushed(frame: Frame) -> FramePushed:
    source = IdentityFilter(name="source")
    return FramePushed(
        source=source,
        destination=source,
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


class TestWorkerObserverFrameTypes(WorkerObserverTestCase):
    """An observer is only told about the frame classes it observes."""

    async def _run(self, *observers: BaseObserver):
        pipeline = Pipeline([IdentityFilter(name="first"), IdentityFilter(name="second")])
        await run_test(
            pipeline,
            frames_to_send=[
                TextFrame("hello"),
                _audio(),
                TranscriptionFrame("hi", user_id="user", timestamp="now"),
                SleepFrame(sleep=0.1),
            ],
            observers=list(observers),
        )

    async def test_an_observer_is_told_only_about_the_classes_it_reads(self):
        observer = FrameTypeRecordingObserver(frame_types=(TextFrame,))
        await self._run(observer)
        self.assertEqual({t for t, _ in observer.pushes}, {TextFrame, TranscriptionFrame})
        self.assertEqual(set(observer.processed), {TextFrame, TranscriptionFrame})

    async def test_an_observer_of_some_classes_handling_a_frame_once_is_told_once(self):
        observer = FrameTypeRecordingObserver(frame_types=(TextFrame,), observe_every_push=False)
        await self._run(observer)
        self.assertEqual(observer.pushes, [(TextFrame, True), (TranscriptionFrame, True)])

    async def test_an_observer_of_every_class_still_gets_audio(self):
        observer = FrameTypeRecordingObserver()
        await self._run(observer)
        self.assertIn(InputAudioRawFrame, {t for t, _ in observer.pushes})
        self.assertIn(InputAudioRawFrame, observer.processed)

    async def test_an_observer_added_later_is_told_about_a_class_already_seen(self):
        worker_observer = await self._worker_observer(
            FrameTypeRecordingObserver(frame_types=(BotSpeakingFrame,))
        )
        await worker_observer.on_push_frame(_pushed(TextFrame("before")))
        await worker_observer.on_process_frame(_processed(TextFrame("before")))
        observer = FrameTypeRecordingObserver()
        worker_observer.add_observer(observer)

        await worker_observer.on_push_frame(_pushed(TextFrame("after")))
        await worker_observer.on_process_frame(_processed(TextFrame("after")))
        await self._drain(worker_observer)

        self.assertEqual(observer.pushes, [(TextFrame, True)])
        self.assertEqual(observer.processed, [TextFrame])

    async def test_a_removed_observer_is_not_queued_a_class_already_seen(self):
        observer = FrameTypeRecordingObserver()
        worker_observer = await self._worker_observer(FrameTypeRecordingObserver(), observer)
        await worker_observer.on_push_frame(_pushed(TextFrame("before")))
        await worker_observer.on_process_frame(_processed(TextFrame("before")))
        await self._drain(worker_observer)
        proxy = worker_observer._proxies[observer]
        await worker_observer.remove_observer(observer)

        await worker_observer.on_push_frame(_pushed(TextFrame("after")))
        await worker_observer.on_process_frame(_processed(TextFrame("after")))

        self.assertEqual(proxy.queue.qsize(), 0)

    async def test_an_observer_set_up_again_is_told_on_its_new_proxy(self):
        observer = FrameTypeRecordingObserver()
        worker_observer = WorkerObserver(observers=[observer])
        await worker_observer.setup(TaskManager())
        await worker_observer.on_push_frame(_pushed(TextFrame("first")))
        await self._drain(worker_observer)
        await worker_observer.cleanup()
        await worker_observer.setup(TaskManager())
        self.addAsyncCleanup(worker_observer.cleanup)

        await worker_observer.on_push_frame(_pushed(TextFrame("second")))
        await self._drain(worker_observer)

        self.assertEqual(observer.pushes, [(TextFrame, True), (TextFrame, True)])

    async def test_a_push_no_observer_reads_still_counts_as_the_frames_first(self):
        worker_observer = await self._worker_observer(
            FrameTypeRecordingObserver(frame_types=(TextFrame,))
        )
        frame = _audio()
        await worker_observer.on_push_frame(_pushed(frame))
        observer = FrameTypeRecordingObserver()
        worker_observer.add_observer(observer)

        await worker_observer.on_push_frame(_pushed(frame))
        await self._drain(worker_observer)

        self.assertEqual(observer.pushes, [(InputAudioRawFrame, False)])

    async def test_an_observer_whose_frame_type_check_raises_still_gets_those_frames(self):
        failing = AudioCheckFailingObserver()
        other = FrameTypeRecordingObserver(frame_types=(TextFrame,))
        await self._run(failing, other)
        self.assertIn(InputAudioRawFrame, {t for t, _ in failing.pushes})
        self.assertIn(InputAudioRawFrame, failing.processed)
        self.assertEqual({t for t, _ in other.pushes}, {TextFrame, TranscriptionFrame})

    async def test_a_frame_type_check_that_raises_is_logged_once_per_class(self):
        observer = AudioCheckFailingObserver()
        worker_observer = await self._worker_observer(observer)
        sink = io.StringIO()
        handler_id = logger.add(sink, level="ERROR", format="{message}")
        try:
            for _ in range(3):
                await worker_observer.on_push_frame(_pushed(_audio()))
            await self._drain(worker_observer)
        finally:
            logger.remove(handler_id)

        self.assertEqual(observer.pushes, [(InputAudioRawFrame, True)] * 3)
        errors = sink.getvalue().splitlines()
        self.assertEqual(len(errors), 1)
        self.assertIn("frame type check failed", errors[0])


class TestObserverFrameTypes(unittest.TestCase):
    """The frame classes an observer is given are checked when it is created."""

    def test_frame_types_given_as_a_list_are_observed(self):
        observer = FrameTypeRecordingObserver(frame_types=[TextFrame])
        self.assertTrue(observer.observes_frame_type(TranscriptionFrame))
        self.assertFalse(observer.observes_frame_type(InputAudioRawFrame))

    def test_frame_types_that_are_not_frame_classes_are_rejected(self):
        for frame_types in ((str,), (TextFrame("hello"),), TextFrame):
            with self.subTest(frame_types=frame_types):
                with self.assertRaises(TypeError):
                    FrameTypeRecordingObserver(frame_types=frame_types)


def _frame_classes_read(observer: BaseObserver) -> set[type[Frame]]:
    """The frame classes an observer's handlers test a frame against.

    Read from every ``isinstance(frame, ...)`` and ``isinstance(data.frame,
    ...)`` in the ``on_push_frame`` and ``on_process_frame`` of each class the
    observer inherits from.
    """
    classes: set[type[Frame]] = set()

    def add(node: ast.expr, scope: dict):
        if isinstance(node, ast.Tuple):
            for element in node.elts:
                add(element, scope)
            return
        if isinstance(node, ast.Name):
            value = scope[node.id]
        elif isinstance(node, ast.Attribute) and ast.unparse(node.value) == "self":
            value = getattr(observer, node.attr)
        else:
            raise AssertionError(f"Can't tell which classes {ast.unparse(node)} holds")
        classes.update(value if isinstance(value, tuple) else (value,))

    for cls in type(observer).__mro__:
        if cls in (BaseObserver, object):
            continue
        for name in ("on_push_frame", "on_process_frame"):
            method = cls.__dict__.get(name)
            if method is None:
                continue
            for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(method)))):
                if (
                    isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "isinstance"
                    and ast.unparse(node.args[0]) in ("frame", "data.frame")
                ):
                    add(node.args[1], method.__globals__)
    return classes


class TestBuiltInObserverFrameTypes(unittest.TestCase):
    """The built-in observers observe every frame class they read, and not caller audio."""

    def _observers(self) -> list[BaseObserver]:
        turn_tracker = TurnTrackingObserver()
        latency_tracker = UserBotLatencyObserver()
        return [
            IdleFrameObserver(
                idle_event=asyncio.Event(),
                idle_timeout_frames=(BotSpeakingFrame, TranscriptionFrame),
            ),
            turn_tracker,
            latency_tracker,
            TurnTraceObserver(turn_tracker, latency_tracker=latency_tracker),
            StartupTimingObserver(),
        ]

    def _public_built_ins(self) -> list[tuple[type[BaseObserver], tuple]]:
        return [
            (TurnTrackingObserver, ()),
            (UserBotLatencyObserver, ()),
            (StartupTimingObserver, ()),
            (TurnTraceObserver, (TurnTrackingObserver(), UserBotLatencyObserver())),
        ]

    def test_every_class_read_is_observed(self):
        for observer in self._observers():
            with self.subTest(observer=type(observer).__name__):
                read = _frame_classes_read(observer)
                self.assertTrue(read)
                for frame_type in read:
                    self.assertTrue(observer.observes_frame_type(frame_type), frame_type)

    def test_caller_audio_is_not_observed(self):
        for observer in self._observers():
            with self.subTest(observer=type(observer).__name__):
                self.assertFalse(observer.observes_frame_type(InputAudioRawFrame))

    def test_a_subclass_of_a_built_in_observes_every_class(self):
        for built_in, args in self._public_built_ins():
            with self.subTest(observer=built_in.__name__):
                subclass = type(f"My{built_in.__name__}", (built_in,), {})
                self.assertTrue(subclass(*args).observes_frame_type(InputAudioRawFrame))

    def test_frame_types_given_to_a_built_in_or_its_subclass_replace_its_own(self):
        for built_in, args in self._public_built_ins():
            for cls in (built_in, type(f"My{built_in.__name__}", (built_in,), {})):
                with self.subTest(observer=cls.__name__):
                    observer = cls(*args, frame_types=(InputAudioRawFrame,))
                    self.assertTrue(observer.observes_frame_type(InputAudioRawFrame))
                    self.assertFalse(observer.observes_frame_type(TextFrame))


if __name__ == "__main__":
    unittest.main()
