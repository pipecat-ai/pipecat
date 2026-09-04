#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import ErrorFrame, Frame, TextFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.error_observer import ErrorObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.errors import ErrorCategory


class FailingProcessor(FrameProcessor):
    """A processor that fails the way a service does, on being given work."""

    def __init__(self, exception: Exception, category: ErrorCategory | None = None, **kwargs):
        super().__init__(**kwargs)
        self._exception = exception
        self._category = category

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            await self.push_error(
                "the provider said no", exception=self._exception, category=self._category
            )
        else:
            await self.push_frame(frame, direction)


class TestErrorObserverInAPipeline(unittest.IsolatedAsyncioTestCase):
    """Errors as they are raised, through a running pipeline."""

    async def _errors_from(self, exception: Exception, category: ErrorCategory | None = None):
        observer = ErrorObserver()
        events = []

        @observer.event_handler("on_error")
        async def on_error(observer, event):
            events.append(event)

        failing = FailingProcessor(exception, category, name="stt")
        await run_test(
            Pipeline([failing]),
            frames_to_send=[TextFrame("work"), SleepFrame(sleep=0.1)],
            expected_down_frames=[],
            observers=[observer],
        )
        return events

    async def test_an_error_is_reported_where_it_is_raised(self):
        """Named for the processor that failed, not the ones it travels through."""
        (event,) = await self._errors_from(ConnectionError("no route to host"))

        self.assertEqual(event.processor, "stt")
        self.assertEqual(event.message, "the provider said no")
        self.assertEqual(event.exception_type, "ConnectionError")
        # Worked out from the exception, since the processor named no category.
        self.assertEqual(event.category, ErrorCategory.CONNECTIVITY)

    async def test_a_recoverable_failure_leaves_the_processor_usable(self):
        """An unreachable service may well answer the next request."""
        (event,) = await self._errors_from(ConnectionError("no route to host"))

        self.assertTrue(event.processor_usable)

    async def test_a_permanent_failure_costs_the_processor_its_usability(self):
        """Rejected credentials stay rejected, so the capability is gone."""
        (event,) = await self._errors_from(
            Exception("invalid api key"), ErrorCategory.AUTHENTICATION
        )

        self.assertFalse(event.processor_usable)


class TestErrorObserver(unittest.IsolatedAsyncioTestCase):
    """What the observer makes of the frames it is shown."""

    async def asyncSetUp(self):
        self.clock = 1_000_000.0
        self.observer = ErrorObserver(time_source=lambda: self.clock)
        # Event handlers run as tasks, so the observer needs a task manager.
        await self.observer.setup(TaskManager())
        self.events = []

        @self.observer.event_handler("on_error")
        async def on_error(observer, event):
            self.events.append(event)

    async def _push(self, frame, source=None):
        """Feed one frame to the observer, as a pipeline push would."""
        await self.observer.on_push_frame(
            FramePushed(
                source=source or IdentityFilter(name="source"),
                destination=IdentityFilter(name="destination"),
                frame=frame,
                direction=FrameDirection.UPSTREAM,
                timestamp=0,
            )
        )
        await self._settle()

    async def _settle(self):
        import asyncio

        await asyncio.sleep(0.01)

    async def test_an_error_is_reported_once_however_far_it_travels(self):
        """Every processor it passes through pushes it again."""
        failing = IdentityFilter(name="tts")
        error = ErrorFrame(error="failed", processor=failing, category=ErrorCategory.SERVER)

        await self._push(error, source=failing)
        await self._push(error, source=IdentityFilter(name="passing it along"))
        await self._push(error, source=IdentityFilter(name="and along"))

        (event,) = self.events
        self.assertEqual(event.processor, "tts")

    async def test_each_error_is_its_own_event(self):
        """A processor that fails twice failed twice."""
        failing = IdentityFilter(name="tts")

        await self._push(ErrorFrame(error="first", processor=failing), source=failing)
        await self._push(ErrorFrame(error="second", processor=failing), source=failing)

        self.assertEqual([event.message for event in self.events], ["first", "second"])

    async def test_an_error_reports_when_it_happened(self):
        await self._push(ErrorFrame(error="failed"))

        (event,) = self.events
        self.assertEqual(event.timestamp, 1_000_000.0)

    async def test_an_error_assembled_by_hand_is_attributed_to_its_pusher(self):
        """`push_error` settles both of these; a bare frame carries neither."""
        await self._push(ErrorFrame(error="failed"), source=IdentityFilter(name="llm"))

        (event,) = self.events
        self.assertEqual(event.processor, "llm")
        self.assertEqual(event.category, ErrorCategory.UNKNOWN)
        self.assertIsNone(event.exception_type)

    async def test_frames_that_are_not_errors_are_not_reported(self):
        await self._push(TextFrame("hello"))

        self.assertEqual(self.events, [])
