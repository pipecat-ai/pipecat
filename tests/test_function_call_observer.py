#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    TextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.function_call_observer import FunctionCallEventKind, FunctionCallObserver
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.asyncio.task_manager import TaskManager


class TestFunctionCallObserver(unittest.IsolatedAsyncioTestCase):
    """The life of a function call, moment by moment."""

    async def asyncSetUp(self):
        self.clock = 1_000_000.0
        self.observer = FunctionCallObserver(time_source=lambda: self.clock)
        # Event handlers run as tasks, so the observer needs a task manager.
        await self.observer.setup(TaskManager())
        self.events = []
        self._watch(self.observer)

    def _watch(self, observer):
        @observer.event_handler("on_function_call_event")
        async def on_function_call_event(observer, event):
            self.events.append(event)

    def _wait(self, seconds: float):
        """Advance the clock without sleeping."""
        self.clock += seconds

    async def _push(self, frame, observer=None, direction=FrameDirection.DOWNSTREAM):
        """Feed one frame to the observer, as a pipeline push would."""
        await (observer or self.observer).on_push_frame(
            FramePushed(
                source=IdentityFilter(name="source"),
                destination=IdentityFilter(name="destination"),
                frame=frame,
                direction=direction,
                timestamp=0,
            )
        )
        await self._settle()

    async def _settle(self):
        import asyncio

        await asyncio.sleep(0.01)

    def _in_progress(self, **kwargs):
        return FunctionCallInProgressFrame(
            **{
                "function_name": "get_weather",
                "tool_call_id": "call_1",
                "arguments": {"city": "SF"},
                "cancel_on_interruption": True,
                "group_id": "group_1",
                **kwargs,
            }
        )

    def _result(self, **kwargs):
        return FunctionCallResultFrame(
            **{
                "function_name": "get_weather",
                "tool_call_id": "call_1",
                "arguments": {"city": "SF"},
                "result": {"temperature": 12},
                **kwargs,
            }
        )

    def _started(self, *calls):
        return FunctionCallsStartedFrame(
            function_calls=[
                FunctionCallFromLLM(
                    function_name=name,
                    tool_call_id=tool_call_id,
                    arguments={"city": "SF"},
                    context=None,
                )
                for name, tool_call_id in calls
            ]
        )

    async def test_a_call_is_reported_when_its_execution_starts(self):
        await self._push(self._started(("get_weather", "call_1")))

        (event,) = self.events
        self.assertEqual(event.kind, FunctionCallEventKind.STARTED)
        self.assertEqual(event.function_name, "get_weather")
        self.assertEqual(event.tool_call_id, "call_1")
        self.assertEqual(event.arguments, {"city": "SF"})

    async def test_every_call_in_an_llm_response_is_its_own_moment(self):
        await self._push(self._started(("get_weather", "call_1"), ("get_time", "call_2")))

        self.assertEqual(
            [(event.kind, event.tool_call_id) for event in self.events],
            [
                (FunctionCallEventKind.STARTED, "call_1"),
                (FunctionCallEventKind.STARTED, "call_2"),
            ],
        )

    async def test_a_call_that_goes_in_progress_names_when_it_started(self):
        """Calls run one at a time by default, so a call can wait to run."""
        await self._push(self._started(("get_weather", "call_1")))
        self._wait(0.9)
        await self._push(self._in_progress())

        _, in_progress = self.events
        self.assertAlmostEqual(in_progress.timestamp - in_progress.started_at, 0.9, places=6)

    async def test_a_call_that_never_runs_is_left_where_it_stopped(self):
        """A call still waiting when the conversation moves on runs no further."""
        await self._push(self._started(("get_weather", "call_1")))

        (event,) = self.events
        self.assertEqual(event.kind, FunctionCallEventKind.STARTED)

    async def test_a_call_describes_itself_when_it_goes_in_progress(self):
        await self._push(self._in_progress())

        (event,) = self.events
        self.assertEqual(event.kind, FunctionCallEventKind.IN_PROGRESS)
        self.assertEqual(event.function_name, "get_weather")
        self.assertEqual(event.tool_call_id, "call_1")
        self.assertEqual(event.group_id, "group_1")
        self.assertEqual(event.arguments, {"city": "SF"})
        self.assertEqual(event.timestamp, 1_000_000.0)

    async def test_a_call_that_settles_names_when_it_began_running(self):
        """So the time a call ran reads from the record that ends it."""
        await self._push(self._in_progress())
        self._wait(1.4)
        await self._push(self._result())

        _, settled = self.events
        self.assertEqual(settled.kind, FunctionCallEventKind.COMPLETED)
        self.assertAlmostEqual(settled.timestamp - settled.in_progress_at, 1.4, places=6)

    async def test_a_call_the_conversation_waits_on_is_marked_blocking(self):
        await self._push(self._in_progress())
        await self._push(self._in_progress(tool_call_id="call_2", cancel_on_interruption=False))

        blocking, non_blocking = self.events
        self.assertTrue(blocking.blocking)
        self.assertFalse(non_blocking.blocking)

    async def test_a_handler_that_raised_is_reported_as_a_failure(self):
        await self._push(self._in_progress())
        await self._push(self._result(error="RuntimeError: the API is down"))

        _, settled = self.events
        self.assertEqual(settled.kind, FunctionCallEventKind.FAILED)
        self.assertEqual(settled.error, "RuntimeError: the API is down")

    async def test_a_deadline_and_an_interruption_settle_a_call_differently(self):
        """Only a call cancelled by its own deadline asks for inference."""
        await self._push(self._in_progress())
        await self._push(
            FunctionCallCancelFrame(
                function_name="get_weather", tool_call_id="call_1", run_llm=True
            )
        )
        await self._push(self._in_progress(tool_call_id="call_2"))
        await self._push(
            FunctionCallCancelFrame(function_name="get_weather", tool_call_id="call_2")
        )

        self.assertEqual(
            [event.kind for event in self.events[1::2]],
            [FunctionCallEventKind.TIMED_OUT, FunctionCallEventKind.CANCELLED],
        )

    async def test_progress_reported_along_the_way_does_not_settle_a_call(self):
        """A call that doesn't block can report before it is done."""
        await self._push(self._in_progress(cancel_on_interruption=False))
        await self._push(
            self._result(
                result="still looking", properties=FunctionCallResultProperties(is_final=False)
            )
        )
        await self._push(self._result())

        self.assertEqual(
            [event.kind for event in self.events],
            [FunctionCallEventKind.IN_PROGRESS, FunctionCallEventKind.COMPLETED],
        )

    async def test_results_travel_only_when_they_are_asked_for(self):
        """They hold whatever a provider decided to return."""
        await self._push(self._result())

        (default,) = self.events
        self.assertIsNone(default.result)

        reporting_results = FunctionCallObserver(
            include_results=True, time_source=lambda: self.clock
        )
        await reporting_results.setup(TaskManager())
        self._watch(reporting_results)
        await self._push(self._result(), observer=reporting_results)

        self.assertEqual(self.events[-1].result, {"temperature": 12})

    async def test_arguments_can_be_left_out(self):
        not_reporting_arguments = FunctionCallObserver(
            include_arguments=False, time_source=lambda: self.clock
        )
        await not_reporting_arguments.setup(TaskManager())
        self._watch(not_reporting_arguments)

        await self._push(self._in_progress(), observer=not_reporting_arguments)

        (event,) = self.events
        self.assertEqual(event.function_name, "get_weather")
        self.assertIsNone(event.arguments)

    async def test_a_call_that_began_before_the_observer_settles_without_that_moment(self):
        await self._push(self._result())

        (event,) = self.events
        self.assertEqual(event.kind, FunctionCallEventKind.COMPLETED)
        self.assertIsNone(event.in_progress_at)

    async def test_a_broadcast_moment_is_reported_once(self):
        """Broadcast frames arrive twice, with two IDs."""
        frame = self._in_progress()
        sibling = self._in_progress()
        frame.broadcast_sibling_id = sibling.id
        sibling.broadcast_sibling_id = frame.id

        await self._push(frame)
        await self._push(sibling, direction=FrameDirection.UPSTREAM)

        self.assertEqual(len(self.events), 1)

    async def test_a_result_is_reported_once_however_far_it_travels(self):
        """A result is pushed again by every processor it passes through."""
        result = self._result()

        await self._push(result)
        await self._push(result)

        self.assertEqual(len(self.events), 1)

    async def test_frames_from_elsewhere_in_the_pipeline_are_ignored(self):
        await self._push(TextFrame("hello"))

        self.assertEqual(self.events, [])
