#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer reporting the function calls a conversation makes.

A function call is the one thing a bot does rather than says, and the part of
a turn whose duration belongs to the application's own code. This observer
reports a call when it starts, when it goes in progress, and when it settles,
so a call that stops short of any of those reads as one still waiting rather
than one that quietly stopped mattering.
"""

import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection


class FunctionCallEventKind(StrEnum):
    """Where a function call has got to.

    A call starts when the LLM asks for it and is in progress once it is
    running, which are two moments rather than one: calls run one at a time
    unless the service was built to run them in parallel, so a call can wait
    in between, and a call still waiting when the conversation moves on never
    runs at all. A call settles in one of four ways: it returned, its handler
    raised, it ran past its deadline, or it was cancelled — by an
    interruption, or by the LLM asking for it.
    """

    STARTED = "function_call_started"
    IN_PROGRESS = "function_call_in_progress"
    COMPLETED = "function_call_completed"
    FAILED = "function_call_failed"
    TIMED_OUT = "function_call_timed_out"
    CANCELLED = "function_call_cancelled"


class FunctionCallEvent(BaseModel):
    """One moment in the life of a function call.

    The moments that open a call describe it; the moment that settles it
    describes what became of it. Each names the call, so they read as a
    sequence without any of them repeating the others.

    Parameters:
        kind: What happened to the call.
        function_name: The name of the function.
        tool_call_id: The LLM's identifier for this call, unique within a
            conversation.
        timestamp: Unix timestamp of the moment.
        group_id: Identifies the calls the LLM asked for in one response, which
            run together. Set when the call goes in progress.
        blocking: Whether the conversation waited for this call. A call that
            doesn't block is answered later through a developer message, while
            the LLM carries on talking. Set when the call goes in progress.
        arguments: What the LLM passed to the function, when the observer is
            reporting arguments. Set both when the call starts and when it goes
            in progress, since a call can be reported at either moment without
            the other.
        started_at: When the call started, on the moment it goes in progress,
            so the wait between the two reads from one record.
        in_progress_at: When the call went in progress, on the moment that
            settles it, so the time it ran reads from one record.
        result: What the handler returned, when the observer is reporting
            results.
        error: What went wrong, on a call whose handler raised.
    """

    kind: FunctionCallEventKind
    function_name: str
    tool_call_id: str
    timestamp: float
    group_id: str | None = None
    blocking: bool | None = None
    arguments: Any | None = None
    started_at: float | None = None
    in_progress_at: float | None = None
    result: Any | None = None
    error: str | None = None


class FunctionCallObserver(BaseObserver):
    """Reports each function call a conversation makes, from start to outcome.

    A call is reported at each moment it reaches rather than summarized once
    it is over, because the moments can be far apart and a call need not reach
    all of them: one waiting its turn to run is dropped if the conversation
    moves on, and one the conversation doesn't wait for can settle long after
    the turn that asked for it.

    Arguments and results are where a call holds whatever the conversation was
    about, so each is a choice: arguments travel by default, being small and
    the reason a call is worth reading at all, and results do not, being
    whatever a provider decided to return.

    Events:
        on_function_call_event(observer, event): Emitted for each moment, as a
            :class:`FunctionCallEvent`.

    Example::

        observer = FunctionCallObserver()

        @observer.event_handler("on_function_call_event")
        async def on_function_call_event(observer, event):
            logger.info(event.model_dump_json())
    """

    def __init__(
        self,
        *,
        include_arguments: bool = True,
        include_results: bool = False,
        time_source: Callable[[], float] = time.time,
        **kwargs,
    ):
        """Initialize the function call observer.

        Args:
            include_arguments: Whether to report the arguments a call was made
                with.
            include_results: Whether to report what a call returned.
            time_source: Reads the current time in seconds. Supplying one lets
                a test place moments without waiting.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._include_arguments = include_arguments
        self._include_results = include_results
        self._now = time_source
        self._reported: set[int] = set()
        # When each call started and when it went in progress, so the moment
        # that follows either can carry it.
        self._started_at: dict[str, float] = {}
        self._in_progress_at: dict[str, float] = {}

        self._register_event_handler("on_function_call_event")

    async def on_push_frame(self, data: FramePushed):
        """Report the moment a frame represents, the first time it is seen.

        Args:
            data: Frame push event containing the frame and direction.
        """
        frame = data.frame

        # These frames are broadcast, arriving as two frames with two IDs, so
        # an ID alone would not tell them apart. Read the downstream one.
        if frame.broadcast_sibling_id is not None and data.direction != FrameDirection.DOWNSTREAM:
            return
        if frame.id in self._reported:
            return

        events = self._as_events(frame)
        if not events:
            return

        self._reported.add(frame.id)
        for event in events:
            await self._call_event_handler("on_function_call_event", event)

    def _as_events(self, frame: Frame) -> list[FunctionCallEvent]:
        """Build the moments a frame represents.

        One frame reports one moment, except the frame starting the calls an
        LLM response asked for, which reports one for each of them.

        Args:
            frame: The frame being pushed.

        Returns:
            The moments, or an empty list if this frame is not part of a
            function call's life.
        """
        if isinstance(frame, FunctionCallsStartedFrame):
            at = self._now()
            events = []
            for call in frame.function_calls:
                self._started_at[call.tool_call_id] = at
                events.append(
                    FunctionCallEvent(
                        kind=FunctionCallEventKind.STARTED,
                        function_name=call.function_name,
                        tool_call_id=call.tool_call_id,
                        timestamp=at,
                        arguments=call.arguments if self._include_arguments else None,
                    )
                )
            return events

        elif isinstance(frame, FunctionCallInProgressFrame):
            at = self._now()
            self._in_progress_at[frame.tool_call_id] = at
            return [
                FunctionCallEvent(
                    kind=FunctionCallEventKind.IN_PROGRESS,
                    function_name=frame.function_name,
                    tool_call_id=frame.tool_call_id,
                    timestamp=at,
                    group_id=frame.group_id,
                    # A call that survives an interruption is one the
                    # conversation was never waiting on.
                    blocking=frame.cancel_on_interruption,
                    arguments=frame.arguments if self._include_arguments else None,
                    started_at=self._started_at.pop(frame.tool_call_id, None),
                )
            ]

        elif isinstance(frame, FunctionCallResultFrame):
            # A call that doesn't block may report progress before it is done,
            # and only its final result settles it.
            if frame.properties and not frame.properties.is_final:
                return []
            return self._settles(
                FunctionCallEventKind.FAILED if frame.error else FunctionCallEventKind.COMPLETED,
                frame.function_name,
                frame.tool_call_id,
                result=frame.result if self._include_results and not frame.error else None,
                error=frame.error,
            )

        elif isinstance(frame, FunctionCallCancelFrame):
            # Inference is asked for by the deadline that settled the call, and
            # by nothing else that cancels one.
            kind = (
                FunctionCallEventKind.TIMED_OUT
                if frame.run_llm
                else FunctionCallEventKind.CANCELLED
            )
            return self._settles(kind, frame.function_name, frame.tool_call_id)

        return []

    def _settles(
        self,
        kind: FunctionCallEventKind,
        function_name: str,
        tool_call_id: str,
        *,
        result: Any | None = None,
        error: str | None = None,
    ) -> list[FunctionCallEvent]:
        """Record the end of a call, naming when it began running if that is known.

        A call that began before the observer was watching settles without
        that moment rather than borrowing one from another call.
        """
        self._started_at.pop(tool_call_id, None)
        return [
            FunctionCallEvent(
                kind=kind,
                function_name=function_name,
                tool_call_id=tool_call_id,
                timestamp=self._now(),
                in_progress_at=self._in_progress_at.pop(tool_call_id, None),
                result=result,
                error=error,
            )
        ]
