#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base strategy for scheduling assistant-initiated responses."""

import asyncio
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from pipecat.frames.frames import ResponseFrame
from pipecat.processors.aggregators.llm_context import LLMContextMessage
from pipecat.turns.response.announcement import AnnouncementConfig, CompletedToolResult
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


@dataclass(frozen=True)
class ResponseActivityState:
    """Snapshot of the conversational activity a response strategy schedules against.

    Pushed into the strategy by the hosting ``LLMAssistantAggregator`` on
    every activity transition — the aggregator owns the state (it already
    tracks it for its built-in function-result deferral); strategies only
    ever see these snapshots.

    Parameters:
        bot_speaking: Whether the bot is currently speaking.
        user_speaking: Whether the user is currently speaking.
        response_pending: Whether a reactive response is still owed to the
            user (LLM response streaming, reactive function call in progress,
            or a deferred post-function-result inference).
    """

    bot_speaking: bool = False
    user_speaking: bool = False
    response_pending: bool = False


class BaseResponseStrategy(BaseObject):
    """Base class for strategies that schedule assistant-initiated responses.

    A response strategy owns the queue of pending ``ResponseFrame``s captured
    by the assistant aggregator and decides when to release them. The base
    class provides the queue, the release flow (with a just-before-release
    re-verification), and a single-slot timer for scheduling deferred release
    checks; subclasses implement the release policy in ``should_release()``.

    The queue holds two kinds of items: explicit ``ResponseFrame``s captured
    from the pipeline, and ``CompletedToolResult``s for async tool calls whose
    announcements route through the strategy natively. Policy is agnostic to
    the item kind; at release, ``CompletedToolResult``s are handed to the
    configurable ``announcement`` composer, which decides what is said about
    them — or that nothing is.

    The strategy holds no reference to the aggregator. Following the same
    pattern as ``LLMContextSummarizer``, the aggregator pushes state in
    (``on_activity_changed()``, ``queue_response()``) and subscribes to the
    events the strategy emits to act on its decisions:

    - on_response_deferred: a queued response was held rather than released
      immediately
    - on_response_released: a batch of responses should be delivered now

    All pending items are released together as one batch, in FIFO order.

    Response strategies are not supported with realtime (speech-to-speech)
    services. Those services take conversational content over their own
    session channel and read only tool results out of the context, so a
    released response's context append reaches them as nothing at all and is
    never spoken. The assistant aggregator warns when a strategy is
    configured on a realtime pipeline.
    """

    def __init__(self, *, announcement: AnnouncementConfig | None = None, **kwargs):
        """Initialize the base response strategy.

        Args:
            announcement: How completed async tool results are announced at
                release — style per cardinality, and the instruction text for
                each. Defaults to ``AnnouncementConfig()``: state a single
                result, offer a batch.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._announcement = announcement or AnnouncementConfig()
        self._activity = ResponseActivityState()
        self._pending: list[ResponseFrame | CompletedToolResult] = []
        self._release_check_task: asyncio.Task | None = None

        self._register_event_handler("on_response_deferred")
        self._register_event_handler("on_response_released", sync=True)

    def compose_announcement(
        self, completed: Sequence[CompletedToolResult]
    ) -> list[LLMContextMessage]:
        """Compose the instruction message announcing released tool results.

        Args:
            completed: The completed tool results being released together.

        Returns:
            The message(s) to append to the LLM context, or none when the
            configured style announces nothing.
        """
        return self._announcement.compose(list(completed))

    @property
    def activity(self) -> ResponseActivityState:
        """The most recent activity snapshot pushed by the aggregator."""
        return self._activity

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the strategy with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        await super().setup(task_manager)

    async def cleanup(self):
        """Cancel any scheduled release check. Safe to call more than once."""
        await self._cancel_release_check()
        await super().cleanup()

    def drain(self) -> list[ResponseFrame | CompletedToolResult]:
        """Remove and return all pending responses (used at shutdown)."""
        items = list(self._pending)
        self._pending.clear()
        return items

    async def queue_response(self, item: ResponseFrame | CompletedToolResult):
        """Accept an assistant-initiated response and release it now or later.

        Args:
            item: A captured ``ResponseFrame``, or a ``CompletedToolResult``
                for an async tool completion routed through the strategy.
        """
        self._pending.append(item)
        await self._maybe_release()
        if item in self._pending:
            await self._call_event_handler("on_response_deferred", item)

    async def on_activity_changed(self, activity: ResponseActivityState):
        """Receive a new activity snapshot from the aggregator.

        Called on every conversational activity transition (user or bot
        speaking starts/stops, LLM response starts/ends, interruptions,
        function-call progress). The default stores the snapshot and
        re-evaluates release; subclasses that track activity timing should
        extend this.

        Args:
            activity: The new activity state.
        """
        self._activity = activity
        await self._maybe_release()

    @abstractmethod
    async def should_release(self) -> bool:
        """Whether the pending batch should be released right now.

        Called with at least one pending response, against the latest
        ``activity`` snapshot. Implementations may
        ``await self._schedule_release_check(delay)`` to arrange a future
        re-evaluation before returning False.
        """
        ...

    async def _maybe_release(self):
        """Release the pending batch if the policy allows it.

        ``should_release()`` runs immediately before the batch is emitted, so
        a release scheduled earlier (e.g. by a timer) is re-verified against
        current activity — if the user just started speaking, the batch is
        held again.
        """
        if not self._pending:
            return
        if not await self.should_release():
            return
        items = self.drain()
        if not items:
            return
        await self._call_event_handler("on_response_released", items)

    async def _schedule_release_check(self, delay: float):
        """Schedule a release re-evaluation after ``delay`` seconds.

        Only one check is scheduled at a time; scheduling cancels and
        replaces any prior pending check. The one exception is a running
        check rescheduling itself (``should_release()`` called from within
        ``check()``): it must not cancel itself, and it's already past its
        sleep and about to finish, so only its reference is replaced.
        """
        task = self._release_check_task
        self._release_check_task = None
        if task and not task.done() and task is not asyncio.current_task():
            await self.cancel_task(task)

        async def check():
            await asyncio.sleep(delay)
            await self._maybe_release()

        self._release_check_task = self.create_task(check(), name="release_check")

    async def _cancel_release_check(self):
        task = self._release_check_task
        self._release_check_task = None
        if task and not task.done():
            await self.cancel_task(task)
