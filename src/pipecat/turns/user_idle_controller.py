#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines a controller for detecting a conversation that has gone quiet."""

import asyncio
import time

from pipecat.frames.frames import (
    BackgroundWorkFinishedFrame,
    BackgroundWorkStartedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    LLMMarkerResponseFrame,
    UserIdleTimeoutUpdateFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserWaitingTimeoutUpdateFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.base_object import BaseObject


class UserIdleController(BaseObject):
    """Controller for detecting a conversation that has gone quiet: the user idle, or waiting.

    The controller times the silence after the bot finishes speaking and
    reports it as one of two events, told apart by whether background work is
    in progress. Background work is what a ``BackgroundWorkStartedFrame``
    announced and no ``BackgroundWorkFinishedFrame`` has ended yet, e.g. an
    asynchronous function call or a task delegated to an ``LLMWithBackend``
    backend.

    - With no background work, the user is idle: it is their turn, and they
      have said nothing. After ``user_idle_timeout`` seconds,
      ``on_user_turn_idle``.
    - With background work in progress, the user is waiting on it. After
      ``user_waiting_timeout`` seconds, ``on_user_waiting``.

    A synchronous function call is not background work: it is part of the
    bot's turn, whose result the LLM waits for before it can say anything,
    so the idle timer does not run while one is pending.

    Either timer starts when BotStoppedSpeakingFrame is received and is
    cancelled when someone starts speaking again (UserStartedSpeakingFrame or
    BotStartedSpeakingFrame). Each fires once per start; the bot speaking
    again, the usual answer to either event, starts it over, and so does a
    reply the LLM chose to leave unspoken, reported as an
    ``LLMMarkerResponseFrame`` of kind ``"silent"``. A handler that answers
    with nothing starts it over by hand, with :meth:`wait_for_user`.
    Background work starting while the conversation is already quiet starts
    the waiting timer, and the last of it finishing hands over to the idle
    one.

    The timers are suppressed while a user turn is in progress to avoid false
    triggers during interruptions (where BotStoppedSpeakingFrame arrives while
    the user is still speaking).

    A UserIdleTimeoutUpdateFrame applies immediately: it restarts a running
    idle timer with the new duration and, while waiting for the user, arms
    the timer even if idle detection was previously disabled. A
    UserWaitingTimeoutUpdateFrame does the same for the waiting timer.

    Event handlers available:

    - on_user_turn_idle: Emitted when the user has been idle, with no background
      work in progress, for the timeout period.
    - on_user_waiting: Emitted when the user has waited quietly on background
      work for the timeout period.

    Example::

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            # Handle user idle - send reminder, prompt, etc.
            ...

        @controller.event_handler("on_user_waiting")
        async def on_user_waiting(controller):
            # Reassure the user: a phrase, music, a prompt for the LLM.
            ...
    """

    def __init__(
        self,
        *,
        user_idle_timeout: float = 0,
        user_waiting_timeout: float = 0,
    ):
        """Initialize the user idle controller.

        Args:
            user_idle_timeout: Timeout in seconds before considering the user idle.
                0 disables idle detection.
            user_waiting_timeout: Timeout in seconds of quiet, with background
                work in progress, before the user counts as waiting. 0 disables
                waiting detection.
        """
        super().__init__()

        self._user_idle_timeout = user_idle_timeout
        self._user_waiting_timeout = user_waiting_timeout

        self._waiting_for_user: bool = False
        self._bot_speaking: bool = False
        self._user_turn_in_progress: bool = False
        # The function calls seen starting and not yet finished, by tool call id.
        self._function_calls_in_progress: set[str] = set()
        # The pending calls explicitly marked asynchronous by FunctionCallInProgressFrame.
        self._async_function_calls: set[str] = set()
        self._background_work: set[str] = set()
        self._timer_task: asyncio.Task | None = None
        self._waiting_timer_deadline: float | None = None
        # When the conversation last went quiet, while it is.
        self._quiet_since: float | None = None

        self._register_event_handler("on_user_turn_idle", sync=True)
        self._register_event_handler("on_user_waiting", sync=True)

    @property
    def quiet_secs(self) -> float:
        """How long nothing has been said, in seconds; 0 while someone is speaking."""
        return 0.0 if self._quiet_since is None else time.monotonic() - self._quiet_since

    @property
    def waiting_for_user(self) -> bool:
        """Whether the bot has finished responding and is waiting for the user.

        False while the bot is thinking, speaking or running a function call,
        and during a user turn.
        """
        return self._waiting_for_user

    @property
    def function_calls_in_progress(self) -> bool:
        """Whether function calls have started and not finished."""
        return bool(self._function_calls_in_progress)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the controller.

        Args:
            setup: Configuration object containing setup parameters.
        """
        return await super().setup(setup.task_manager)

    async def stop(self):
        """Stop the timer.

        Called at session end so it can't report a silence that only means
        the session is over.
        """
        await self._cancel_timer()

    async def cleanup(self):
        """Cleanup the controller."""
        await super().cleanup()
        await self.stop()

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to track user activity state.

        Args:
            frame: The frame to be processed.
        """
        if isinstance(frame, UserIdleTimeoutUpdateFrame):
            self._user_idle_timeout = frame.timeout
            if self._user_idle_timeout <= 0:
                if not self._background_work:
                    await self._cancel_timer()
            elif self._waiting_for_user and not self._background_work:
                # Apply the new timeout now: restart a running timer with the
                # new duration, or arm one if idle detection was previously
                # disabled.
                await self._start_idle_timer()
            return

        if isinstance(frame, UserWaitingTimeoutUpdateFrame):
            self._user_waiting_timeout = frame.timeout
            if self._user_waiting_timeout <= 0:
                if self._background_work:
                    await self._cancel_timer()
            elif self._quiet and self._background_work:
                if self._function_calls_in_progress <= self._async_function_calls:
                    await self._start_waiting_timer()
                else:
                    self._waiting_timer_deadline = time.monotonic() + self._user_waiting_timeout
                    await self._cancel_timer(preserve_waiting_deadline=True)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            # Interruption case: the frame order is UserStartedSpeaking →
            # BotStoppedSpeaking → (user keeps talking) → UserStoppedSpeaking.
            # Without the user-turn guard the timer would start while the user
            # is still speaking.
            if self._user_turn_in_progress:
                return
            self._quiet_since = time.monotonic()
            await self._start_after_bot_turn()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._waiting_for_user = False
            self._quiet_since = None
            await self._cancel_timer()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._waiting_for_user = False
            self._user_turn_in_progress = True
            self._quiet_since = None
            await self._cancel_timer()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_turn_in_progress = False
            if not self._bot_speaking:
                self._quiet_since = time.monotonic()
        elif isinstance(frame, LLMMarkerResponseFrame) and frame.kind == "silent":
            # A reply the LLM chose to leave unspoken ends its turn like speech
            # would, with the silence going on: the next interval gets a fresh
            # decision.
            if self._quiet:
                await self._start_after_bot_turn()
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._waiting_for_user = False
            self._function_calls_in_progress.update(
                function_call.tool_call_id for function_call in frame.function_calls
            )
            # Wait until each new call is classified before timing a wait.
            await self._cancel_timer(preserve_waiting_deadline=bool(self._background_work))
        elif isinstance(frame, FunctionCallInProgressFrame):
            if frame.tool_call_id not in self._function_calls_in_progress:
                return
            if frame.cancel_on_interruption:
                self._async_function_calls.discard(frame.tool_call_id)
            else:
                self._async_function_calls.add(frame.tool_call_id)
                # The background-work system frame can be handled before
                # this control frame, even when it was sent after it.
                await self._resume_waiting_timer()
        elif isinstance(frame, FunctionCallResultFrame):
            # An async tool can report progress with intermediate results
            # while it keeps running; only its final result settles the call.
            is_final = frame.properties.is_final if frame.properties else True
            if is_final:
                await self._settle_function_call(frame.tool_call_id)
        elif isinstance(frame, FunctionCallCancelFrame):
            await self._settle_function_call(frame.tool_call_id)
        elif isinstance(frame, BackgroundWorkStartedFrame):
            first = not self._background_work
            self._background_work.add(frame.work_id)
            if first and self._quiet:
                await self._cancel_timer()
            await self._resume_waiting_timer()
        elif isinstance(frame, BackgroundWorkFinishedFrame):
            self._background_work.discard(frame.work_id)
            if not self._background_work:
                self._waiting_timer_deadline = None
            if not self._background_work and self._quiet:
                # Nothing left to wait on: the user is idle from here.
                if not self._function_calls_in_progress:
                    self._waiting_for_user = True
                await self._start_timer()

    async def wait_for_user(self):
        """Start waiting for the user after a turn the bot will not answer.

        The timer normally starts when the bot stops speaking. A user turn that
        ends with nothing for the bot to answer (e.g. no transcript) cancels
        the timer without the bot speaking again afterwards, so the caller
        re-arms it here: the waiting timer with background work in progress,
        the idle timer otherwise. Does nothing while the bot is speaking or a
        user turn is in progress, and starts no idle timer while function
        calls are pending.
        """
        if self._quiet:
            await self._start_after_bot_turn()

    async def _start_after_bot_turn(self):
        """Start the timer for the silence after the bot's turn.

        Function call case: normally FunctionCallsStarted arrives after
        BotStoppedSpeaking and cancels the timer directly. But a race
        condition can cause FunctionCallsStarted to arrive before
        BotStoppedSpeaking when pushing a TTSSpeakFrame in the
        on_function_calls_started event handler, so the counter guard
        prevents the idle timer from starting while a function call is in
        progress.
        """
        if not self._function_calls_in_progress:
            # Track the waiting-for-user window even when the timeout is
            # currently <= 0 (no timer), so a later timeout update can arm
            # the timer without waiting for the next bot turn.
            self._waiting_for_user = True
        await self._start_timer()

    async def _settle_function_call(self, tool_call_id: str):
        """Count a function call as finished.

        The call's start cancels an idle timer. A reply that speaks first and
        calls the function after, and then says nothing on its result, leaves
        the conversation quiet with background work still in flight. Resume
        the waiting timer if no pending call needs to hold it.

        A call the controller did not see start settles nothing: a built-in
        tool whose calls are not announced still reports a result, and a bus
        bridge can deliver a result a second time.
        """
        if tool_call_id not in self._function_calls_in_progress:
            return
        self._function_calls_in_progress.remove(tool_call_id)
        self._async_function_calls.discard(tool_call_id)
        await self._resume_waiting_timer()

    async def _resume_waiting_timer(self):
        """If quiet, resume waiting when every pending call is known to be async."""
        if (
            self._timer_task is None
            and self._background_work
            and self._function_calls_in_progress <= self._async_function_calls
            and self._quiet
        ):
            await self._start_waiting_timer(preserve_deadline=True)

    @property
    def _quiet(self) -> bool:
        """Whether nobody is speaking: no bot speech and no open user turn."""
        return not self._bot_speaking and not self._user_turn_in_progress

    async def _start_timer(self):
        """Start the timer for the kind of silence this is.

        The waiting timer runs when background work is in progress and every
        pending call is known to be asynchronous. Otherwise, pending calls hold
        the timer until their result runs the bot; without background work or
        pending calls, the idle timer runs.
        A timer of the other kind that is still running is cancelled either
        way, since it was timing a silence that is over.
        """
        if self._background_work:
            if self._function_calls_in_progress <= self._async_function_calls:
                await self._start_waiting_timer()
            else:
                await self._cancel_timer(preserve_waiting_deadline=True)
        else:
            await self._cancel_timer()
            if not self._function_calls_in_progress:
                await self._start_idle_timer()

    async def _start_idle_timer(self):
        """Start (or restart) the idle timer."""
        if self._user_idle_timeout <= 0:
            return
        await self._cancel_timer()
        self._timer_task = self.create_task(
            self._timer_expired(self._user_idle_timeout, "on_user_turn_idle")
        )

    async def _start_waiting_timer(self, *, preserve_deadline: bool = False):
        """Start the waiting timer, optionally resuming its existing deadline."""
        if self._user_waiting_timeout <= 0:
            return
        deadline = self._waiting_timer_deadline if preserve_deadline else None
        await self._cancel_timer(preserve_waiting_deadline=preserve_deadline)
        if deadline is None:
            deadline = time.monotonic() + self._user_waiting_timeout
        self._waiting_timer_deadline = deadline
        self._timer_task = self.create_task(
            self._timer_expired(max(0, deadline - time.monotonic()), "on_user_waiting")
        )

    async def _cancel_timer(self, *, preserve_waiting_deadline: bool = False):
        """Cancel whichever timer is running."""
        task = self._timer_task
        if task:
            await self.cancel_task(task)
            if self._timer_task is task:
                self._timer_task = None
        if not preserve_waiting_deadline:
            self._waiting_timer_deadline = None

    async def _timer_expired(self, timeout: float, event: str):
        """Sleep for the timeout duration then fire the event."""
        await asyncio.sleep(timeout)
        if self._timer_task is not asyncio.current_task():
            return
        self._timer_task = None
        if event == "on_user_waiting":
            self._waiting_timer_deadline = None
        await self._call_event_handler(event)
