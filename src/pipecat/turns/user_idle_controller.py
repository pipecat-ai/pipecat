#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines a controller for detecting a conversation that has gone quiet."""

import asyncio

from pipecat.frames.frames import (
    BackgroundWorkFinishedFrame,
    BackgroundWorkStartedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    UserIdleTimeoutUpdateFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.base_object import BaseObject


class UserIdleController(BaseObject):
    """Controller for detecting a conversation that has gone quiet, and whose silence it is.

    The controller times the silence after the bot finishes speaking and
    reports it as one of two events, told apart by whether work is in flight.
    Work in flight is what a ``BackgroundWorkStartedFrame`` announced and no
    ``BackgroundWorkFinishedFrame`` has ended yet: an asynchronous function
    call, a backend's delegated task.

    - With nothing in flight, the silence is the user's: their turn, left
      unanswered. After ``user_idle_timeout`` seconds, ``on_user_turn_idle``.
    - With work in flight, the silence is the bot's: it holds the turn while
      its work runs. After ``user_waiting_timeout`` seconds, ``on_user_waiting``.

    Either timer starts when BotStoppedSpeakingFrame is received and is
    cancelled when someone starts speaking again (UserStartedSpeakingFrame or
    BotStartedSpeakingFrame). Each fires once per start; the bot speaking
    again, the usual answer to either event, starts it over. A handler that
    stays silent starts it over by hand, with :meth:`wait_for_user` or
    :meth:`wait_for_work`. Work starting while the conversation is already
    quiet starts the waiting timer, and the last work finishing hands over to
    the idle one.

    The timers are suppressed while a user turn is in progress to avoid false
    triggers during interruptions (where BotStoppedSpeakingFrame arrives while
    the user is still speaking). The idle timer is also suppressed while a
    function call is pending, since its result will run the bot again.

    A UserIdleTimeoutUpdateFrame applies immediately: it restarts a running
    idle timer with the new duration and, while waiting for the user, arms
    the timer even if idle detection was previously disabled.

    Event handlers available:

    - on_user_turn_idle: Emitted when the user has been idle for the timeout period.
    - on_user_waiting: Emitted when the user has waited quietly on work in
      flight for the timeout period.

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
            user_waiting_timeout: Timeout in seconds of quiet, with work in
                flight, before the user counts as waiting. 0 disables waiting
                detection.
        """
        super().__init__()

        self._user_idle_timeout = user_idle_timeout
        self._user_waiting_timeout = user_waiting_timeout

        self._waiting_for_user: bool = False
        self._bot_speaking: bool = False
        self._user_turn_in_progress: bool = False
        self._function_calls_in_progress: int = 0
        self._background_work: set[str] = set()
        self._timer_task: asyncio.Task | None = None

        self._register_event_handler("on_user_turn_idle", sync=True)
        self._register_event_handler("on_user_waiting", sync=True)

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
        return self._function_calls_in_progress > 0

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

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            # Interruption case: the frame order is UserStartedSpeaking →
            # BotStoppedSpeaking → (user keeps talking) → UserStoppedSpeaking.
            # Without the user-turn guard the timer would start while the user
            # is still speaking.
            if self._user_turn_in_progress:
                return
            # Function call case: normally FunctionCallsStarted arrives after
            # BotStoppedSpeaking and cancels the timer directly. But a race
            # condition can cause FunctionCallsStarted to arrive before
            # BotStoppedSpeaking when pushing a TTSSpeakFrame in the
            # on_function_calls_started event handler, so the counter guard
            # prevents the idle timer from starting while a function call is
            # in progress.
            if self._function_calls_in_progress == 0:
                # Track the waiting-for-user window even when the timeout is
                # currently <= 0 (no timer), so a later timeout update can arm
                # the timer without waiting for the next bot turn.
                self._waiting_for_user = True
            await self._start_timer()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._waiting_for_user = False
            await self._cancel_timer()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._waiting_for_user = False
            self._user_turn_in_progress = True
            await self._cancel_timer()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_turn_in_progress = False
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._waiting_for_user = False
            self._function_calls_in_progress += len(frame.function_calls)
            await self._cancel_timer()
        elif isinstance(frame, FunctionCallResultFrame):
            # An async tool can report progress with intermediate results
            # while it keeps running; only its final result settles the call.
            is_final = frame.properties.is_final if frame.properties else True
            if is_final:
                self._function_calls_in_progress = max(0, self._function_calls_in_progress - 1)
        elif isinstance(frame, FunctionCallCancelFrame):
            self._function_calls_in_progress = max(0, self._function_calls_in_progress - 1)
        elif isinstance(frame, BackgroundWorkStartedFrame):
            first = not self._background_work
            self._background_work.add(frame.work_id)
            # Work starting in a quiet conversation is waited on from now; an
            # idle timer running at that point was timing the wrong silence.
            if first and self._quiet:
                await self._cancel_timer()
                await self._start_waiting_timer()
        elif isinstance(frame, BackgroundWorkFinishedFrame):
            self._background_work.discard(frame.work_id)
            if not self._background_work and self._quiet:
                # Nothing left to wait on: the silence is the user's again.
                if self._function_calls_in_progress == 0:
                    self._waiting_for_user = True
                await self._start_timer()

    async def wait_for_user(self):
        """Start waiting for the user after a turn the bot will not answer.

        The timer normally starts when the bot stops speaking. A user turn that
        ends with nothing for the bot to answer (e.g. no transcript) cancels
        the timer without the bot speaking again afterwards, so the caller
        re-arms it here. Does nothing while the bot is speaking, a user turn
        is in progress, or function calls are pending.
        """
        if not self._quiet:
            return
        if self._function_calls_in_progress > 0:
            return
        self._waiting_for_user = True
        await self._start_timer()

    async def wait_for_work(self):
        """Start waiting on the work in flight again, after a silent answer to ``on_user_waiting``.

        The waiting timer normally starts over when the bot stops speaking. A
        handler that decided to say nothing re-arms it here, so the next
        interval gets a fresh decision. Does nothing while the bot is
        speaking, a user turn is in progress, or no work is in flight.
        """
        if not self._quiet or not self._background_work:
            return
        await self._start_waiting_timer()

    @property
    def _quiet(self) -> bool:
        """Whether nobody is speaking: no bot speech and no open user turn."""
        return not self._bot_speaking and not self._user_turn_in_progress

    async def _start_timer(self):
        """Start the timer for the kind of silence this is.

        The waiting timer when work is in flight; the idle timer otherwise,
        unless a function call is pending, whose result will run the bot.
        A timer of the other kind that is still running is cancelled either
        way, since it was timing a silence that is over.
        """
        await self._cancel_timer()
        if self._background_work:
            await self._start_waiting_timer()
        elif self._function_calls_in_progress == 0:
            await self._start_idle_timer()

    async def _start_idle_timer(self):
        """Start (or restart) the idle timer."""
        if self._user_idle_timeout <= 0:
            return
        await self._cancel_timer()
        self._timer_task = self.create_task(
            self._timer_expired(self._user_idle_timeout, "on_user_turn_idle")
        )

    async def _start_waiting_timer(self):
        """Start (or restart) the waiting timer."""
        if self._user_waiting_timeout <= 0:
            return
        await self._cancel_timer()
        self._timer_task = self.create_task(
            self._timer_expired(self._user_waiting_timeout, "on_user_waiting")
        )

    async def _cancel_timer(self):
        """Cancel whichever timer is running."""
        if self._timer_task:
            await self.cancel_task(self._timer_task)
            self._timer_task = None

    async def _timer_expired(self, timeout: float, event: str):
        """Sleep for the timeout duration then fire the event."""
        await asyncio.sleep(timeout)
        self._timer_task = None
        await self._call_event_handler(event)
