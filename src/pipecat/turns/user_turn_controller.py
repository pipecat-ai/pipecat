#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines a controller for managing user turn lifecycle."""

import asyncio
from typing import Optional, Type

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.turns.user_start import BaseUserTurnStartStrategy, UserTurnStartedParams
from pipecat.turns.user_stop import BaseUserTurnStopStrategy, UserTurnStoppedParams
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class UserTurnController(BaseObject):
    """Controller for managing user turn lifecycle.

    This class manages user turn state (active/inactive), handles start and stop
    strategies, and emits events when user turns begin, end, or timeout occurs.

    Event handlers available:

    - on_user_turn_started: Emitted when a user turn starts.
    - on_user_turn_stopped: Emitted when a user turn stops.
    - on_user_turn_stop_timeout: Emitted if no stop strategy triggers before timeout.
    - on_push_frame: Emitted when a strategy wants to push a frame.
    - on_broadcast_frame: Emitted when a strategy wants to broadcast a frame.

    Example::

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy: BaseUserTurnStartStrategy, params: UserTurnStartedParams):
            ...

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy: BaseUserTurnStopStrategy, params: UserTurnStoppedParams):
            ...

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            ...

        @controller.event_handler("on_push_frame")
        async def on_push_frame(controller, frame: Frame, direction: FrameDirection):
            ...

        @controller.event_handler("on_broadcast_frame")
        async def on_broadcast_frame(controller, frame_cls: Type[Frame], **kwargs):
            ...
    """

    def __init__(
        self,
        *,
        user_turn_strategies: UserTurnStrategies,
        user_turn_stop_timeout: float = 5.0,
    ):
        """Initialize the user turn controller.

        Args:
            user_turn_strategies: Configured strategies for starting and stopping user turns.
            user_turn_stop_timeout: Timeout in seconds to automatically stop a user turn
                if no activity is detected.
        """
        super().__init__()

        self._user_turn_strategies = user_turn_strategies
        self._user_turn_stop_timeout = user_turn_stop_timeout

        self._task_manager: Optional[BaseTaskManager] = None

        self._user_speaking = False

        self._user_turn = False
        self._user_turn_stop_timeout_event = asyncio.Event()
        self._user_turn_stop_timeout_task: Optional[asyncio.Task] = None

        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
        self._register_event_handler("on_user_turn_started", sync=True)
        self._register_event_handler("on_user_turn_stopped", sync=True)
        self._register_event_handler("on_user_turn_stop_timeout", sync=True)

    @property
    def task_manager(self) -> BaseTaskManager:
        """Returns the configured task manager."""
        if not self._task_manager:
            raise RuntimeError(f"{self} user turn controller was not properly setup")
        return self._task_manager

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the controller with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        self._task_manager = task_manager

        if not self._user_turn_stop_timeout_task:
            self._user_turn_stop_timeout_task = self.task_manager.create_task(
                self._user_turn_stop_timeout_task_handler(),
                f"{self}::_user_turn_stop_timeout_task_handler",
            )

        await self._setup_strategies()

    async def cleanup(self):
        """Cleanup the controller."""
        await super().cleanup()

        if self._user_turn_stop_timeout_task:
            await self.task_manager.cancel_task(self._user_turn_stop_timeout_task)
            self._user_turn_stop_timeout_task = None

        await self._cleanup_strategies()

    async def update_strategies(self, strategies: UserTurnStrategies):
        """Replace the current strategies with the given ones.

        Args:
            strategies: The new user turn strategies the controller should use.
        """
        await self._cleanup_strategies()
        self._user_turn_strategies = strategies
        await self._setup_strategies()

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to detect user turn start or stop.

        The frame is passed to the configured user turn strategies, which are
        responsible for deciding when a user turn starts or stops and emitting
        the corresponding events.

        Args:
            frame: The frame to be processed.

        """
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            await self._handle_transcription(frame)

        for strategy in self._user_turn_strategies.start or []:
            await strategy.process_frame(frame)

        for strategy in self._user_turn_strategies.stop or []:
            await strategy.process_frame(frame)

    async def _setup_strategies(self):
        for s in self._user_turn_strategies.start or []:
            await s.setup(self.task_manager)
            s.add_event_handler("on_push_frame", self._on_push_frame)
            s.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
            s.add_event_handler("on_user_turn_started", self._on_user_turn_started)

        for s in self._user_turn_strategies.stop or []:
            await s.setup(self.task_manager)
            s.add_event_handler("on_push_frame", self._on_push_frame)
            s.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
            s.add_event_handler("on_user_turn_stopped", self._on_user_turn_stopped)

    async def _cleanup_strategies(self):
        for s in self._user_turn_strategies.start or []:
            await s.cleanup()

        for s in self._user_turn_strategies.stop or []:
            await s.cleanup()

    async def _handle_user_started_speaking(self, frame: UserStartedSpeakingFrame):
        self._user_speaking = True

        # The user started talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_user_stopped_speaking(self, frame: UserStoppedSpeakingFrame):
        self._user_speaking = False

        # The user stopped talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        self._user_speaking = True

        # The user started talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._user_speaking = False

        # The user stopped talking, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _handle_transcription(self, frame: TranscriptionFrame | InterimTranscriptionFrame):
        # We have received a transcription, let's reset the user turn timeout.
        self._user_turn_stop_timeout_event.set()

    async def _on_push_frame(
        self,
        strategy: BaseUserTurnStartStrategy | BaseUserTurnStopStrategy,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        await self._call_event_handler("on_push_frame", frame, direction)

    async def _on_broadcast_frame(
        self,
        strategy: BaseUserTurnStartStrategy | BaseUserTurnStopStrategy,
        frame_cls: Type[Frame],
        **kwargs,
    ):
        await self._call_event_handler("on_broadcast_frame", frame_cls, **kwargs)

    async def _on_user_turn_started(
        self,
        strategy: BaseUserTurnStartStrategy,
        params: UserTurnStartedParams,
    ):
        await self._trigger_user_turn_start(strategy, params)

    async def _on_user_turn_stopped(
        self, strategy: BaseUserTurnStopStrategy, params: UserTurnStoppedParams
    ):
        await self._trigger_user_turn_stop(strategy, params)

    async def _trigger_user_turn_start(
        self, strategy: Optional[BaseUserTurnStartStrategy], params: UserTurnStartedParams
    ):
        # Prevent two consecutive user turn starts.
        if self._user_turn:
            return

        self._user_turn = True
        self._user_turn_stop_timeout_event.set()

        # Reset all user turn start strategies to start fresh.
        for s in self._user_turn_strategies.start or []:
            await s.reset()

        await self._call_event_handler("on_user_turn_started", strategy, params)

    async def _trigger_user_turn_stop(
        self, strategy: Optional[BaseUserTurnStopStrategy], params: UserTurnStoppedParams
    ):
        # Prevent two consecutive user turn stops.
        if not self._user_turn:
            return

        self._user_turn = False
        self._user_turn_stop_timeout_event.set()

        # Reset all user turn stop strategies to start fresh.
        for s in self._user_turn_strategies.stop or []:
            await s.reset()

        await self._call_event_handler("on_user_turn_stopped", strategy, params)

    async def _user_turn_stop_timeout_task_handler(self):
        while True:
            try:
                await asyncio.wait_for(
                    self._user_turn_stop_timeout_event.wait(),
                    timeout=self._user_turn_stop_timeout,
                )
                self._user_turn_stop_timeout_event.clear()
            except asyncio.TimeoutError:
                if self._user_turn and not self._user_speaking:
                    await self._call_event_handler("on_user_turn_stop_timeout")
                    await self._trigger_user_turn_stop(
                        None, UserTurnStoppedParams(enable_user_speaking_frames=True)
                    )
