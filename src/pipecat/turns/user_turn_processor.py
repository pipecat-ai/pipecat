#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame processor for managing the user turn lifecycle."""

from typing import Optional, Type

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_idle_controller import UserIdleController
from pipecat.turns.user_start import BaseUserTurnStartStrategy, UserTurnStartedParams
from pipecat.turns.user_stop import BaseUserTurnStopStrategy, UserTurnStoppedParams
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import UserTurnStrategies


class UserTurnProcessor(FrameProcessor):
    """Frame processor for managing the user turn lifecycle.

    This processor uses a turn controller to determine when a user turn starts
    or stops. The actual frames emitted (e.g., UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame) or interruptions depend on the configured
    strategies.

    Event handlers available:

    - on_user_turn_started: Emitted when a user turn starts.
    - on_user_turn_stopped: Emitted when a user turn stops.
    - on_user_turn_stop_timeout: Emitted if no stop strategy triggers before timeout.
    - on_user_turn_idle: Emitted when the user has been idle for the configured timeout.

    Example::

        @processor.event_handler("on_user_turn_started")
        async def on_user_turn_started(processor, strategy: BaseUserTurnStartStrategy):
            ...

        @processor.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(processor, strategy: BaseUserTurnStopStrategy):
            ...

        @processor.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(processor):
            ...

        @processor.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(processor):
            ...

    """

    def __init__(
        self,
        *,
        user_turn_strategies: Optional[UserTurnStrategies] = None,
        user_turn_stop_timeout: float = 5.0,
        user_idle_timeout: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the user turn processor.

        Args:
            user_turn_strategies: Configured strategies for starting and stopping user turns.
            user_turn_stop_timeout: Timeout in seconds to automatically stop a user turn
                if no activity is detected.
            user_idle_timeout: Optional timeout in seconds for detecting user idle state.
                If set, the processor will emit an `on_user_turn_idle` event when the user
                has been idle (not speaking) for this duration. Set to None to disable
                idle detection.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        self._register_event_handler("on_user_turn_started")
        self._register_event_handler("on_user_turn_stopped")
        self._register_event_handler("on_user_turn_stop_timeout")
        self._register_event_handler("on_user_turn_idle")

        self._user_turn_controller = UserTurnController(
            user_turn_strategies=user_turn_strategies or UserTurnStrategies(),
            user_turn_stop_timeout=user_turn_stop_timeout,
        )
        self._user_turn_controller.add_event_handler("on_push_frame", self._on_push_frame)
        self._user_turn_controller.add_event_handler("on_broadcast_frame", self._on_broadcast_frame)
        self._user_turn_controller.add_event_handler(
            "on_user_turn_started", self._on_user_turn_started
        )
        self._user_turn_controller.add_event_handler(
            "on_user_turn_stopped", self._on_user_turn_stopped
        )
        self._user_turn_controller.add_event_handler(
            "on_user_turn_stop_timeout", self._on_user_turn_stop_timeout
        )

        # Optional user idle controller
        self._user_idle_controller: Optional[UserIdleController] = None
        if user_idle_timeout:
            self._user_idle_controller = UserIdleController(user_idle_timeout=user_idle_timeout)
            self._user_idle_controller.add_event_handler(
                "on_user_turn_idle", self._on_user_turn_idle
            )

    async def cleanup(self):
        """Clean up processor resources."""
        await super().cleanup()
        await self._cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame to detect user turn start or stop.

        The frame is passed to the user turn controlled which is responsible for
        deciding when a user turn starts or stops and emitting the corresponding
        events.

        Args:
            frame: The frame to be processed.
            direction: The direction of the incoming frame.

        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self._start(frame)
        elif isinstance(frame, EndFrame):
            # Push EndFrame before stop(), because stop() waits on the task to
            # finish and the task finishes when EndFrame is processed.
            await self.push_frame(frame, direction)
            await self._stop(frame)
        elif isinstance(frame, CancelFrame):
            await self._cancel(frame)
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

        await self._user_turn_controller.process_frame(frame)

        if self._user_idle_controller:
            await self._user_idle_controller.process_frame(frame)

    async def _start(self, frame: StartFrame):
        await self._user_turn_controller.setup(self.task_manager)

        if self._user_idle_controller:
            await self._user_idle_controller.setup(self.task_manager)

    async def _stop(self, frame: EndFrame):
        await self._cleanup()

    async def _cancel(self, frame: CancelFrame):
        await self._cleanup()

    async def _cleanup(self):
        await self._user_turn_controller.cleanup()

        if self._user_idle_controller:
            await self._user_idle_controller.cleanup()

    async def _on_push_frame(
        self, controller, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        await self.push_frame(frame, direction)

    async def _on_broadcast_frame(self, controller, frame_cls: Type[Frame], **kwargs):
        await self.broadcast_frame(frame_cls, **kwargs)

    async def _on_user_turn_started(
        self,
        controller: UserTurnController,
        strategy: BaseUserTurnStartStrategy,
        params: UserTurnStartedParams,
    ):
        logger.debug(f"{self}: User started speaking (strategy: {strategy})")

        if params.enable_user_speaking_frames:
            await self.broadcast_frame(UserStartedSpeakingFrame)

        if params.enable_interruptions and self._allow_interruptions:
            await self.push_interruption_task_frame_and_wait()

        await self._call_event_handler("on_user_turn_started", strategy)

    async def _on_user_turn_stopped(
        self,
        controller: UserTurnController,
        strategy: BaseUserTurnStopStrategy,
        params: UserTurnStoppedParams,
    ):
        logger.debug(f"{self}: User stopped speaking (strategy: {strategy})")

        if params.enable_user_speaking_frames:
            await self.broadcast_frame(UserStoppedSpeakingFrame)

        await self._call_event_handler("on_user_turn_stopped", strategy)

    async def _on_user_turn_stop_timeout(self, controller):
        await self._call_event_handler("on_user_turn_stop_timeout")

    async def _on_user_turn_idle(self, controller):
        await self._call_event_handler("on_user_turn_idle")
