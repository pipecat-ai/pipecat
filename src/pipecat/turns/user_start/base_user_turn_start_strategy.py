#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base turn start strategy for determining when the user starts speaking."""

import warnings
from dataclasses import dataclass

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.turns.types import ProcessFrameResult
from pipecat.utils.base_object import BaseObject


@dataclass
class UserTurnStartedParams:
    """Parameters emitted when a user turn starts.

    These parameters are passed to the `on_user_turn_started` event and provide
    contextual information about how the user turn should be handled by the user
    aggregator.

    Parameters:
        enable_user_speaking_frames: Whether the user aggregator should emit
            frames indicating user speaking state (e.g., user started speaking)
            during the bot's turn. This is typically enabled by default, but may
            be disabled when another component (such as an STT service) is already
            responsible for generating user speaking frames.

    """

    enable_interruptions: bool
    enable_user_speaking_frames: bool


class BaseUserTurnStartStrategy(BaseObject):
    """Base class for strategies that determine when a user starts speaking.

    Subclasses should implement logic to detect the start of a user's turn.
    This could be based on voice activity, number of words spoken, or other
    heuristics.

    Events triggered by user turn start strategies:

      - `on_push_frame`: Indicates the strategy wants to push a frame.
      - `on_broadcast_frame`: Indicates the strategy wants to broadcast a frame.
      - `on_user_turn_started`: Signals that a user turn has started.
    """

    def __init__(
        self,
        *,
        enable_interruptions: bool = True,
        enable_user_speaking_frames: bool = True,
        **kwargs,
    ):
        """Initialize the base user turn start strategy.

        Args:
            enable_interruptions: If True, the user aggregator will emit an
                interruption frame when the user turn starts.
            enable_user_speaking_frames: If True, the user aggregator will emit
                frames indicating when the user starts speaking, as well as
                interruption frames. This is enabled by default, but you may want
                to disable it if another component (e.g., an STT service) is
                already generating these frames.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._enable_interruptions = enable_interruptions
        self._enable_user_speaking_frames = enable_user_speaking_frames
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
        self._register_event_handler("on_user_turn_started", sync=True)
        self._register_event_handler("on_reset_aggregation", sync=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # reset() is deprecated.
        if cls.reset is not BaseUserTurnStartStrategy.reset:
            warnings.warn(
                f"`{cls.__name__}` overrides `reset`, which is deprecated since 1.6.0 "
                "and will be removed in 2.0.0. Override `handle_user_turn_started` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    async def cleanup(self):
        """Cleanup the strategy."""
        pass

    async def reset(self):
        """Reset the strategy to its initial state.

        .. deprecated:: 1.6.0
            Use :meth:`handle_user_turn_started` instead. Will be removed in
            2.0.0.

        For a start strategy this only ever ran at turn start, so its work *is*
        "on turn start" — which is exactly what :meth:`handle_user_turn_started`
        names. New strategies should override that callback directly.
        """
        pass

    async def handle_user_turn_started(self):
        """Notify the strategy that a user turn has started.

        The controller calls this on every start strategy when a turn begins.
        Override to run, for example, logic to reset state and prepare for the
        next detection.
        """
        # Backward compatibility: a custom strategy may still override the
        # deprecated reset(); invoke it here (the base reset() is a no-op).
        await self.reset()

    async def handle_user_turn_stopped(self):
        """Notify the strategy that the user turn has stopped.

        The controller calls this on every start strategy when a turn ends.
        Override if the strategy needs to act on turn end (likely uncommon).
        """
        pass

    async def process_frame(self, frame: Frame) -> ProcessFrameResult | None:
        """Process an incoming frame.

        Subclasses should override this to implement logic that decides whether
        the user turn has started.

        Args:
            frame: The frame to be processed.

        Returns:
            A ProcessFrameResult indicating the outcome, or None (treated as
            CONTINUE for backward compatibility).
        """
        pass

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Emit on_push_frame to push a frame using the user aggreagtor.

        Args:
            frame: The frame to be pushed.
            direction: What direction the frame should be pushed to.
        """
        await self._call_event_handler("on_push_frame", frame, direction)

    async def broadcast_frame(self, frame_cls: type[Frame], **kwargs):
        """Emit on_broadcast_frame to broadcast a frame using the user aggreagtor.

        Args:
            frame_cls: The class of the frame to be broadcasted.
            **kwargs: Keyword arguments to be passed to the frame's constructor.
        """
        await self._call_event_handler("on_broadcast_frame", frame_cls, **kwargs)

    async def trigger_user_turn_started(self):
        """Trigger the `on_user_turn_started` event."""
        await self._call_event_handler(
            "on_user_turn_started",
            UserTurnStartedParams(
                enable_interruptions=self._enable_interruptions,
                enable_user_speaking_frames=self._enable_user_speaking_frames,
            ),
        )

    async def trigger_reset_aggregation(self):
        """Trigger the `on_reset_aggregation` event."""
        await self._call_event_handler("on_reset_aggregation")
