#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base turn start strategy for determining when the user starts speaking."""

from contextvars import ContextVar
from dataclasses import dataclass

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.turns.types import ProcessFrameResult
from pipecat.utils.base_object import BaseObject
from pipecat.utils.deprecation import deprecated, warn_deprecated

# The start strategy whose override of the deprecated
# trigger_user_turn_started() is running.
_running_override: ContextVar[object | None] = ContextVar("_running_start_override", default=None)


@dataclass
class UserTurnStartedParams:
    """Parameters emitted when a user turn starts.

    These parameters are passed to the `on_user_turn_started` event and provide
    contextual information about how the user turn should be handled by the user
    aggregator.

    Parameters:
        enable_user_speaking_frames: Whether to emit
            :class:`~pipecat.frames.frames.UserStartedSpeakingFrame` for this
            turn. False when the turn was already announced elsewhere — by a
            shared :class:`~pipecat.turns.user_turn_processor.UserTurnProcessor`,
            or by a service that emits turn frames rather than proposing them.

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
      - `on_user_turn_started`: Reports the strategy's decision that the
        user started, from :meth:`report_user_turn_started`. The turn
        controller starts the turn on the first decision, or once every
        start strategy has decided in a ``StrategyMode.ALL`` list.
      - `on_reset_aggregation`: Asks the user aggregator to drop the text
        aggregated so far, from :meth:`trigger_reset_aggregation`.
    """

    # Whether a subclass overrides the deprecated trigger_user_turn_started().
    _overrides_trigger_user_turn_started = False

    def __init__(
        self,
        *,
        enable_interruptions: bool = True,
        enable_user_speaking_frames: bool | None = None,
        **kwargs,
    ):
        """Initialize the base user turn start strategy.

        Args:
            enable_interruptions: If True, the user aggregator will emit an
                interruption frame when the user turn starts.
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStartedSpeakingFrame` when a
                turn starts, unless :meth:`report_user_turn_started` overrides
                it for a given turn.

                .. deprecated:: 1.8.0
                    Use :meth:`report_user_turn_started`'s
                    ``enable_user_speaking_frames`` argument instead. Will be
                    removed in 2.0.0.

            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._enable_interruptions = enable_interruptions
        if enable_user_speaking_frames is not None:
            warn_deprecated(
                "`BaseUserTurnStartStrategy.enable_user_speaking_frames` is deprecated "
                "since 1.8.0 and will be removed in 2.0.0. Use "
                "`report_user_turn_started(enable_user_speaking_frames=...)` instead.",
                stacklevel=2,
            )
        self._enable_user_speaking_frames = (
            True if enable_user_speaking_frames is None else enable_user_speaking_frames
        )
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
        self._register_event_handler("on_user_turn_started", sync=True)
        self._register_event_handler("on_reset_aggregation", sync=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # trigger_user_turn_started() is deprecated, but an override of it
        # still gets the reports, as report_user_turn_started() says.
        cls._overrides_trigger_user_turn_started = (
            cls.trigger_user_turn_started is not BaseUserTurnStartStrategy.trigger_user_turn_started
        )
        if cls._overrides_trigger_user_turn_started:
            warn_deprecated(
                f"`{cls.__name__}.trigger_user_turn_started` is deprecated since 1.15.0 and "
                "will be removed in 2.0.0. Use `report_user_turn_started` instead.",
                stacklevel=3,
            )
        # reset() is deprecated. stacklevel=3 steps past ABCMeta.__new__, which
        # calls this hook, so the warning points at the subclass's definition.
        if cls.reset is not BaseUserTurnStartStrategy.reset:
            warn_deprecated(
                f"`{cls.__name__}.reset` is deprecated since 1.6.0 and will be removed in "
                "2.0.0. Use an override of `handle_user_turn_started` instead.",
                stacklevel=3,
            )

    @property
    def resolves_proposed_turn_start_frames(self) -> bool:
        """Whether this strategy resolves proposals into turn starts.

        A :class:`~pipecat.frames.frames.ProposedUserStartedSpeakingFrame` is a
        request for a decision, so a strategy that acts on one consumes it: the
        frame stops travelling, and no resolver further along decides the same
        turn a second time. Override to True in a strategy that handles that
        frame.
        """
        return False

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the strategy.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup.task_manager)

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
            Nothing. A returned :class:`~pipecat.turns.types.ProcessFrameResult`
            is deprecated: ``STOP`` keeps the strategies after this one from
            seeing the frame, and only in a ``StrategyMode.FIRST`` list.
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

    async def report_user_turn_started(
        self,
        *,
        enable_interruptions: bool | None = None,
        enable_user_speaking_frames: bool | None = None,
    ):
        """Report that the user started, by firing ``on_user_turn_started``.

        A strategy calls this when it decides the user's turn has started. The
        turn controller starts the turn on the first decision, or once every
        start strategy has decided in a ``StrategyMode.ALL`` list.

        Args:
            enable_interruptions: Whether to broadcast an interruption for this
                turn, overriding the strategy's configured setting. Pass False
                when something else in the pipeline has already broadcast one.
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStartedSpeakingFrame` for this
                turn. Pass False when something else in the pipeline has already
                emitted it.
        """
        # Backward compatibility: a custom strategy may still override the
        # deprecated trigger_user_turn_started(), so its override gets the
        # report. A report the override makes itself, directly or from a task
        # it starts, fires.
        if self._overrides_trigger_user_turn_started and _running_override.get() is not self:
            token = _running_override.set(self)
            try:
                await self.trigger_user_turn_started(
                    enable_interruptions=enable_interruptions,
                    enable_user_speaking_frames=enable_user_speaking_frames,
                )
            finally:
                _running_override.reset(token)
        else:
            await self._fire_user_turn_started(enable_interruptions, enable_user_speaking_frames)

    @deprecated(
        "`BaseUserTurnStartStrategy.trigger_user_turn_started` is deprecated since 1.15.0 and "
        "will be removed in 2.0.0. Use `report_user_turn_started` instead."
    )
    async def trigger_user_turn_started(
        self,
        *,
        enable_interruptions: bool | None = None,
        enable_user_speaking_frames: bool | None = None,
    ):
        """Report that the user started.

        .. deprecated:: 1.15.0
            Use :meth:`report_user_turn_started` instead. Will be removed in
            2.0.0.

        Args:
            enable_interruptions: Whether to broadcast an interruption for this
                turn, overriding the strategy's configured setting.
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStartedSpeakingFrame` for this
                turn.
        """
        # An override of this method reaches here through super(), so this
        # fires the event rather than handing the report back to it.
        if self._overrides_trigger_user_turn_started:
            await self._fire_user_turn_started(enable_interruptions, enable_user_speaking_frames)
        else:
            await self.report_user_turn_started(
                enable_interruptions=enable_interruptions,
                enable_user_speaking_frames=enable_user_speaking_frames,
            )

    async def trigger_reset_aggregation(self):
        """Trigger the `on_reset_aggregation` event."""
        await self._call_event_handler("on_reset_aggregation")

    async def _fire_user_turn_started(
        self, enable_interruptions: bool | None, enable_user_speaking_frames: bool | None
    ):
        await self._call_event_handler(
            "on_user_turn_started",
            UserTurnStartedParams(
                enable_interruptions=(
                    self._enable_interruptions
                    if enable_interruptions is None
                    else enable_interruptions
                ),
                enable_user_speaking_frames=(
                    self._enable_user_speaking_frames
                    if enable_user_speaking_frames is None
                    else enable_user_speaking_frames
                ),
            ),
        )
