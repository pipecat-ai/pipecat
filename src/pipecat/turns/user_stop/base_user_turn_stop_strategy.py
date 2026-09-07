#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base user turn stop strategy for determining when the user stopped speaking."""

import warnings
from dataclasses import dataclass

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.turns.types import ProcessFrameResult
from pipecat.utils.base_object import BaseObject


@dataclass
class UserTurnStoppedParams:
    """Parameters emitted when a user turn stops.

    These parameters are passed to the `on_user_turn_stopped` event and provide
    contextual information about how the end of user turn should be handled by
    the user aggregator.

    Parameters:
        enable_user_speaking_frames: Whether to emit
            :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` for this
            turn. False when the turn end was already announced elsewhere — by a
            shared :class:`~pipecat.turns.user_turn_processor.UserTurnProcessor`,
            or by a service that emits turn frames rather than proposing them.

    """

    enable_user_speaking_frames: bool


class BaseUserTurnStopStrategy(BaseObject):
    """Base class for strategies that determine when the user stops speaking.

    Subclasses should implement logic to detect when the user stops
    speaking. This could be based on analyzing incoming frames (such as
    transcriptions), conversation state, or other heuristics.

    Events triggered by strategies:

      - `on_push_frame`: Indicates the strategy wants to push a frame.
      - `on_user_turn_inference_triggered`: Signals that enough evidence
        exists to start LLM inference for the current user turn. In most
        cases this fires together with `on_user_turn_stopped`. Strategies
        that gate finalization on the LLM (e.g.
        ``LLMTurnCompletionUserTurnStopStrategy``) fire only this event
        upstream and a separate strategy fires `on_user_turn_stopped` once
        the LLM confirms the turn is complete.
      - `on_user_turn_stopped`: Signals that the user turn is semantically
        final. Observers, transcript appenders, and UI indicators should
        bind this event.

    """

    def __init__(self, *, enable_user_speaking_frames: bool | None = None, **kwargs):
        """Initialize the base user turn stop strategy.

        Args:
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` when a
                turn ends, unless :meth:`trigger_user_turn_stopped` overrides it
                for a given turn.

                .. deprecated:: 1.8.0
                    Use :meth:`trigger_user_turn_stopped`'s
                    ``enable_user_speaking_frames`` argument instead. Will be
                    removed in 2.0.0.

            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        if enable_user_speaking_frames is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "`BaseUserTurnStopStrategy.enable_user_speaking_frames` is deprecated "
                    "since 1.8.0 and will be removed in 2.0.0. Use "
                    "`trigger_user_turn_stopped(enable_user_speaking_frames=...)` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        self._enable_user_speaking_frames = (
            True if enable_user_speaking_frames is None else enable_user_speaking_frames
        )
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
        self._register_event_handler("on_user_turn_inference_triggered", sync=True)
        self._register_event_handler("on_user_turn_stopped", sync=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # reset() is deprecated.
        if cls.reset is not BaseUserTurnStopStrategy.reset:
            warnings.warn(
                f"`{cls.__name__}` overrides `reset`, which is deprecated since 1.6.0 "
                "and will be removed in 2.0.0. Override `handle_user_turn_started` and "
                "`handle_user_turn_stopped` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def resolves_proposed_turn_stop_frames(self) -> bool:
        """Whether this strategy resolves proposals into turn stops.

        A :class:`~pipecat.frames.frames.ProposedUserStoppedSpeakingFrame` is a
        request for a decision, so a strategy that acts on one consumes it: the
        frame stops travelling, and no resolver further along decides the same
        turn a second time. Override to True in a strategy that handles that
        frame — including one that holds the proposal for a while before
        deciding.
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
            Use :meth:`handle_user_turn_started` and
            :meth:`handle_user_turn_stopped` instead. Will be removed in 2.0.0.

        A stop strategy is reset at both turn boundaries — armed on start,
        cleaned up on stop — so this historically ran at both. New strategies
        should override the boundary callbacks directly, which say plainly
        *when* the work runs, and reset however they like inside them (a private
        helper called from both, an extra ``clear()`` on stop, and so on).
        """
        pass

    async def handle_user_turn_started(self):
        """Notify the strategy that a user turn has started.

        The controller calls this on every stop strategy when a turn begins.
        Override to run, for example, logic to arm the strategy to detect the
        end of the turn now in progress.
        """
        # Backward compatibility: a custom strategy may still override the
        # deprecated reset(); invoke it here (the base reset() is a no-op).
        await self.reset()

    async def handle_user_turn_stopped(self):
        """Notify the strategy that the user turn has stopped.

        The controller calls this on every stop strategy when a turn ends,
        regardless of which strategy (or the stop watchdog timeout) ended it.
        Override to run stop-specific logic — e.g. dropping a turn analyzer's
        buffered speech that must not survive an externally-ended turn.
        """
        # Backward compatibility: a custom strategy may still override the
        # deprecated reset(); invoke it here (the base reset() is a no-op).
        await self.reset()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult | None:
        """Process an incoming frame to decide whether the user stopped speaking.

        Subclasses should override this to implement logic that decides whether
        the user has stopped speaking.

        Args:
            frame: The frame to be analyzed.

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

    async def trigger_user_turn_stopped(self, *, enable_user_speaking_frames: bool | None = None):
        """Fire both ``on_user_turn_inference_triggered`` and ``on_user_turn_stopped``.

        Most strategies call this when they decide a turn has ended. To
        defer finalization to another strategy (so this strategy fires
        only the inference-triggered event), wrap this strategy with
        :func:`~pipecat.turns.user_stop.deferred` instead of changing
        the trigger call.

        Args:
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` for this
                turn. Pass False when something else in the pipeline has already
                emitted it.
        """
        await self.trigger_user_turn_inference_triggered()
        await self.trigger_user_turn_finalized(
            enable_user_speaking_frames=enable_user_speaking_frames
        )

    async def trigger_user_turn_inference_triggered(self):
        """Trigger only the `on_user_turn_inference_triggered` event."""
        await self._call_event_handler("on_user_turn_inference_triggered")

    async def trigger_user_turn_finalized(self, *, enable_user_speaking_frames: bool | None = None):
        """Trigger only the `on_user_turn_stopped` event.

        Args:
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` for this
                turn. Pass False when something else in the pipeline has already
                emitted it.
        """
        await self._call_event_handler(
            "on_user_turn_stopped",
            UserTurnStoppedParams(
                enable_user_speaking_frames=(
                    self._enable_user_speaking_frames
                    if enable_user_speaking_frames is None
                    else enable_user_speaking_frames
                )
            ),
        )
