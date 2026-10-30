#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base user turn stop strategy for determining when the user stopped speaking."""

from contextvars import ContextVar
from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.turns.types import ProcessFrameResult, UserTurnSpeculation
from pipecat.utils.base_object import BaseObject
from pipecat.utils.deprecation import deprecated, warn_deprecated

# The stop strategy whose override of the deprecated trigger_user_turn_stopped()
# is running.
_running_override: ContextVar[object | None] = ContextVar("_running_stop_override", default=None)


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
        confirms_speculation: Whether this turn end confirms a speculative
            response that was already generated. The aggregator then writes the
            turn to the context without running inference again, since the
            answer is already on its way.

    """

    enable_user_speaking_frames: bool
    confirms_speculation: bool = False


class BaseUserTurnStopStrategy(BaseObject):
    """Base class for strategies that determine when the user stops speaking.

    Subclasses should implement logic to detect when the user stops
    speaking. This could be based on analyzing incoming frames (such as
    transcriptions), conversation state, or other heuristics.

    Events triggered by strategies:

      - `on_push_frame`: Indicates the strategy wants to push a frame.
      - `on_broadcast_frame`: Indicates the strategy wants to broadcast a frame.
      - `on_user_turn_stopped`: Reports the strategy's decision that the user
        is done, from :meth:`report_user_turn_stopped`. The turn controller
        combines the stop strategies' decisions into the turn end.
      - `on_user_turn_inference_triggered`: Starts a speculative inference for
        a turn that hasn't ended yet, carrying a
        :class:`~pipecat.turns.types.UserTurnSpeculation`, from
        :meth:`start_user_turn_speculation`.
      - `on_user_turn_speculation_cancelled`: Withdraws that speculation, from
        :meth:`cancel_user_turn_speculation`.

    """

    # Whether a subclass overrides the deprecated trigger_user_turn_stopped().
    _overrides_trigger_user_turn_stopped = False

    def __init__(self, *, enable_user_speaking_frames: bool | None = None, **kwargs):
        """Initialize the base user turn stop strategy.

        Args:
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` when a
                turn ends, unless :meth:`report_user_turn_stopped` overrides it
                for a given turn.

                .. deprecated:: 1.8.0
                    Use :meth:`report_user_turn_stopped`'s
                    ``enable_user_speaking_frames`` argument instead. Will be
                    removed in 2.0.0.

            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        if enable_user_speaking_frames is not None:
            warn_deprecated(
                "`BaseUserTurnStopStrategy.enable_user_speaking_frames` is deprecated "
                "since 1.8.0 and will be removed in 2.0.0. Use "
                "`report_user_turn_stopped(enable_user_speaking_frames=...)` instead.",
                stacklevel=2,
            )
        self._enable_user_speaking_frames = (
            True if enable_user_speaking_frames is None else enable_user_speaking_frames
        )
        self._warned_inference_without_speculation = False
        self._register_event_handler("on_push_frame", sync=True)
        self._register_event_handler("on_broadcast_frame", sync=True)
        self._register_event_handler("on_user_turn_inference_triggered", sync=True)
        self._register_event_handler("on_user_turn_speculation_cancelled", sync=True)
        self._register_event_handler("on_user_turn_stopped", sync=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # trigger_user_turn_stopped() is deprecated, but an override of it still
        # gets the reports, as report_user_turn_stopped() says.
        cls._overrides_trigger_user_turn_stopped = (
            cls.trigger_user_turn_stopped is not BaseUserTurnStopStrategy.trigger_user_turn_stopped
        )
        if cls._overrides_trigger_user_turn_stopped:
            warn_deprecated(
                f"`{cls.__name__}.trigger_user_turn_stopped` is deprecated since 1.15.0 and "
                "will be removed in 2.0.0. Use `report_user_turn_stopped` instead.",
                stacklevel=3,
            )
        # reset() is deprecated. stacklevel=3 steps past ABCMeta.__new__, which
        # calls this hook, so the warning points at the subclass's definition.
        if cls.reset is not BaseUserTurnStopStrategy.reset:
            warn_deprecated(
                f"`{cls.__name__}.reset` is deprecated since 1.6.0 and will be removed in "
                "2.0.0. Use overrides of `handle_user_turn_started` and "
                "`handle_user_turn_stopped` instead.",
                stacklevel=3,
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

    @property
    def waits_for_inference(self) -> bool:
        """Whether this strategy waits for the turn's inference before deciding.

        In a ``StrategyMode.ALL`` list, the turn's inference runs once every
        other stop strategy has decided, and this strategy then decides on
        it. Override to True in a strategy that decides from the turn's
        answer, such as one reading the LLM's turn-completion marker.
        """
        return False

    @property
    def decision_scheduled(self) -> bool:
        """Whether this strategy has a decision of its own scheduled.

        While a stop strategy has one, the turn controller's stop timeout
        waits for it instead of ending the turn. Override to return True only
        while the strategy is sure to decide soon on its own, as when a timer
        of its own will fire, so a turn whose decision never comes still
        times out.
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

    async def cancel_user_turn_speculation(self):
        """Withdraw the speculation this strategy has in flight.

        Emitted rather than pushed so the host broadcasts it the same way it
        broadcasts the turn end. A withdrawal that took the queued path would
        reach a downstream gate *after* the turn end that follows it, and the
        gate would release the response the withdrawal was meant to discard.
        """
        await self._call_event_handler("on_user_turn_speculation_cancelled")

    async def report_user_turn_stopped(
        self,
        *,
        enable_user_speaking_frames: bool | None = None,
        confirms_speculation: bool = False,
    ):
        """Report that the user is done, by firing ``on_user_turn_stopped``.

        A strategy calls this when it decides the user's turn has ended. The
        controller combines the stop strategies' decisions into the turn end,
        and runs the turn's inference itself once they have decided.

        Args:
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` for this
                turn. Pass False when something else in the pipeline has already
                emitted it.
            confirms_speculation: Whether a response generated ahead of this
                turn ending already answers it, so inference should not run
                again.
        """
        # Backward compatibility: a custom strategy may still override the
        # deprecated trigger_user_turn_stopped(), which never confirmed a
        # speculation, so its override gets every other report. A report the
        # override makes itself, directly or from a task it starts, fires.
        if (
            self._overrides_trigger_user_turn_stopped
            and not confirms_speculation
            and _running_override.get() is not self
        ):
            token = _running_override.set(self)
            try:
                await self.trigger_user_turn_stopped(
                    enable_user_speaking_frames=enable_user_speaking_frames
                )
            finally:
                _running_override.reset(token)
        else:
            await self._fire_user_turn_stopped(enable_user_speaking_frames, confirms_speculation)

    async def start_user_turn_speculation(self, speculation: UserTurnSpeculation):
        """Start a speculative inference, by firing ``on_user_turn_inference_triggered``.

        The controller runs a turn's inference itself once the stop strategies
        have decided, so a strategy only calls this to answer ahead of that, as an
        eager end of turn does.

        Args:
            speculation: The speculation to run, against a provisional context
                carrying the turn text it names, for a turn that hasn't ended
                yet. The speculation gate keeps its response back until the turn
                is confirmed.
        """
        await self._call_event_handler("on_user_turn_inference_triggered", speculation)

    @deprecated(
        "`BaseUserTurnStopStrategy.trigger_user_turn_speculation_cancelled` is deprecated "
        "since 1.15.0 and will be removed in 2.0.0. Use `cancel_user_turn_speculation` instead."
    )
    async def trigger_user_turn_speculation_cancelled(self):
        """Withdraw the speculation this strategy has in flight.

        .. deprecated:: 1.15.0
            Use :meth:`cancel_user_turn_speculation` instead. Will be removed in
            2.0.0.
        """
        await self.cancel_user_turn_speculation()

    @deprecated(
        "`BaseUserTurnStopStrategy.trigger_user_turn_stopped` is deprecated since 1.15.0 and "
        "will be removed in 2.0.0. Use `report_user_turn_stopped` instead."
    )
    async def trigger_user_turn_stopped(self, *, enable_user_speaking_frames: bool | None = None):
        """Report that the user is done.

        .. deprecated:: 1.15.0
            Use :meth:`report_user_turn_stopped` instead. Will be removed in
            2.0.0.

        Args:
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` for this
                turn. Pass False when something else in the pipeline has already
                emitted it.
        """
        # An override of this method reaches here through super(), so this
        # fires the event rather than handing the report back to it.
        if self._overrides_trigger_user_turn_stopped:
            await self._fire_user_turn_stopped(enable_user_speaking_frames, False)
        else:
            await self.report_user_turn_stopped(
                enable_user_speaking_frames=enable_user_speaking_frames
            )

    @deprecated(
        "`BaseUserTurnStopStrategy.trigger_user_turn_inference_triggered` is deprecated since "
        "1.15.0 and will be removed in 2.0.0. Use `start_user_turn_speculation` instead."
    )
    async def trigger_user_turn_inference_triggered(
        self, *, speculation: UserTurnSpeculation | None = None
    ):
        """Start a speculative inference.

        .. deprecated:: 1.15.0
            Use :meth:`start_user_turn_speculation` instead. Will be removed in
            2.0.0.

        Args:
            speculation: The speculation to run. A call without one is ignored,
                with a warning the first time, since the turn controller runs the
                turn's inference itself.
        """
        if not speculation:
            if not self._warned_inference_without_speculation:
                self._warned_inference_without_speculation = True
                logger.warning(
                    f"{self}: trigger_user_turn_inference_triggered() without a speculation "
                    "does nothing, since the turn controller runs the turn's inference once "
                    "the stop strategies have decided. Call report_user_turn_stopped() "
                    "when the user is done."
                )
            return

        await self.start_user_turn_speculation(speculation)

    @deprecated(
        "`BaseUserTurnStopStrategy.trigger_user_turn_finalized` is deprecated since 1.15.0 "
        "and will be removed in 2.0.0. Use `report_user_turn_stopped` instead."
    )
    async def trigger_user_turn_finalized(
        self,
        *,
        enable_user_speaking_frames: bool | None = None,
        confirms_speculation: bool = False,
    ):
        """Fire only ``on_user_turn_stopped``.

        .. deprecated:: 1.15.0
            Use :meth:`report_user_turn_stopped` instead, which fires the same
            event. Will be removed in 2.0.0.

        Args:
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` for this
                turn. Pass False when something else in the pipeline has already
                emitted it.
            confirms_speculation: Whether a response generated ahead of this
                turn ending already answers it, so inference should not run
                again.
        """
        await self._fire_user_turn_stopped(enable_user_speaking_frames, confirms_speculation)

    async def _fire_user_turn_stopped(
        self, enable_user_speaking_frames: bool | None, confirms_speculation: bool
    ):
        await self._call_event_handler(
            "on_user_turn_stopped",
            UserTurnStoppedParams(
                enable_user_speaking_frames=(
                    self._enable_user_speaking_frames
                    if enable_user_speaking_frames is None
                    else enable_user_speaking_frames
                ),
                confirms_speculation=confirms_speculation,
            ),
        )
