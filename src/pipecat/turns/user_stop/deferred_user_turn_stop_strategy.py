#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Wrapper that defers a stop strategy's finalization to another strategy."""

import warnings

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.utils.deprecation import deprecated


@deprecated(
    "`DeferredUserTurnStopStrategy` is deprecated since 1.15.0 and will be removed in 2.0.0. "
    "No replacement."
)
class DeferredUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """Wraps a stop strategy and suppresses its ``on_user_turn_stopped`` event.

    Event subscriptions added to the wrapper are forwarded directly to
    the inner strategy, except for ``on_user_turn_stopped``, which is
    dropped. The inner strategy's frame-side and inference-triggered
    events therefore reach external listeners (the controller, etc.)
    unchanged; finalization is left to another strategy in the chain
    such as ``LLMTurnCompletionUserTurnStopStrategy``.

    The turn controller now runs the turn's inference itself, and a stop
    strategy that waits for that inference ends the turn in either stop mode,
    so a deferred strategy just decides as the strategy it wraps.

    Use the :func:`deferred` helper for ergonomic construction::

        stop=[
            deferred(TurnAnalyzerUserTurnStopStrategy(turn_analyzer=...)),
            LLMTurnCompletionUserTurnStopStrategy(),
        ]

    .. deprecated:: 1.15.0
        No replacement. The turn controller runs the turn's inference itself,
        and a stop strategy that waits for that inference, such as
        :class:`~pipecat.turns.user_stop.LLMTurnCompletionUserTurnStopStrategy`,
        ends the turn in either stop mode, so pass the wrapped strategy itself.
        Will be removed in 2.0.0.
    """

    def __init__(self, inner: BaseUserTurnStopStrategy, **kwargs):
        """Initialize the deferred wrapper.

        Args:
            inner: The strategy whose finalization should be deferred.
            **kwargs: Additional keyword arguments forwarded to the base
                class.
        """
        super().__init__(**kwargs)
        self._inner = inner

    @property
    def inner(self) -> BaseUserTurnStopStrategy:
        """Return the wrapped strategy."""
        return self._inner

    @property
    def resolves_proposed_turn_stop_frames(self) -> bool:
        """Report what the inner strategy does with proposals.

        Frame processing is forwarded, so a proposal reaches the inner strategy
        and is resolved there. Deferring finalization changes when the turn ends,
        not who decides it.
        """
        return self._inner.resolves_proposed_turn_stop_frames

    def add_event_handler(self, event_name: str, handler):
        """Forward event subscriptions to the inner strategy.

        ``on_user_turn_stopped`` is silently dropped — that's the whole
        point of the wrapper. Every other event handler is attached to
        the inner strategy directly, so the inner's events reach the
        listener without any per-event proxy method on the wrapper.
        """
        if event_name == "on_user_turn_stopped":
            return
        self._inner.add_event_handler(event_name, handler)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the inner strategy."""
        await super().setup(setup)
        await self._inner.setup(setup)

    async def cleanup(self):
        """Clean up the inner strategy."""
        await super().cleanup()
        await self._inner.cleanup()

    async def handle_user_turn_started(self):
        """Forward the turn-started callback to the inner strategy."""
        await self._inner.handle_user_turn_started()

    async def handle_user_turn_stopped(self):
        """Forward the turn-stopped callback to the inner strategy."""
        await self._inner.handle_user_turn_stopped()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult | None:
        """Forward frame processing to the inner strategy."""
        return await self._inner.process_frame(frame)


@deprecated("`deferred` is deprecated since 1.15.0 and will be removed in 2.0.0. No replacement.")
def deferred(strategy: BaseUserTurnStopStrategy) -> DeferredUserTurnStopStrategy:
    """Defer this stop strategy's finalization to another strategy.

    Wraps ``strategy`` in a :class:`DeferredUserTurnStopStrategy`: the
    inner strategy continues to drive inference-triggered events, but
    its ``on_user_turn_stopped`` event is suppressed. Use when another
    strategy in the chain (e.g.
    ``LLMTurnCompletionUserTurnStopStrategy``) owns finalization.

    Example::

        stop=[
            deferred(TurnAnalyzerUserTurnStopStrategy(turn_analyzer=...)),
            LLMTurnCompletionUserTurnStopStrategy(),
        ]

    Args:
        strategy: The stop strategy to defer.

    Returns:
        A wrapper that exposes the inner strategy's behavior with
        finalization suppressed.

    .. deprecated:: 1.15.0
        No replacement. The turn controller runs the turn's inference itself,
        and a stop strategy that waits for that inference, such as
        :class:`~pipecat.turns.user_stop.LLMTurnCompletionUserTurnStopStrategy`,
        ends the turn in either stop mode, so pass the wrapped strategy itself.
        Will be removed in 2.0.0.
    """
    # This call already warned; the wrapper it builds would warn again.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return DeferredUserTurnStopStrategy(strategy)
