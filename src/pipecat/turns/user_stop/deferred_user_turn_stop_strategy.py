#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Wrapper that defers a stop strategy's finalization to another strategy."""

from pipecat.frames.frames import Frame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class DeferredUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """Wraps a stop strategy and suppresses its ``on_user_turn_stopped`` event.

    Event subscriptions added to the wrapper are forwarded directly to
    the inner strategy, except for ``on_user_turn_stopped``, which is
    dropped. The inner strategy's frame-side and inference-triggered
    events therefore reach external listeners (the controller, etc.)
    unchanged; finalization is left to another strategy in the chain
    such as ``LLMTurnCompletionUserTurnStopStrategy``.

    Use the :func:`deferred` helper for ergonomic construction::

        stop=[
            deferred(TurnAnalyzerUserTurnStopStrategy(turn_analyzer=...)),
            LLMTurnCompletionUserTurnStopStrategy(),
        ]
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

    async def setup(self, task_manager: BaseTaskManager):
        """Set up the inner strategy."""
        await super().setup(task_manager)
        await self._inner.setup(task_manager)

    async def cleanup(self):
        """Clean up the inner strategy."""
        await super().cleanup()
        await self._inner.cleanup()

    async def reset(self):
        """Reset the inner strategy for a new user turn."""
        await super().reset()
        await self._inner.reset()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult | None:
        """Forward frame processing to the inner strategy."""
        return await self._inner.process_frame(frame)


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
    """
    return DeferredUserTurnStopStrategy(strategy)
