#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Response strategy that waits for a conversational opening."""

import time

from pipecat.frames.frames import ResponseFrame
from pipecat.turns.response.announcement import CompletedToolResult
from pipecat.turns.response.base_response_strategy import (
    BaseResponseStrategy,
    ResponseActivityState,
)


class DelayedResponseStrategy(BaseResponseStrategy):
    """Delivers assistant-initiated responses at the next conversational opening.

    An opening requires all of:

    - the bot is not speaking,
    - the user is not speaking,
    - no reactive response is owed (no LLM response streaming, no reactive
      function calls in progress, no deferred post-function-result
      inference), and
    - a settle window has elapsed since the last activity transition, so the
      response doesn't land in the natural pause between the user's utterances
      or right on the heels of the bot's own turn.

    Queuing a response restarts that window too, which is what makes batching
    work: results finishing milliseconds apart all land inside one window and
    are announced together, instead of the first one leaving alone and the
    rest trailing it one spoken turn at a time.

    This strategy never interrupts: its contract is that there is nothing to
    interrupt. Immediately before a scheduled release it re-verifies the
    opening — if the user just started speaking, the batch is held again.

    All pending responses release together as a single batch (one LLM run for
    the merged message appends, rather than one spoken response per frame).
    """

    def __init__(self, *, settle_secs: float = 1.5, **kwargs):
        """Initialize the delayed response strategy.

        Args:
            settle_secs: Seconds of conversational quiet required after the
                last activity transition before a pending response is
                released. Zero releases at the first quiet evaluation.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._settle_secs = settle_secs
        self._last_activity_time: float | None = None

    async def on_activity_changed(self, activity: ResponseActivityState):
        """Restart the settle window on every activity transition."""
        self._last_activity_time = time.monotonic()
        await super().on_activity_changed(activity)

    async def queue_response(self, item: ResponseFrame | CompletedToolResult):
        """Restart the settle window, then accept the response.

        The window has to start when the item arrives, not merely when the
        conversation last changed: results completing in a long-quiet stretch
        would otherwise find an expired window and release one at a time.

        Args:
            item: The response to queue.
        """
        self._last_activity_time = time.monotonic()
        await super().queue_response(item)

    async def should_release(self) -> bool:
        """Release only in a settled conversational opening."""
        activity = self.activity
        if activity.bot_speaking or activity.user_speaking or activity.response_pending:
            return False
        if self._last_activity_time is None:
            # No conversational activity observed at all (e.g. text-mode
            # evals, where no speaking frames flow): release immediately.
            return True
        remaining = self._settle_secs - (time.monotonic() - self._last_activity_time)
        if remaining <= 0:
            return True
        await self._schedule_release_check(remaining)
        return False
