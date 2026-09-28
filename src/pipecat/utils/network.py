#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base class for network utilities, providing exponential backoff time calculation."""

from dataclasses import dataclass


def exponential_backoff_time(
    attempt: int, min_wait: float = 4, max_wait: float = 10, multiplier: float = 1
) -> float:
    """Calculate exponential backoff wait time.

    Args:
        attempt: Current attempt number (1-based)
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        multiplier: Base multiplier for exponential calculation

    Returns:
        Wait time in seconds
    """
    try:
        exp = 2 ** (attempt - 1) * multiplier
        result = max(0, min(exp, max_wait))
        return max(min_wait, result)
    except (ValueError, ArithmeticError):
        return max_wait


@dataclass
class QuickFailureResult:
    """Outcome of recording one failed connection attempt.

    Parameters:
        is_quick_failure: Whether this attempt lasted less than the tracker's
            ``min_stable_duration``.
        should_give_up: Whether ``max_consecutive_failures`` quick failures
            have now happened in a row and the caller should stop retrying.
    """

    is_quick_failure: bool
    should_give_up: bool


class QuickFailureTracker:
    """Detects a connection that keeps failing immediately after connecting.

    Exponential backoff alone doesn't help when a connection attempt succeeds
    just enough to fail again right away (e.g. a server that accepts a
    WebSocket handshake but then immediately rejects invalid credentials) —
    waiting longer between attempts won't fix that, only stopping will.

    Call ``record()`` with how long each failed attempt lasted. Once
    ``max_consecutive_failures`` attempts in a row each lasted under
    ``min_stable_duration`` seconds, the result's ``should_give_up`` is True,
    telling the caller to stop retrying instead of trying again.
    """

    def __init__(self, min_stable_duration: float = 5.0, max_consecutive_failures: int = 3):
        """Initialize the tracker.

        Args:
            min_stable_duration: Minimum time, in seconds, a connection must
                survive to be considered stable rather than a quick failure.
            max_consecutive_failures: Number of consecutive quick failures
                after which ``record()`` reports that the caller should give up.
        """
        self.min_stable_duration = min_stable_duration
        self.max_consecutive_failures = max_consecutive_failures
        self.count = 0

    def record(self, duration: float) -> QuickFailureResult:
        """Record a failed connection attempt that lasted ``duration`` seconds.

        Increments the consecutive-failure streak if this attempt was a quick
        failure, or resets it if the attempt was stable.

        Args:
            duration: How long the failed attempt lasted, in seconds.

        Returns:
            A `QuickFailureResult` describing this attempt and whether the
            caller should give up.
        """
        is_quick_failure = duration < self.min_stable_duration
        if is_quick_failure:
            self.count += 1
        else:
            self.count = 0
        return QuickFailureResult(
            is_quick_failure=is_quick_failure,
            should_give_up=is_quick_failure and self.count >= self.max_consecutive_failures,
        )

    def reset(self):
        """Reset the consecutive-failure streak, e.g. on a fresh connect."""
        self.count = 0
