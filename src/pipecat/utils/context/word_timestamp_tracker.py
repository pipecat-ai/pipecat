#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Timestamps for the words a TTS service speaks."""

from dataclasses import dataclass

from pipecat.utils.time import seconds_to_nanoseconds


@dataclass
class WordTimestamp:
    """A word a TTS provider reported, and when it is spoken.

    Parameters:
        word: The word.
        timestamp: Seconds from the start of the word's audio.
        context_id: The audio context the word belongs to.
        includes_inter_frame_spaces: Whether the word carries its own spacing.
    """

    word: str
    timestamp: float
    context_id: str | None
    includes_inter_frame_spaces: bool = False


class WordTimestampTracker:
    """Turns word times reported from the start of their audio into PTS.

    Timing starts when a context's audio starts playing, or after the last
    word if that is later, so words keep their order across contexts. Words
    reported before then are cached and handed back by :meth:`start`.

    ``last_pts`` is the PTS of the last word frame, and
    ``uninterruptible_last_pts`` that of the last uninterruptible one.
    """

    def __init__(self):
        """Initialize the word timestamp tracker."""
        self._baseline: int | None = None
        self._cached: list[WordTimestamp] = []
        self.last_pts: int = 0
        self.uninterruptible_last_pts: int = 0

    @property
    def started(self) -> bool:
        """Whether words are being timed."""
        return self._baseline is not None

    def start(self, now: int) -> list[WordTimestamp]:
        """Start timing words, if not started yet.

        Args:
            now: The current clock time, in nanoseconds.

        Returns:
            The words cached until now, to time in order.
        """
        if self._baseline is not None:
            return []
        self._baseline = max(self.last_pts, now)
        cached, self._cached = self._cached, []
        return cached

    def cache(self, word: WordTimestamp):
        """Hold a word until timing starts.

        Args:
            word: The word reported before timing started.
        """
        self._cached.append(word)

    def pts(self, timestamp: float) -> int:
        """Get the PTS of a word.

        Args:
            timestamp: Seconds from the start of the word's audio.

        Returns:
            The word's PTS, in nanoseconds.
        """
        assert self._baseline is not None
        return self._baseline + seconds_to_nanoseconds(timestamp)

    def reset(self):
        """Stop timing words and drop the cached ones."""
        self._baseline = None
        self._cached = []
