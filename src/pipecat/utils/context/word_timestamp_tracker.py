#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Timestamps for the words a TTS service speaks."""

from dataclasses import dataclass

from pipecat.utils.time import seconds_to_nanoseconds

#: A word a TTS provider reported: ``(word, start)``, or ``(word, start, end)``
#: when the provider reports where it ends, in seconds from the start of its audio.
WordTime = tuple[str, float] | tuple[str, float, float | None]


def word_time_end(word_time: WordTime) -> float | None:
    """Get where a reported word ends, or ``None`` when it wasn't reported."""
    return word_time[2] if len(word_time) > 2 else None


@dataclass
class WordTimestamp:
    """A word a TTS provider reported, and when it is spoken.

    Parameters:
        word: The word.
        timestamp: Seconds from the start of the word's audio.
        context_id: The audio context the word belongs to.
        includes_inter_frame_spaces: Whether the word carries its own spacing.
        end: Seconds from the start of the word's audio to where the word
            ends, or ``None`` when the provider doesn't report it.
    """

    word: str
    timestamp: float
    context_id: str | None
    includes_inter_frame_spaces: bool = False
    end: float | None = None


class WordTimestampTracker:
    """Turns word times reported from the start of their audio into PTS.

    Timing starts when a context's audio starts playing, or where the words
    before it end if that is later. A context's audio is pushed as soon as it
    arrives, often while the one before it is still playing, so it is the
    previous words that say when it is heard. Words reported before timing
    starts are cached and handed back by :meth:`start`.

    ``last_pts`` is the PTS of the last word frame, and
    ``uninterruptible_last_pts`` that of the last uninterruptible one.
    ``last_end_pts`` is the PTS where the words timed so far end, as far as
    the provider reports word ends. :attr:`reply_last_pts` is the PTS of the
    last word since :meth:`start_reply`.
    """

    def __init__(self):
        """Initialize the word timestamp tracker."""
        self._baseline: int | None = None
        self._cached: list[WordTimestamp] = []
        self._reply_start_pts: int = 0
        self.last_pts: int = 0
        self.uninterruptible_last_pts: int = 0
        self.last_end_pts: int = 0

    @property
    def started(self) -> bool:
        """Whether words are being timed."""
        return self._baseline is not None

    @property
    def reply_last_pts(self) -> int | None:
        """The PTS of the reply's last word, or ``None`` if the reply timed no words."""
        return self.last_pts if self.last_pts != self._reply_start_pts else None

    def start_reply(self):
        """Note where a reply's words start, for :attr:`reply_last_pts`."""
        self._reply_start_pts = self.last_pts

    def baseline(self, now: int) -> int:
        """Get the PTS the next context's words are timed from.

        Args:
            now: The current clock time, in nanoseconds.

        Returns:
            The later of now and where the words before it end.
        """
        return max(self.last_pts, self.last_end_pts, now)

    def start(self, now: int) -> list[WordTimestamp]:
        """Start timing words, if not started yet.

        Args:
            now: The current clock time, in nanoseconds.

        Returns:
            The words cached until now, to time in order.
        """
        if self._baseline is not None:
            return []
        self._baseline = self.baseline(now)
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

    def end_word(self, end: float):
        """Note where a timed word ends.

        Args:
            end: Seconds from the start of the word's audio to its end.
        """
        self.last_end_pts = max(self.last_end_pts, self.pts(end))

    def reset(self):
        """Stop timing words and drop the cached ones."""
        self._baseline = None
        self._cached = []
