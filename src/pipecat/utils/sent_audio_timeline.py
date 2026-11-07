#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Map a streaming provider's audio offsets back to the input audio's PTS."""

from bisect import bisect_right
from dataclasses import dataclass


@dataclass
class _SentChunk:
    offset_ns: int
    duration_ns: int
    pts: int


class SentAudioTimeline:
    """Where each chunk of audio sent on one provider stream sits on the input's timeline.

    A streaming speech-to-text provider times words in seconds of audio received on
    its stream. That drifts from the input audio's PTS whenever audio never reached
    the provider (muted, gated, dropped while reconnecting) or the input's own media
    time has gaps, so this records both positions for every chunk sent.
    """

    def __init__(self):
        """Initialize an empty timeline."""
        self._chunks: list[_SentChunk] = []
        self._offsets: list[int] = []
        self._sent_ns = 0

    def reset(self):
        """Start a new provider stream, whose offsets start again from zero."""
        self._chunks = []
        self._offsets = []
        self._sent_ns = 0

    def append(self, pts: int | None, num_frames: int, sample_rate: int):
        """Record a chunk sent to the provider.

        Args:
            pts: The chunk's PTS in nanoseconds, or ``None`` if its input had none.
                The chunk still advances the stream, but nothing maps into it.
            num_frames: Samples per channel in the chunk.
            sample_rate: The chunk's sample rate, in Hz.
        """
        duration_ns = num_frames * 1_000_000_000 // sample_rate
        if pts is not None and duration_ns > 0:
            self._chunks.append(_SentChunk(self._sent_ns, duration_ns, pts))
            self._offsets.append(self._sent_ns)
        self._sent_ns += duration_ns

    def to_pts(self, seconds: float, *, end: bool = False) -> int | None:
        """Get the input PTS of a provider stream offset.

        A word's start belongs to the chunk it begins, and its end to the chunk it
        finishes, which differ where two chunks meet across a gap in the input.

        Args:
            seconds: Offset from the start of the provider stream, in seconds.
            end: Whether the offset ends a span, rather than starts one.

        Returns:
            The PTS in nanoseconds, or ``None`` if no recorded chunk holds the offset.
        """
        offset_ns = round(seconds * 1_000_000_000)
        index = bisect_right(self._offsets, offset_ns) - 1
        if end and index >= 0 and self._chunks[index].offset_ns == offset_ns:
            index -= 1
        if index < 0:
            return None
        chunk = self._chunks[index]
        into = offset_ns - chunk.offset_ns
        if into > chunk.duration_ns or (into == chunk.duration_ns and not end):
            return None
        return chunk.pts + into
