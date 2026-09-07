#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio volume tracking over a rolling window of recent audio."""

import math

from pipecat.audio.utils import calculate_audio_volume

# Loudness is measured over a BS.1770 gating block, which is 400ms. The audio
# chunks flowing through a pipeline are shorter than that, so they accumulate
# into a rolling window of this size.
VOLUME_WINDOW_SECS = 0.4


class AudioVolumeTracker:
    """Tracks the volume of an audio stream over a rolling window.

    Audio is fed in chunks of any size with :meth:`update`, which retains the
    most recent ``VOLUME_WINDOW_SECS`` of it, and :attr:`volume` measures that
    window. Volume reads 0 until the window holds enough audio for loudness to
    be measurable, and the window is discarded if the sample rate changes.

    Measuring costs a few hundred microseconds and grows with the sample rate,
    so it happens on read and the result is cached until more audio arrives.
    Callers that report volume less often than they receive audio pay only for
    the reads.

    Audio is expected to be mono; interleaved channels would be measured as if
    they were consecutive samples.
    """

    def __init__(self):
        """Initialize the volume tracker."""
        self._sample_rate = 0
        self._window_num_bytes = 0
        self._buffer = b""
        # None once the window holds audio that hasn't been measured yet.
        self._volume: float | None = 0.0

    @property
    def volume(self) -> float:
        """Get the volume of the audio in the rolling window.

        Returns:
            Volume between 0 (quiet) and 1 (loud). Reads 0 until the window
            holds a measurable amount of audio.
        """
        if self._volume is None:
            self._volume = calculate_audio_volume(self._buffer, self._sample_rate)
        return self._volume

    def update(self, audio: bytes, sample_rate: int):
        """Add audio to the rolling window.

        Args:
            audio: Audio data as raw bytes (16-bit signed integers, mono).
            sample_rate: Sample rate of the audio in Hz.
        """
        if sample_rate != self._sample_rate:
            self._sample_rate = sample_rate
            # Rounded up so the window is never a sample short of a gating
            # block, which loudness rejects.
            self._window_num_bytes = math.ceil(VOLUME_WINDOW_SECS * sample_rate) * 2
            self.reset()

        self._buffer = (self._buffer + audio)[-self._window_num_bytes :]
        if len(self._buffer) == self._window_num_bytes:
            self._volume = None

    def reset(self):
        """Clear the rolling window and the tracked volume."""
        self._buffer = b""
        self._volume = 0.0
