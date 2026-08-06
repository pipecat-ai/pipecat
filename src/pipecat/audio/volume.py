#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio volume tracking over a rolling window of recent audio."""

from pipecat.audio.utils import calculate_audio_volume, exp_smoothing

# Loudness is measured over a BS.1770 gating block, which is 400ms. The audio
# chunks flowing through a pipeline are shorter than that, so they accumulate
# into a rolling window of this size.
VOLUME_WINDOW_SECS = 0.4


class AudioVolumeTracker:
    """Tracks the volume of an audio stream over a rolling window.

    Audio is fed in chunks of any size and the most recent
    ``VOLUME_WINDOW_SECS`` of it are retained. Volume reads 0 until the window
    holds enough audio for loudness to be measurable, and the window is
    discarded if the sample rate changes.
    """

    def __init__(self, *, smoothing_factor: float = 0.2):
        """Initialize the volume tracker.

        Args:
            smoothing_factor: Exponential smoothing factor between 0 and 1.
                Higher values follow the measured volume more closely; 1.0
                disables smoothing.
        """
        self._smoothing_factor = smoothing_factor
        self._sample_rate = 0
        self._window_num_bytes = 0
        self._buffer = b""
        self._volume = 0.0

    @property
    def volume(self) -> float:
        """Get the current volume.

        Returns:
            Volume between 0 (quiet) and 1 (loud).
        """
        return self._volume

    def update(self, audio: bytes, sample_rate: int) -> float:
        """Add audio to the rolling window and recompute the volume.

        Args:
            audio: Audio data as raw bytes (16-bit signed integers, mono).
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            Volume between 0 (quiet) and 1 (loud). Reads 0 until the window
            holds a measurable amount of audio.
        """
        if sample_rate != self._sample_rate:
            self._sample_rate = sample_rate
            self._window_num_bytes = round(VOLUME_WINDOW_SECS * sample_rate) * 2
            self.reset()

        self._buffer = (self._buffer + audio)[-self._window_num_bytes :]
        if len(self._buffer) == self._window_num_bytes:
            volume = calculate_audio_volume(self._buffer, self._sample_rate)
            self._volume = exp_smoothing(volume, self._volume, self._smoothing_factor)

        return self._volume

    def reset(self):
        """Clear the rolling window and the tracked volume."""
        self._buffer = b""
        self._volume = 0.0
