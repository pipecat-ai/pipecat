#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Streaming windowed-sinc resampler whose output keeps pace with its input.

Every call returns exactly the audio its input amounts to at the output rate,
carrying any fraction of a sample into the next call, so equal input chunks
come out as equal output chunks. That makes it safe to use on audio that has
already been paced, such as the chunks an output transport sends.
"""

import time

from pipecat_resampler import StreamResampler

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler


class SincStreamAudioResampler(BaseAudioResampler):
    """Streaming resampler with a polyphase windowed-sinc filter.

    The output keeps pace with the input: every call returns exactly the
    audio its input amounts to at the output rate, so equal input chunks come
    out as equal output chunks, the first one included. The filter delays the
    audio by a few milliseconds instead of holding output back.

    The filter keeps the lower rate's band flat up to 90% of its Nyquist
    frequency, and attenuates aliasing and images by about 90 dB. The
    resampling itself runs in the pipecat-resampler C extension.

    Notes:
        - Only supports mono audio (1 channel).
        - Input must be 16-bit signed PCM audio as raw bytes.
        - A resampler converts between one pair of rates.
    """

    def __init__(self, *, clear_after_secs: float | None = 0.2, **kwargs):
        """Initialize the resampler.

        Args:
            clear_after_secs: Seconds of inactivity after which the audio held in
                the filter is discarded, so a new stream doesn't start with the
                end of the previous one. Set to ``None`` to never discard it.
                Defaults to ``0.2``.
            **kwargs: Reserved for forward compatibility; currently ignored.
        """
        self._clear_after_secs = clear_after_secs
        self._resampler: StreamResampler | None = None
        self._last_resample_time = 0.0

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resample one chunk of audio.

        Args:
            audio: Input audio data as raw bytes (16-bit signed integers).
            in_rate: Original sample rate in Hz.
            out_rate: Target sample rate in Hz.

        Returns:
            Resampled audio data as raw bytes (16-bit signed integers).

        Raises:
            ValueError: If the resampler already converts between other rates.
        """
        if in_rate == out_rate:
            return audio
        resampler = self._use_rates(in_rate, out_rate)
        self._maybe_clear(resampler)
        return resampler.process(audio)

    async def flush(self) -> bytes:
        """Emit the audio the filter is delaying and reset for the next stream.

        Returns:
            The remaining resampled audio as raw bytes, empty if no audio has
            been resampled yet.
        """
        if not self._resampler:
            return b""
        return self._resampler.flush()

    async def reset(self):
        """Discard the audio held in the filter and reset for the next stream."""
        if self._resampler:
            self._resampler.reset()

    def _use_rates(self, in_rate: int, out_rate: int) -> StreamResampler:
        """Create the resampler on first use, and refuse another pair of rates after."""
        if not self._resampler:
            self._resampler = StreamResampler(in_rate, out_rate)
        elif (in_rate, out_rate) != (self._resampler.in_rate, self._resampler.out_rate):
            raise ValueError(
                f"SincStreamAudioResampler cannot be reused with different sample rates: "
                f"expected {self._resampler.in_rate}->{self._resampler.out_rate}, "
                f"got {in_rate}->{out_rate}"
            )
        return self._resampler

    def _maybe_clear(self, resampler: StreamResampler):
        """Discard the held audio after more than `clear_after_secs` of inactivity."""
        now = time.time()
        if (
            self._clear_after_secs is not None
            and now - self._last_resample_time > self._clear_after_secs
        ):
            resampler.reset()
        self._last_resample_time = now
