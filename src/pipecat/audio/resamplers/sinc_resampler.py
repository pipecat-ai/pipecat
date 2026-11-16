#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""One-shot windowed-sinc resampler for whole clips of audio.

Each call resamples a complete clip, so the output lines up with the input and
nothing is held back for a later call. Use it for audio you already have in
full, such as a recorded tone or a buffered segment.
"""

from pipecat_resampler import Resampler

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler


class SincAudioResampler(BaseAudioResampler):
    """One-shot resampler with a polyphase windowed-sinc filter.

    Each call resamples a whole clip, taken to be silent before and after it.
    The output lines up with the input and holds the clip's length at the
    output rate, rounded to the nearest sample. The filter keeps the lower
    rate's band flat up to 90% of its Nyquist frequency, and attenuates
    aliasing and images by about 90 dB. The resampling itself runs in the
    pipecat-resampler C extension, with the same filter as
    :class:`~pipecat.audio.resamplers.sinc_stream_resampler.SincStreamAudioResampler`.

    Notes:
        - Only supports mono audio (1 channel).
        - Input must be 16-bit signed PCM audio as raw bytes.
    """

    def __init__(self, **kwargs):
        """Initialize the resampler.

        Args:
            **kwargs: Reserved for forward compatibility; currently ignored.
        """
        self._resampler: Resampler | None = None

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resample a whole clip of audio.

        Args:
            audio: Input audio data as raw bytes (16-bit signed integers).
            in_rate: Original sample rate in Hz.
            out_rate: Target sample rate in Hz.

        Returns:
            Resampled audio data as raw bytes (16-bit signed integers).
        """
        if in_rate == out_rate:
            return audio
        return self._use_rates(in_rate, out_rate).resample(audio)

    def _use_rates(self, in_rate: int, out_rate: int) -> Resampler:
        """Return a resampler for these rates, building a new one when they change."""
        resampler = self._resampler
        if not resampler or (resampler.in_rate, resampler.out_rate) != (in_rate, out_rate):
            resampler = self._resampler = Resampler(in_rate, out_rate)
        return resampler
