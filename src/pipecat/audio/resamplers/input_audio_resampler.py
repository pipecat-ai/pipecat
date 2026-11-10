#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Resampler for a continuous stream of input audio."""

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.utils import create_stream_resampler


class InputAudioResampler(BaseAudioResampler):
    """Resamples a continuous stream of input audio to the rate its consumer takes.

    Audio already at the output rate passes through unchanged, and audio at
    another rate goes through a stream resampler. A stream resampler converts
    between one pair of rates, so a change of either rate starts a new stream
    and drops the few milliseconds the previous one still held.

    Input audio keeps arriving, so its stream never ends; call :meth:`reset`
    where the consumer stops taking it for a while, such as when it is muted.

    Example::

        resampler = InputAudioResampler()
        audio = await resampler.resample(frame.audio, frame.sample_rate, 16000)
    """

    def __init__(self):
        """Initialize the resampler."""
        self._resampler: BaseAudioResampler | None = None
        self._rates = (0, 0)

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resample one chunk of the stream.

        Args:
            audio: Input audio data as raw bytes (16-bit signed integers).
            in_rate: Original sample rate in Hz.
            out_rate: Target sample rate in Hz.

        Returns:
            Resampled audio data as raw bytes (16-bit signed integers).
        """
        if in_rate == out_rate:
            return audio
        if not self._resampler or (in_rate, out_rate) != self._rates:
            self._resampler = create_stream_resampler(clear_after_secs=None)
            self._rates = (in_rate, out_rate)
        return await self._resampler.resample(audio, in_rate, out_rate)

    async def flush(self) -> bytes:
        """Emit the audio the stream resampler is delaying and reset it.

        Returns:
            The remaining resampled audio as raw bytes, empty if nothing has
            been resampled yet.
        """
        if not self._resampler:
            return b""
        return await self._resampler.flush()

    async def reset(self):
        """Discard the audio held in the stream resampler."""
        if self._resampler:
            await self._resampler.reset()
