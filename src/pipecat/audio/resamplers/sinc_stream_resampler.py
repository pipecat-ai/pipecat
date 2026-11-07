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

import math
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler

# How much the filter attenuates what the output rate can't carry, and how much
# of the lower rate's band it keeps flat.
_ATTENUATION_DB = 90.0
_PASSBAND = 0.9

# Up to this many filter phases, outputs are computed one phase at a time on
# strided views of the input. Above it, each output's window is gathered.
_MAX_PHASE_GROUPS = 8

# Most input samples copied into windows at once. Outputs are computed in
# blocks this bounds, so a long chunk doesn't copy a window per output sample
# all at once; a paced chunk fits in one block.
_MAX_BLOCK_SAMPLES = 1 << 19


class SincStreamAudioResampler(BaseAudioResampler):
    """Streaming resampler with a polyphase windowed-sinc filter.

    The output keeps pace with the input: every call returns exactly the
    audio its input amounts to at the output rate, so equal input chunks come
    out as equal output chunks, the first one included. The filter delays the
    audio by a few milliseconds instead of holding output back.

    The filter keeps the lower rate's band flat up to 90% of its Nyquist
    frequency, and attenuates aliasing and images by about 90 dB.

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
        self._in_rate = 0
        self._out_rate = 0
        self._up = 1
        self._down = 1
        self._taps = np.ones((1, 1), dtype=np.float32)
        self._history = np.zeros(0, dtype=np.float32)
        self._offset = 0
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
        self._use_rates(in_rate, out_rate)
        self._maybe_clear()
        return self._process(np.frombuffer(audio, dtype=np.int16))

    async def flush(self) -> bytes:
        """Emit the audio the filter is delaying and reset for the next stream.

        Returns:
            The remaining resampled audio as raw bytes, empty if no audio has
            been resampled yet.
        """
        if not self._in_rate:
            return b""
        tail = self._process(np.zeros(self._delay_samples(), dtype=np.int16))
        self._clear()
        return tail

    async def reset(self):
        """Discard the audio held in the filter and reset for the next stream."""
        self._clear()

    @property
    def _width(self) -> int:
        """Number of input samples each output sample is computed from."""
        return self._taps.shape[1]

    def _use_rates(self, in_rate: int, out_rate: int):
        """Set up the filter on first use, and refuse another pair of rates after."""
        if not self._in_rate:
            self._set_up(in_rate, out_rate)
        elif (in_rate, out_rate) != (self._in_rate, self._out_rate):
            raise ValueError(
                f"SincStreamAudioResampler cannot be reused with different sample rates: "
                f"expected {self._in_rate}->{self._out_rate}, got {in_rate}->{out_rate}"
            )

    def _set_up(self, in_rate: int, out_rate: int):
        """Design the filter for a pair of rates and start with empty history."""
        divisor = math.gcd(in_rate, out_rate)
        self._in_rate, self._out_rate = in_rate, out_rate
        self._up, self._down = out_rate // divisor, in_rate // divisor
        self._taps = self._design_taps()
        self._clear()

    def _design_taps(self) -> np.ndarray:
        """Split a Kaiser-windowed sinc low-pass into one reversed filter per phase."""
        up_rate = self._in_rate * self._up
        nyquist = min(self._in_rate, self._out_rate) / 2
        transition = nyquist * (1 - _PASSBAND)
        cutoff = nyquist - transition / 2
        length = (_ATTENUATION_DB - 7.95) / (2.285 * 2 * math.pi * transition / up_rate) + 1
        length = math.ceil(math.ceil(length) / self._up) * self._up
        t = np.arange(length) - (length - 1) / 2
        beta = 0.1102 * (_ATTENUATION_DB - 8.7)
        taps = np.sinc(2 * cutoff / up_rate * t) * np.kaiser(length, beta)
        taps *= self._up / taps.sum()
        phases = taps.reshape(-1, self._up).T[:, ::-1]
        return np.ascontiguousarray(phases, dtype=np.float32)

    def _delay_samples(self) -> int:
        """Input samples it takes for the last input to reach the output."""
        return math.ceil((self._up * self._width - 1) / (2 * self._up))

    def _maybe_clear(self):
        """Discard the history after more than `clear_after_secs` of inactivity."""
        now = time.time()
        if (
            self._clear_after_secs is not None
            and now - self._last_resample_time > self._clear_after_secs
        ):
            self._clear()
        self._last_resample_time = now

    def _clear(self):
        """Start the next input with the filter holding only silence."""
        self._history = np.zeros(self._width - 1, dtype=np.float32)
        self._offset = 0

    def _process(self, samples: np.ndarray) -> bytes:
        """Resample int16 samples, carrying the filter's history to the next call."""
        length = len(samples)
        if not length:
            return b""
        buffer = np.empty(self._width - 1 + length, dtype=np.float32)
        buffer[: self._width - 1] = self._history
        buffer[self._width - 1 :] = samples
        # Output n sits at n * down on the input upsampled by `up`. The offset
        # is where the next output sits relative to the next input there.
        count = -(-(length * self._up - self._offset) // self._down)
        output = self._convolve(buffer, count)
        self._offset += count * self._down - length * self._up
        self._history = buffer[length:].copy()
        np.rint(output, out=output)
        np.clip(output, -32768, 32767, out=output)
        return output.astype(np.int16).tobytes()

    def _convolve(self, buffer: np.ndarray, count: int) -> np.ndarray:
        """Compute this call's output samples from the buffered input, block by block."""
        convolve = (
            self._convolve_by_phase if self._up <= _MAX_PHASE_GROUPS else self._convolve_gathered
        )
        block = max(1, _MAX_BLOCK_SAMPLES // self._width)
        if count <= block:
            return convolve(buffer, self._offset, count)
        output = np.empty(count, dtype=np.float32)
        for first in range(0, count, block):
            size = min(block, count - first)
            start, offset = divmod(self._offset + first * self._down, self._up)
            output[first : first + size] = convolve(buffer[start:], offset, size)
        return output

    def _convolve_by_phase(self, buffer: np.ndarray, offset: int, count: int) -> np.ndarray:
        """Compute outputs one phase at a time, each phase a strided view of the input."""
        output = np.empty(count, dtype=np.float32)
        item = buffer.itemsize
        # Outputs `up` apart share a phase and start `down` input samples apart.
        for first in range(min(self._up, count)):
            start, phase = divmod(offset + first * self._down, self._up)
            outputs = len(range(first, count, self._up))
            windows = np.ndarray(
                shape=(outputs, self._width),
                dtype=np.float32,
                buffer=buffer,
                offset=start * item,
                strides=(self._down * item, item),
            )
            # The windows overlap, which BLAS can't read in place, and copying
            # them out costs less than the slower dot products without BLAS.
            output[first :: self._up] = np.ascontiguousarray(windows) @ self._taps[phase]
        return output

    def _convolve_gathered(self, buffer: np.ndarray, offset: int, count: int) -> np.ndarray:
        """Compute outputs by gathering each one's window and phase."""
        positions = offset + np.arange(count) * self._down
        starts, phases = np.divmod(positions, self._up)
        windows = sliding_window_view(buffer, self._width)[starts]
        return np.einsum("ij,ij->i", windows, self._taps[phases])
