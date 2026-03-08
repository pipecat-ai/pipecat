#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""G.711 mu-law (PCMU) codec using NumPy lookup tables.

Provides fast encode/decode via pre-built 65536-entry (encode) and 256-entry
(decode) lookup tables. Used by the RTP session to convert between PCM16 and
G.711 wire format.
"""

from __future__ import annotations

import numpy as np

# mu-law constants
_BIAS = 0x84
_CLIP = 32635
_EXP_LUT = np.array([0, 132, 396, 924, 1980, 4092, 8316, 16764], dtype=np.int16)


def _build_encode_lut() -> np.ndarray:
    """Build 65536-entry encode LUT: int16 sample -> uint8 mu-law byte."""
    lut = np.zeros(65536, dtype=np.uint8)
    for i in range(65536):
        sample = i if i < 32768 else i - 65536
        sign = 0
        if sample < 0:
            sign = 0x80
            sample = -sample
        sample = min(sample, _CLIP)
        sample += _BIAS
        exponent = 7
        exp_mask = 0x4000
        for _ in range(8):
            if sample & exp_mask:
                break
            exponent -= 1
            exp_mask >>= 1
        mantissa = (sample >> (exponent + 3)) & 0x0F
        byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
        lut[i & 0xFFFF] = byte
    return lut


def _build_decode_lut() -> np.ndarray:
    """Build 256-entry decode LUT: uint8 mu-law byte -> int16 sample."""
    lut = np.zeros(256, dtype=np.int16)
    for i in range(256):
        b = ~i & 0xFF
        sign = b & 0x80
        exponent = (b >> 4) & 0x07
        mantissa = b & 0x0F
        sample = int(_EXP_LUT[exponent]) + (mantissa << (exponent + 3))
        if sign:
            sample = -sample
        lut[i] = np.int16(max(-32768, min(32767, sample)))
    return lut


class G711Codec:
    """G.711 mu-law codec with lazy-initialized LUT singleton."""

    _instance: G711Codec | None = None

    def __init__(self):
        """Initialize the codec by building encode and decode lookup tables."""
        self._encode_lut = _build_encode_lut()
        self._decode_lut = _build_decode_lut()

    @classmethod
    def get_instance(cls) -> G711Codec:
        """Get or create the singleton codec instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, pcm: np.ndarray) -> np.ndarray:
        """Encode int16 PCM samples to mu-law bytes."""
        indices = pcm.view(np.uint16)
        return self._encode_lut[indices]

    def decode(self, ulaw: np.ndarray) -> np.ndarray:
        """Decode mu-law bytes to int16 PCM samples."""
        return self._decode_lut[ulaw]


# Pre-computed index arrays for 2x upsampling (160 -> 320 samples).
_UP2_X_OLD = np.arange(160, dtype=np.float64)
_UP2_X_NEW = np.arange(320, dtype=np.float64) / 2


def resample_up(samples: np.ndarray, factor: int) -> np.ndarray:
    """Upsample by integer factor using linear interpolation.

    Args:
        samples: Input int16 PCM samples.
        factor: Upsampling factor (e.g. 2 for 8kHz -> 16kHz).

    Returns:
        Upsampled int16 PCM array.
    """
    if len(samples) == 0:
        return np.array([], dtype=np.int16)
    if factor == 2 and len(samples) == 160:
        result = np.interp(_UP2_X_NEW, _UP2_X_OLD, samples.astype(np.float64))
        return np.clip(result, -32768, 32767).astype(np.int16)
    indices = np.arange(len(samples) * factor, dtype=np.float64) / factor
    result = np.interp(indices, np.arange(len(samples)), samples.astype(np.float64))
    return np.clip(result, -32768, 32767).astype(np.int16)


def resample_down(samples: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by integer factor with moving-average anti-alias filter.

    Args:
        samples: Input int16 PCM samples.
        factor: Downsampling factor (e.g. 2 for 16kHz -> 8kHz).

    Returns:
        Downsampled int16 PCM array.
    """
    if len(samples) == 0:
        return np.array([], dtype=np.int16)
    pad = factor - 1
    sig = samples.astype(np.float32)
    padded = np.pad(sig, (pad, pad), mode="edge")
    kernel = np.ones(factor, dtype=np.float32) / factor
    filtered = np.convolve(padded, kernel, mode="same")
    filtered = filtered[pad : pad + len(sig)]
    decimated = filtered[::factor]
    return np.clip(decimated, -32768, 32767).astype(np.int16)
