#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""G.711 μ-law and A-law conversion for 16-bit PCM audio.

Telephony providers carry audio as G.711, one byte per sample. These functions
convert it to and from the 16-bit signed PCM that pipelines carry, at the same
sample rate, producing the same bytes as the standard library's ``audioop``.
Each conversion is a lookup in a table built when the module is imported.
"""

import numpy as np

# The largest magnitude in each of G.711's eight segments, for the 14-bit
# magnitudes μ-law encodes and the 13-bit magnitudes A-law encodes.
_ULAW_SEGMENT_ENDS = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])
_ALAW_SEGMENT_ENDS = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF])

# Every 16-bit sample, at the index of its unsigned bit pattern.
_SAMPLES = np.arange(1 << 16, dtype=np.uint16).view(np.int16).astype(np.int64)


def _ulaw_to_pcm_table() -> np.ndarray:
    code = ~np.arange(256) & 0xFF
    segment = (code >> 4) & 0x07
    magnitude = ((((code & 0x0F) << 3) + 0x84) << segment) - 0x84
    return np.where(code & 0x80, -magnitude, magnitude).astype(np.int16)


def _alaw_to_pcm_table() -> np.ndarray:
    code = np.arange(256) ^ 0x55
    segment = (code >> 4) & 0x07
    magnitude = ((code & 0x0F) << 4) + np.where(segment == 0, 0x08, 0x108)
    magnitude <<= np.maximum(segment - 1, 0)
    return np.where(code & 0x80, magnitude, -magnitude).astype(np.int16)


def _pcm_to_ulaw_table() -> np.ndarray:
    sample = _SAMPLES >> 2
    magnitude = np.abs(sample) + 0x21
    segment = np.searchsorted(_ULAW_SEGMENT_ENDS, magnitude)
    code = np.where(segment > 7, 0x7F, (segment << 4) | ((magnitude >> (segment + 1)) & 0x0F))
    return (code ^ np.where(sample < 0, 0x7F, 0xFF)).astype(np.uint8)


def _pcm_to_alaw_table() -> np.ndarray:
    sample = _SAMPLES >> 3
    magnitude = np.where(sample < 0, -sample - 1, sample)
    segment = np.searchsorted(_ALAW_SEGMENT_ENDS, magnitude)
    code = (segment << 4) | ((magnitude >> np.maximum(segment, 1)) & 0x0F)
    return (code ^ np.where(sample < 0, 0x55, 0xD5)).astype(np.uint8)


_ULAW_TO_PCM = _ulaw_to_pcm_table()
_ALAW_TO_PCM = _alaw_to_pcm_table()
_PCM_TO_ULAW = _pcm_to_ulaw_table()
_PCM_TO_ALAW = _pcm_to_alaw_table()


def ulaw_to_pcm(ulaw: bytes) -> bytes:
    """Convert μ-law audio to 16-bit PCM.

    Args:
        ulaw: μ-law audio, one byte per sample.

    Returns:
        16-bit signed PCM audio at the same sample rate.
    """
    return _ULAW_TO_PCM[np.frombuffer(ulaw, dtype=np.uint8)].tobytes()


def pcm_to_ulaw(pcm: bytes) -> bytes:
    """Convert 16-bit PCM audio to μ-law.

    Args:
        pcm: 16-bit signed PCM audio.

    Returns:
        μ-law audio at the same sample rate, one byte per sample.

    Raises:
        ValueError: If ``pcm`` has an odd number of bytes.
    """
    return _PCM_TO_ULAW[np.frombuffer(pcm, dtype=np.uint16)].tobytes()


def alaw_to_pcm(alaw: bytes) -> bytes:
    """Convert A-law audio to 16-bit PCM.

    Args:
        alaw: A-law audio, one byte per sample.

    Returns:
        16-bit signed PCM audio at the same sample rate.
    """
    return _ALAW_TO_PCM[np.frombuffer(alaw, dtype=np.uint8)].tobytes()


def pcm_to_alaw(pcm: bytes) -> bytes:
    """Convert 16-bit PCM audio to A-law.

    Args:
        pcm: 16-bit signed PCM audio.

    Returns:
        A-law audio at the same sample rate, one byte per sample.

    Raises:
        ValueError: If ``pcm`` has an odd number of bytes.
    """
    return _PCM_TO_ALAW[np.frombuffer(pcm, dtype=np.uint16)].tobytes()
