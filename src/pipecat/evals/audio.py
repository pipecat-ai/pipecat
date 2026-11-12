#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio files played as a user's turn.

A turn's ``audio:`` names a recording to play instead of synthesizing its
text. Any format libsndfile reads works: WAV, MP3, FLAC, OGG.
"""

import asyncio

import numpy as np
import soundfile as sf

__all__ = ["load_user_audio"]


async def load_user_audio(path: str) -> tuple[bytes, int]:
    """Read an audio file as the PCM a user turn is sent as.

    The file keeps its own sample rate; the bot resamples on its way in.

    Args:
        path: Path to the audio file.

    Returns:
        Tuple of ``(pcm_bytes, sample_rate)`` -- raw 16-bit little-endian mono PCM.

    Raises:
        ValueError: If the file cannot be read as audio.
    """
    return await asyncio.to_thread(_read, path)


def _read(path: str) -> tuple[bytes, int]:
    """Read and downmix the file (blocking; run off the event loop)."""
    data: np.ndarray
    try:
        # always_2d keeps the channel axis so mono and multi-channel read alike.
        data, sample_rate = sf.read(path, dtype="int16", always_2d=True)
    except Exception as e:
        raise ValueError(f"Could not read user audio {path!r}: {e}") from e

    if data.shape[0] == 0:
        raise ValueError(f"User audio {path!r} is empty")

    # A recording of one speaker carries the same voice on every channel, so the
    # mean is the mono take on it. Averaging in float avoids int16 overflow.
    mono = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1).astype(np.int16)
    return mono.tobytes(), int(sample_rate)
