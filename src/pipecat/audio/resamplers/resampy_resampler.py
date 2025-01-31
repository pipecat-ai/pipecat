#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import numpy as np
import resampy

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler


class ResampyResampler(BaseAudioResampler):
    """Audio resampler implementation using the resampy library."""

    def __init__(self, **kwargs):
        pass

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        if in_rate == out_rate:
            return audio
        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = resampy.resample(audio_data, in_rate, out_rate, filter="kaiser_fast")
        result = resampled_audio.astype(np.int16).tobytes()
        return result
