#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import numpy as np

from pipecat.audio.filters.base_filter import AudioFilter

import noisereduce as nr


class NoiseReduceFilter(AudioFilter):
    async def filter(self, audio: bytes, sample_rate: int, num_channels: int) -> bytes:
        data = np.frombuffer(audio, dtype=np.int16)

        # Add a small epsilon to avoid division by zero.
        epsilon = 1e-10
        data = data.astype(np.float32) + epsilon

        # Noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=sample_rate)
        audio = np.clip(reduced_noise, -32768, 32767).astype(np.int16).tobytes()

        return audio
