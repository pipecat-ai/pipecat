#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    import noisereduce as nr
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the noisereduce filter, you need to `pip install pipecat-ai[noisereduce]`."
    )
    raise Exception(f"Missing module: {e}")


class NoisereduceFilter(BaseAudioFilter):
    def __init__(self) -> None:
        self._filtering = True
        self._sample_rate = 0

    async def start(self, sample_rate: int):
        self._sample_rate = sample_rate

    async def stop(self):
        pass

    async def process_frame(self, frame: FilterControlFrame):
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        if not self._filtering:
            return audio

        data = np.frombuffer(audio, dtype=np.int16)

        # Add a small epsilon to avoid division by zero.
        epsilon = 1e-10
        data = data.astype(np.float32) + epsilon

        # Noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=self._sample_rate)
        audio = np.clip(reduced_noise, -32768, 32767).astype(np.int16).tobytes()

        return audio
