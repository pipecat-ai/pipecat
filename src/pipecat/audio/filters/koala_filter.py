#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Sequence

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    import pvkoala
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the Koala filter, you need to `pip install pipecat-ai[koala]`.")
    raise Exception(f"Missing module: {e}")


class KoalaFilter(BaseAudioFilter):
    """This is an audio filter that uses Koala Noise Suppression (from
    PicoVoice).

    """

    def __init__(self, *, access_key: str) -> None:
        self._access_key = access_key

        self._filtering = True
        self._sample_rate = 0
        self._koala = pvkoala.create(access_key=f"{self._access_key}")
        self._koala_ready = True
        self._audio_buffer = bytearray()

    async def start(self, sample_rate: int):
        self._sample_rate = sample_rate
        if self._sample_rate != self._koala.sample_rate:
            logger.warning(
                f"Koala filter needs sample rate {self._koala.sample_rate} (got {self._sample_rate})"
            )
            self._koala_ready = False

    async def stop(self):
        self._koala.reset()

    async def process_frame(self, frame: FilterControlFrame):
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        if not self._koala_ready or not self._filtering:
            return audio

        self._audio_buffer.extend(audio)

        filtered_data: Sequence[int] = []

        num_frames = len(self._audio_buffer) // 2
        while num_frames >= self._koala.frame_length:
            # Grab the number of frames required by Koala.
            num_bytes = self._koala.frame_length * 2
            audio = bytes(self._audio_buffer[:num_bytes])
            # Process audio
            data = np.frombuffer(audio, dtype=np.int16).tolist()
            filtered_data += self._koala.process(data)
            # Adjust audio buffer and check again
            self._audio_buffer = self._audio_buffer[num_bytes:]
            num_frames = len(self._audio_buffer) // 2

        filtered = np.array(filtered_data, dtype=np.int16).tobytes()

        return filtered
