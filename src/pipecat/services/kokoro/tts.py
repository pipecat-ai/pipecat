#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Kokoro TTS service implementation."""

from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from kokoro import KPipeline
except ImportError as e:
    logger.error(f"Failed to import kokoro: {e}")
    logger.error("In order to use Kokoro, you need to `pip install pipecat-ai[kokoro]`.")
    raise


class KokoroTTSService(TTSService):
    """Kokoro TTS service.

    This service uses Kokoro to generate audio from text.
    """

    def __init__(self, *, lang_code: str = "a", voice: str = "af_heart", sample_rate: int = 24000, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._pipeline = KPipeline(lang_code=lang_code)
        self._voice = voice

    def can_generate_metrics(self) -> bool:
        return False

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        yield TTSStartedFrame()

        generator = self._pipeline(text, voice=self._voice)

        for _, _, audio in generator:
            import torch
            import numpy as np

            # Ensure tensor → numpy
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()

            # Convert float32 [-1, 1] → int16 PCM
            if audio.dtype != np.int16:
                audio = np.clip(audio, -1.0, 1.0)  # avoid clipping distortion
                audio = (audio * 32767).astype(np.int16)

            # Yield proper PCM16 audio frame
            yield TTSAudioRawFrame(audio.tobytes(), self.sample_rate, 1)

        yield TTSStoppedFrame()

