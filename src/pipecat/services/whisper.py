#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements Whisper transcription with a locally-downloaded model."""

import asyncio
from enum import Enum
from typing import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.ai_services import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Whisper, you need to `pip install pipecat-ai[whisper]`.")
    raise Exception(f"Missing module: {e}")


class Model(Enum):
    """Class of basic Whisper model selection options"""

    TINY = "tiny"
    BASE = "base"
    MEDIUM = "medium"
    LARGE = "large-v3"
    DISTIL_LARGE_V2 = "Systran/faster-distil-whisper-large-v2"
    DISTIL_MEDIUM_EN = "Systran/faster-distil-whisper-medium.en"


class WhisperSTTService(SegmentedSTTService):
    """Class to transcribe audio with a locally-downloaded Whisper model"""

    def __init__(
        self,
        *,
        model: str | Model = Model.DISTIL_MEDIUM_EN,
        device: str = "auto",
        compute_type: str = "default",
        no_speech_prob: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._device: str = device
        self._compute_type = compute_type
        self.set_model_name(model if isinstance(model, str) else model.value)
        self._no_speech_prob = no_speech_prob
        self._model: WhisperModel | None = None
        self._load()

    def can_generate_metrics(self) -> bool:
        return True

    def _load(self):
        """Loads the Whisper model. Note that if this is the first time
        this model is being run, it will take time to download.
        """
        logger.debug("Loading Whisper model...")
        self._model = WhisperModel(
            self.model_name, device=self._device, compute_type=self._compute_type
        )
        logger.debug("Loaded Whisper model")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribes given audio using Whisper"""
        if not self._model:
            logger.error(f"{self} error: Whisper model not available")
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        # Divide by 32768 because we have signed 16-bit data.
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = await asyncio.to_thread(self._model.transcribe, audio_float)
        text: str = ""
        for segment in segments:
            if segment.no_speech_prob < self._no_speech_prob:
                text += f"{segment.text} "

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        if text:
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, "", time_now_iso8601())
