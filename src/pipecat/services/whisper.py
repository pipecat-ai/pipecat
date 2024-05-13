#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements Whisper transcription with a locally-downloaded model."""

import asyncio
import time

from enum import Enum
from typing import BinaryIO

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.ai_services import STTService

from loguru import logger

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Whisper, you need to `pip install pipecat-ai[whisper]`.")
    raise Exception(f"Missing module: {e}")


class Model(Enum):
    """Class of basic Whisper model selection options"""
    TINY = "tiny"
    BASE = "base"
    MEDIUM = "medium"
    LARGE = "large-v3"
    DISTIL_LARGE_V2 = "Systran/faster-distil-whisper-large-v2"
    DISTIL_MEDIUM_EN = "Systran/faster-distil-whisper-medium.en"


class WhisperSTTService(STTService):
    """Class to transcribe audio with a locally-downloaded Whisper model"""

    def __init__(self, model_name: Model = Model.DISTIL_MEDIUM_EN,
                 device: str = "auto",
                 compute_type: str = "default"):

        super().__init__()
        self._device: str = device
        self._compute_type = compute_type
        self._model_name: Model = model_name
        self._model: WhisperModel | None = None
        self._load()

    def _load(self):
        """Loads the Whisper model. Note that if this is the first time
        this model is being run, it will take time to download."""
        logger.debug("Loading Whisper model...")
        self._model = WhisperModel(
            self._model_name.value,
            device=self._device,
            compute_type=self._compute_type)
        logger.debug("Loaded Whisper model")

    async def run_stt(self, audio: BinaryIO):
        """Transcribes given audio using Whisper"""
        if not self._model:
            logger.error("Whisper model not available")
            return

        segments, _ = await asyncio.to_thread(self._model.transcribe, audio)
        text: str = ""
        for segment in segments:
            text += f"{segment.text} "

        await self.push_frame(TranscriptionFrame(text, "", int(time.time_ns() / 1000000)))
