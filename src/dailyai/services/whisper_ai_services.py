"""This module implements Whisper transcription with a locally-downloaded model."""
import asyncio
from enum import Enum
import logging
from typing import BinaryIO
from faster_whisper import WhisperModel
from dailyai.services.local_stt_service import LocalSTTService


class Model(Enum):
    """Class of basic Whisper model selection options"""
    TINY = "tiny"
    BASE = "base"
    MEDIUM = "medium"
    LARGE = "large-v3"
    DISTIL_LARGE_V2 = "Systran/faster-distil-whisper-large-v2"
    DISTIL_MEDIUM_EN = "Systran/faster-distil-whisper-medium.en"


class WhisperSTTService(LocalSTTService):
    """Class to transcribe audio with a locally-downloaded Whisper model"""
    _model: WhisperModel

    # Model configuration
    _model_name: Model
    _device: str
    _compute_type: str

    def __init__(self, model_name: Model = Model.DISTIL_MEDIUM_EN,
                 device: str = "auto",
                 compute_type: str = "default"):

        super().__init__()
        self.logger: logging.Logger = logging.getLogger("dailyai")
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._load()

    def _load(self):
        """Loads the Whisper model. Note that if this is the first time
        this model is being run, it will take time to download."""
        model = WhisperModel(
            self._model_name.value,
            device=self._device,
            compute_type=self._compute_type)
        self._model = model

    async def run_stt(self, audio: BinaryIO) -> str:
        """Transcribes given audio using Whisper"""
        segments, _ = await asyncio.to_thread(self._model.transcribe, audio)
        res: str = ""
        for segment in segments:
            res += f"{segment.text} "
        return res
