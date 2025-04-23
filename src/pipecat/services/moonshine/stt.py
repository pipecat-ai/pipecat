#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# This code originally written by Marmik Pandya (marmikcfc - github.com/marmikcfc)

"""This module implements Moonshine ASR transcription using a locally available ONNX model."""

import asyncio
import time
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Moonshine ASR, you need to install the moonshine-onnx dependencies.")
    raise Exception(f"Missing module: {e}")


class Transcriber:
    """
    Provides an interface to transcribe a speech array using Moonshine ASR. 
    This helper class mirrors the logic in your standalone demo.
    """
    def __init__(self, model_name: str, rate: int = 16000):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.rate = rate
        self.tokenizer = load_tokenizer()

        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        # Warmup the model with an array of zeros.
        self.__call__(np.zeros(int(rate), dtype=np.float32))

    def __call__(self, speech: np.ndarray) -> str:
        """
        Transcribes the provided speech waveform to text.
        """
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()

        # Moonshine expects an array with shape (1, length)
        tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]
        self.inference_secs += time.time() - start_time
        return text


class MoonshineSTTService(SegmentedSTTService):
    """
    Transcribes audio using Moonshine ASR via an ONNX model.
    This service implements the same asynchronous interface as the local Whisper
    service but uses MoonshineOnnxModel. It assumes the provided audio is 16-bit PCM.
    
    Args:
        model_name: The Moonshine model to load (e.g. "moonshine/base").
        rate: Sampling rate of the audio (must be 16000 Hz).
        language: The default language for transcription (stored but not used by Moonshine).
        **kwargs: Additional arguments passed to SegmentedSTTService.
    
    Attributes:
        _transcriber: Instance of Transcriber used for inference.
        _settings: Dictionary storing service settings.
    """
    def __init__(
        self,
        *,
        model_name: str = "moonshine/base",
        rate: int = 16000,
        language: Language = Language.EN,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_model_name(model_name)
        self._rate = rate
        self._transcriber: Optional[Transcriber] = None

        self._settings = {
            "language": language,
        }

        self._load()

    def _load(self):
        """Loads the Moonshine ASR model and warms it up."""
        logger.debug("Loading Moonshine ONNX model...")
        self._transcriber = Transcriber(model_name=self.model_name, rate=self._rate)
        logger.debug("Loaded Moonshine model.")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Transcribes the given 16-bit PCM audio using Moonshine ASR.
        
        Args:
            audio: Raw audio bytes in 16-bit PCM format.
        
        Yields:
            Frame: A TranscriptionFrame if transcription is successful,
                   or an ErrorFrame if an error occurs.
        """
        if not self._transcriber:
            logger.error("Moonshine model not available")
            yield ErrorFrame("Moonshine model not available")
            return

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        # Convert audio bytes to a float32 waveform in the range [-1, 1].
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Execute transcription on a separate thread.
        text: str = await asyncio.to_thread(self._transcriber, audio_float)

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        if text:
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, "", time_now_iso8601(), self._settings["language"])
        else:
            yield ErrorFrame("No transcription produced.") 