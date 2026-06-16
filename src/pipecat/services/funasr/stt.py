#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""FunASR speech-to-text service for Pipecat.

Self-hosted STT using FunASR (SenseVoice / Paraformer / Fun-ASR-Nano). Runs the
model locally with no cloud API; strong on Chinese and 50+ languages.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use FunASR, you need to `pip install pipecat-ai[funasr]`.")
    raise Exception(f"Missing module: {e}")


class FunASRSTTService(SegmentedSTTService):
    """Self-hosted speech-to-text using FunASR.

    Runs a FunASR model (SenseVoice, Paraformer, Fun-ASR-Nano) locally and
    transcribes VAD-segmented audio. Non-streaming: Pipecat buffers each speech
    segment and passes it to ``run_stt`` as 16-bit PCM.
    """

    def __init__(
        self,
        *,
        model: str = "iic/SenseVoiceSmall",
        device: str = "cpu",
        hub: str = "ms",
        language: Optional[Language] = None,
        sense_voice_language: str = "auto",
        use_itn: bool = True,
        sample_rate: Optional[int] = 16000,
        **kwargs,
    ):
        """Initialize the FunASR STT service.

        Args:
            model: FunASR model id. Defaults to ``iic/SenseVoiceSmall`` (ModelScope).
                Use e.g. ``FunAudioLLM/SenseVoiceSmall`` with ``hub="hf"``.
            device: Inference device, e.g. ``cpu`` or ``cuda:0``.
            hub: Model hub, ``ms`` (ModelScope) or ``hf`` (HuggingFace).
            language: Language label attached to produced TranscriptionFrames.
            sense_voice_language: SenseVoice decoding language (``auto``/``zh``/``en``/...).
            use_itn: Apply inverse text normalization (punctuation, digits).
            sample_rate: Input sample rate. FunASR expects 16 kHz.
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._model_name = model
        self._language = language
        self._sense_voice_language = sense_voice_language
        self._use_itn = use_itn
        logger.debug(f"Loading FunASR model {model} on {device} (hub={hub})")
        self._model = AutoModel(model=model, device=device, hub=hub, disable_update=True)
        logger.debug("Loaded FunASR model")

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate processing metrics."""
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe a segment of audio with FunASR.

        Args:
            audio: Raw audio bytes in 16-bit signed PCM format.

        Yields:
            Frame: A TranscriptionFrame with the transcribed text, or an
            ErrorFrame on failure.
        """
        if not self._model:
            yield ErrorFrame("FunASR model not available")
            return

        await self.start_processing_metrics()

        # Divide by 32768 because we have signed 16-bit data.
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        def _infer() -> str:
            gen_kwargs = dict(input=audio_float, cache={}, use_itn=self._use_itn, batch_size_s=300)
            if "SenseVoice" in self._model_name:
                gen_kwargs["language"] = self._sense_voice_language
            res = self._model.generate(**gen_kwargs)
            text = res[0]["text"] if res else ""
            return rich_transcription_postprocess(text)

        try:
            text = await asyncio.to_thread(_infer)
        except Exception as e:
            logger.exception(f"FunASR transcription error: {e}")
            await self.stop_processing_metrics()
            yield ErrorFrame(f"FunASR transcription error: {e}")
            return

        await self.stop_processing_metrics()

        text = text.strip()
        if text:
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, "", time_now_iso8601(), self._language, result=text)
