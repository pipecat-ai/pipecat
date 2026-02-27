#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smallest STT service implementation."""

import io
import json
from enum import Enum
from typing import AsyncGenerator, List, Optional, Union
from urllib.parse import urlencode

import httpx
import numpy as np
import soundfile as sf
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")


class AudioChannel(int, Enum):
    MONO = 1
    STEREO = 2


class AudioEncoding(str, Enum):
    LINEAR16 = "linear16"
    FLAC = "flac"
    MULAW = "mulaw"
    OPUS = "opus"


class SensitiveData(str, Enum):
    PCI = "pci"
    SSN = "ssn"
    NUMBERS = "numbers"


class EventType(str, Enum):
    TRANSCRIPTION = "transcription"
    ERROR = "error"


class TranscriptionResponse(BaseModel):
    type: EventType = EventType.TRANSCRIPTION
    text: str
    isEndOfTurn: bool
    isFinal: bool


class ErrorResponse(BaseModel):
    type: EventType = EventType.ERROR
    message: str
    error: Union[List[str], str]


class Model(Enum):
    LIGHTNING = "lightning"


def language_to_smallest_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Smallest's language code format."""
    BASE_LANGUAGES = {
        Language.EN: "en",
        Language.HI: "hi",
    }

    result = BASE_LANGUAGES.get(language)

    if not result:
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result

class SmallestSTTHTTPService(SegmentedSTTService):
    """Class to transcribe audio with the Smallest AI HTTP API."""
    def __init__(
        self,
        *,
        api_key: str,
        model: str | Model = Model.LIGHTNING,
        language: str = "en",
        age_detection: bool = False,
        emotion_detection: bool = False,
        gender_detection: bool = False,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._language = language

        self._client = httpx.AsyncClient()
        self._url = "https://waves-api.smallest.ai/api/v1/lightning/get_text"
        self._headers = {
            "Authorization": f"Bearer {self._api_key}"
        }
        self._payload = {
            "model": model if isinstance(model, str) else model.value,
            "age_detection": "true" if age_detection else "false",
            "gender_detection": "true" if gender_detection else "false",
            "emotion_detection": "true" if emotion_detection else "false",
            "language": language
        }

    @property
    def language(self) -> str:
        return self._language

    def audio_array_to_wav_buffer(self, audio_float, sr=16000):
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_float, sr, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        return wav_buffer

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate metrics."""
        return True

    def _load(self):
        """No-op for API-based service."""
        pass

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribes given audio using the Smallest AI HTTP API."""
        # Divide by 32768 because we have signed 16-bit data.
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        wav_buffer = self.audio_array_to_wav_buffer(audio_float, self._sample_rate)

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                content=wav_buffer.getvalue(),
                params=self._payload
            )
            response.raise_for_status()
            result = response.json()
            text: str = result.get("transcription", "").strip()
        except httpx.HTTPStatusError as e:
            logger.error(f"API error: {e.response.status_code} - {e.response.text}")
            yield ErrorFrame(f"API error: {e.response.status_code}")
            return
        except Exception as e:
            logger.exception(f"Error during transcription API call: {type(e).__name__}: {e}")
            yield ErrorFrame(f"Transcription error: {type(e).__name__}: {e}")
            return

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        if text:
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601())
            await self._handle_transcription(text, True, self.language)