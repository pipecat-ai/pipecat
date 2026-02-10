#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smallest AI speech-to-text service implementation.

This module provides a segmented (HTTP-based) Speech-to-Text service using
Smallest AI's Waves API. Audio is buffered during speech, then sent as a single
request once the user stops speaking (VAD-triggered).
"""

import io
from enum import Enum
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TranscriptionFrame,
)
from pipecat.services.stt_latency import SMALLEST_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import httpx
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")

try:
    import numpy as np
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")

try:
    import soundfile as sf
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")


def language_to_smallest_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Smallest's language code format.

    Smallest AI currently supports English and Hindi. Falls back to extracting
    the base language code if the exact Language enum isn't mapped.

    Args:
        language: The Language enum value to convert.

    Returns:
        The Smallest language code string, or None if unsupported.
    """
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


class SmallestSTTModel(str, Enum):
    """Available Smallest AI STT models."""

    LIGHTNING = "lightning"


class SmallestSTTService(SegmentedSTTService):
    """Smallest AI speech-to-text service using the Waves HTTP API.

    This is a segmented STT service that buffers audio while the user speaks
    (using VAD) and sends the complete audio segment to Smallest AI's HTTP
    endpoint for transcription once the user stops speaking.

    Requires VAD to be enabled in the pipeline.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Smallest STT service.

        Parameters:
            language: Language code for transcription. Defaults to "en".
            age_detection: Enable age detection. Defaults to False.
            emotion_detection: Enable emotion detection. Defaults to False.
            gender_detection: Enable gender detection. Defaults to False.
        """

        language: str = "en"
        age_detection: bool = False
        emotion_detection: bool = False
        gender_detection: bool = False

    def __init__(
        self,
        *,
        api_key: str,
        model: str = SmallestSTTModel.LIGHTNING,
        url: str = "https://waves-api.smallest.ai/api/v1/lightning/get_text",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        ttfs_p99_latency: Optional[float] = SMALLEST_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Smallest AI STT service.

        Args:
            api_key: Smallest AI API key for authentication.
            model: Model to use for transcription. Defaults to "lightning".
            url: API endpoint URL. Defaults to the Smallest Waves API endpoint.
            sample_rate: Audio sample rate. If None, will be determined from the
                start frame.
            params: Configuration parameters for the STT service.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment.
            **kwargs: Additional arguments passed to the parent SegmentedSTTService.
        """
        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        params = params or SmallestSTTService.InputParams()

        self._api_key = api_key
        self._url = url
        self._language = params.language

        model_str = model.value if isinstance(model, Enum) else model
        self.set_model_name(model_str)

        self._client = httpx.AsyncClient()
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        self._payload = {
            "model": model_str,
            "age_detection": "true" if params.age_detection else "false",
            "gender_detection": "true" if params.gender_detection else "false",
            "emotion_detection": "true" if params.emotion_detection else "false",
            "language": params.language,
        }

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Smallest STT supports metrics generation.
        """
        return True

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing.

        This method is decorated with @traced_stt for observability.
        The actual work (pushing frames) is done in run_stt; this method
        exists solely as a tracing hook.
        """
        pass

    def _audio_bytes_to_wav_buffer(self, audio: bytes) -> io.BytesIO:
        """Convert raw PCM16 audio bytes to a WAV-formatted buffer.

        The Smallest API expects WAV-formatted audio. This converts raw signed
        16-bit PCM audio bytes into a WAV buffer with proper headers.

        Args:
            audio: Raw PCM16 audio bytes.

        Returns:
            A BytesIO buffer containing WAV-formatted audio data.
        """
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_float, self.sample_rate, format="WAV", subtype="PCM_16")
        wav_buffer.seek(0)
        return wav_buffer

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio using the Smallest AI HTTP API.

        Called by the base SegmentedSTTService when the user stops speaking.
        The audio parameter contains the complete WAV-encoded speech segment.

        Args:
            audio: WAV-encoded audio bytes from the speech segment.

        Yields:
            TranscriptionFrame on success, ErrorFrame on failure.
        """
        wav_buffer = self._audio_bytes_to_wav_buffer(audio)

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                content=wav_buffer.getvalue(),
                params=self._payload,
            )
            response.raise_for_status()
            result = response.json()
            text: str = result.get("transcription", "").strip()
        except httpx.HTTPStatusError as e:
            logger.error(f"{self} API error: {e.response.status_code} - {e.response.text}")
            yield ErrorFrame(error=f"Smallest API error: {e.response.status_code}", exception=e)
            return
        except Exception as e:
            logger.exception(f"{self} transcription error: {type(e).__name__}: {e}")
            yield ErrorFrame(error=f"Smallest transcription error: {type(e).__name__}: {e}")
            return

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        if text:
            logger.debug(f"Transcription: [{text}]")
            await self._handle_transcription(text, True, self._language)
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
            )

    async def cleanup(self):
        """Clean up resources used by the Smallest STT service."""
        await super().cleanup()
        await self._client.aclose()
