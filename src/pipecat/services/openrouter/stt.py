#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenRouter speech-to-text service implementation.

This module provides integration with OpenRouter's speech-to-text API for
transcribing audio input to text. OpenRouter's STT endpoint is not a
file-upload API like OpenAI's — it accepts base64-encoded audio with an
explicit format field, supports multiple providers (e.g. OpenAI Whisper),
and returns JSON with the transcribed text.
"""

import base64
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_latency import WHISPER_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

OPENROUTER_STT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_STT_MODEL = "openai/whisper-1"


@dataclass
class OpenRouterSTTSettings(STTSettings):
    """Settings for OpenRouterSTTService.

    Parameters:
        temperature: Sampling temperature (0–1). Lower values are more
            deterministic. Omit to use the model default.
        prompt: Optional text to guide transcription style or provide
            keyword hints.
    """

    temperature: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prompt: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class OpenRouterSTTService(SegmentedSTTService):
    """OpenRouter Speech-to-Text service that transcribes audio to text.

    Sends audio to OpenRouter's ``POST /api/v1/audio/transcriptions``
    endpoint. Audio is base64-encoded and sent as JSON (not a multipart
    file upload). Supports multiple providers discoverable via
    ``GET /api/v1/models?output_modalities=transcription``.

    Example usage::

        stt = OpenRouterSTTService(
            api_key=os.environ["OPENROUTER_API_KEY"],
            settings=OpenRouterSTTService.Settings(
                model="openai/whisper-1",
                language=Language.EN,
            ),
        )
    """

    Settings = OpenRouterSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = OPENROUTER_STT_BASE_URL,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = WHISPER_TTFS_P99,
        **kwargs,
    ):
        """Initialize OpenRouter STT service.

        Args:
            api_key: OpenRouter API key for authentication.
            base_url: OpenRouter API base URL. Defaults to ``https://openrouter.ai/api/v1``.
            settings: Model, language, and transcription options. When omitted,
                defaults to model ``openai/whisper-1`` and language ``en``.
            ttfs_p99_latency: P99 latency from speech end to first transcript in
                seconds. Override for your deployment.
            **kwargs: Additional keyword arguments passed to SegmentedSTTService.
        """
        default_settings = self.Settings(
            model=OPENROUTER_DEFAULT_STT_MODEL,
            language=Language.EN,
            temperature=None,
            prompt=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        # OpenRouter STT uses ISO-639-1 codes; Language values are already
        # in that form (e.g. Language.EN == "en", Language.FR == "fr").
        return str(language).split("-")[0].lower()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio using OpenRouter's transcription API.

        Args:
            audio: Raw PCM audio bytes at the pipeline sample rate.

        Yields:
            Frame: A ``TranscriptionFrame`` on success, or an ``ErrorFrame``
                on failure.
        """
        try:
            await self.start_processing_metrics()

            payload: dict = {
                "model": self._settings.model,
                "input_audio": {
                    "data": base64.b64encode(audio).decode("utf-8"),
                    "format": "wav",
                },
            }

            language = self._settings.language
            if language is not None:
                payload["language"] = self.language_to_service_language(language)

            if self._settings.temperature is not None:
                payload["temperature"] = self._settings.temperature

            if self._settings.prompt is not None:
                payload["prompt"] = self._settings.prompt

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/audio/transcriptions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"{self} transcription error (status: {response.status}, error: {error_text})"
                        )
                        yield ErrorFrame(
                            error=f"Transcription error (status: {response.status}): {error_text}"
                        )
                        return

                    data = await response.json()

            await self.stop_processing_metrics()

            text = data.get("text", "").strip()

            if not text:
                logger.warning(f"{self}: received empty transcription")
                return

            await self._handle_transcription(text, True, self._settings.language)
            logger.debug(f"{self}: Transcription: [{text}]")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), result=data)

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
