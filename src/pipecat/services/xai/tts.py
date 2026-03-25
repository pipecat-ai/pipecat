#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""xAI text-to-speech service implementation.

Uses xAI's HTTP TTS endpoint documented at:
https://docs.x.ai/developers/model-capabilities/audio/text-to-speech
"""

from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts


def language_to_xai_language(language: Language) -> Optional[str]:
    """Convert a Language enum to xAI language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding xAI language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.AR: "ar-EG",
        Language.AR_EG: "ar-EG",
        Language.AR_SA: "ar-SA",
        Language.AR_AE: "ar-AE",
        Language.BN: "bn",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es-ES",
        Language.ES_ES: "es-ES",
        Language.ES_MX: "es-MX",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.PT: "pt-PT",
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",
        Language.RU: "ru",
        Language.TR: "tr",
        Language.VI: "vi",
        Language.ZH: "zh",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class XAITTSSettings(TTSSettings):
    """Settings for XAIHttpTTSService."""

    pass


class XAIHttpTTSService(TTSService):
    """xAI HTTP text-to-speech service.

    The service requests raw PCM audio so emitted ``TTSAudioRawFrame`` objects
    match Pipecat's downstream expectations without extra decoding.
    """

    Settings = XAITTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.x.ai/v1/tts",
        sample_rate: Optional[int] = None,
        encoding: Optional[str] = "pcm",
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the xAI TTS service.

        Args:
            api_key: xAI API key for authentication.
            base_url: xAI TTS endpoint. Defaults to ``https://api.x.ai/v1/tts``.
            sample_rate: Audio sample rate. If None, uses default.
            encoding: Output encoding format. Defaults to "pcm".
            aiohttp_session: Optional shared aiohttp session.
            settings: Runtime-updatable settings.
            **kwargs: Additional keyword arguments passed to ``TTSService``.
        """
        default_settings = self.Settings(
            model=None,
            voice="eve",
            language=Language.EN,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session
        self._session_owner = aiohttp_session is None
        self._encoding = encoding

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to xAI language format.

        Args:
            language: The language to convert.

        Returns:
            The xAI-specific language code, or None if not supported.
        """
        return language_to_xai_language(language)

    async def start(self, frame):
        """Start the xAI TTS service."""
        await super().start(frame)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

    async def stop(self, frame):
        """Stop the xAI TTS service."""
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame):
        """Cancel the xAI TTS service."""
        await super().cancel(frame)
        await self._close_session()

    async def _close_session(self):
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()
        if self._session_owner:
            self._session = None

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using xAI's TTS API."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

        payload = {
            "text": text,
            "voice_id": self._settings.voice,
            "output_format": {
                "codec": self._encoding,
                "sample_rate": self.sample_rate,
            },
        }
        if self._settings.language:
            payload["language"] = str(self._settings.language)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        measuring_ttfb = True
        try:
            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error = await response.text(errors="ignore")
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {response.status}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                async for chunk in response.content.iter_chunked(self.chunk_size):
                    if not chunk:
                        continue
                    if measuring_ttfb:
                        await self.stop_ttfb_metrics()
                        measuring_ttfb = False
                    yield TTSAudioRawFrame(
                        chunk,
                        self.sample_rate,
                        1,
                        context_id=context_id,
                    )
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
