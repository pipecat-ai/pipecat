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
from pydantic import BaseModel

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class XAITTSSettings(TTSSettings):
    """Settings for XAITTSService."""

    pass


class XAITTSService(TTSService):
    """xAI HTTP text-to-speech service.

    The service requests raw PCM audio so emitted ``TTSAudioRawFrame`` objects
    match Pipecat's downstream expectations without extra decoding.
    """

    Settings = XAITTSSettings
    _settings: Settings

    XAI_DEFAULT_SAMPLE_RATE = 24000
    XAI_PCM_CODEC = "pcm"

    class InputParams(BaseModel):
        """Input parameters for xAI TTS configuration.

        .. deprecated:: 0.0.105
            Use ``settings=XAITTSService.Settings(...)`` instead.

        Parameters:
            language: Language for speech synthesis.
        """

        language: Optional[Language] = None

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.x.ai/v1/tts",
        voice: Optional[str] = None,
        language: Optional[str | Language] = None,
        sample_rate: Optional[int] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        params: Optional[InputParams] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the xAI TTS service.

        Args:
            api_key: xAI API key for authentication.
            base_url: xAI TTS endpoint. Defaults to ``https://api.x.ai/v1/tts``.
            voice: Voice identifier. Defaults to ``"eve"``.

                .. deprecated:: 0.0.105
                    Use ``settings=XAITTSService.Settings(voice=...)`` instead.

            language: BCP-47 or base language code (for example ``"en"`` or ``"pt-BR"``).
                Defaults to ``"en"``.

                .. deprecated:: 0.0.105
                    Use ``settings=XAITTSService.Settings(language=...)`` instead.

            sample_rate: Output sample rate for PCM audio. Defaults to 24000 Hz.
            aiohttp_session: Optional shared aiohttp session.
            params: Deprecated input parameters object.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to ``TTSService``.
        """
        default_settings = self.Settings(
            model=None,
            voice="eve",
            language="en",
        )

        if voice is not None:
            self._warn_init_param_moved_to_settings("voice", "voice")
            default_settings.voice = voice
        if language is not None:
            self._warn_init_param_moved_to_settings("language", "language")
            default_settings.language = (
                self.language_to_service_language(language)
                if isinstance(language, Language)
                else language
            )

        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings and params.language is not None:
                default_settings.language = self.language_to_service_language(params.language)

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            pause_frame_processing=True,
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate or self.XAI_DEFAULT_SAMPLE_RATE,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session
        self._session_owner = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to xAI's language format."""
        return str(language)

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
                "codec": self.XAI_PCM_CODEC,
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
            async with self._session.post(self._base_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error = await response.text(errors="ignore")
                    logger.error(
                        f"{self} error getting audio (status: {response.status}, error: {error})"
                    )
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
                    yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
