#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""timepay.ai text-to-speech service implementations.

This module provides HTTP-based TTS services using timepay.ai API
with support for streaming audio.
"""

from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts


def language_to_timepayai_language(language: Language) -> Optional[str]:
    """Convert a Language enum to timepay.ai language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding timepay.ai language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.BN: "bn",  # Bengali
        Language.EN: "en",  # English (India)
        Language.GU: "gu",  # Gujarati
        Language.HI: "hi",  # Hindi
        Language.KN: "kn",  # Kannada
        Language.ML: "ml",  # Malayalam
        Language.MR: "mr",  # Marathi
        Language.OR: "od",  # Odia
        Language.PA: "pa",  # Punjabi
        Language.TA: "ta",  # Tamil
        Language.TE: "te",  # Telugu
        Language.AS: "as",  # Assamese
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)

class TimepayHttpTTSService(TTSService):
    """Text-to-Speech service using timepay.ai API.

    Converts text to speech using timepay.ai TTS models with support for multiple
    Indian languages.

    Example::

        tts = TimepayHttpTTSService(
            api_key="your-api-key",
            voice_id="Ogbs15oBevLzXsUuTtA1",
            aiohttp_session=session,
            sample_rate=16000
            params=TimepayHttpTTSService.InputParams(
                language=Language.HI,
                speed=1.0
            )
        )
    """

    class InputParams(BaseModel):
        """Input parameters for timepay.ai TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English (India).
            speed: Speech pace multiplier (0.5 to 2.0). Defaults to 1.0.
            add_wav_header: Adds a WAV header to the stream for immediate playback.
        """

        language: Optional[Language] = Language.EN
        speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
        add_wav_header: Optional[bool] = False

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str = "Ogbs15oBevLzXsUuTtA1",
        base_url: str = "https://api.tts.timepay.ai",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the timepay.ai TTS service.

        Args:
            api_key: timepay.ai AI API subscription key.
            aiohttp_session: Shared aiohttp session for making requests.
            voice_id: Speaker voice ID.
            base_url: timepay.ai AI API base URL. Defaults to "https://api.tts.timepay.ai".
            sample_rate: Audio sample rate in Hz (8000, 16000, 24000). If None, uses default.
            params: Additional voice and preprocessing parameters. If None, uses defaults.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or TimepayHttpTTSService.InputParams()

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session

        # Build base settings common to all models
        self._settings = {
            "language": (
                self.language_to_service_language(params.language) if params.language else "en"
            ),
            "speed": params.speed,
            "add_wav_header": params.add_wav_header
        }

        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as timepay.ai service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to timepay.ai language format.

        Args:
            language: The language to convert.

        Returns:
            The timepay.ai AI-specific language code, or None if not supported.
        """
        return language_to_timepayai_language(language)

    async def start(self, frame: StartFrame):
        """Start the timepay.ai TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using timepay.ai API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            payload = {
                "text": text,
                "language": self._settings["language"],
                "voice_id": self._voice_id,
                "speed": self._settings["speed"],
                "sample_rate": self.sample_rate,
                "add_wav_header": self._settings["add_wav_header"]
            }

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            url = f"{self._base_url}/api/v1/get_speech"

            yield TTSStartedFrame()

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"timepay.ai API error: {error_text}")
                    return

                await self.start_tts_usage_metrics(text)

                # Process the streaming response
                CHUNK_SIZE = 512

                yield TTSStartedFrame()
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()