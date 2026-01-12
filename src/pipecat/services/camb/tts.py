#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Camb.ai MARS text-to-speech service implementation.

This module provides TTS functionality using Camb.ai's MARS model family,
offering high-quality text-to-speech synthesis with HTTP streaming support.

Features:
    - MARS models: mars-flash, mars-pro, mars-instruct
    - 140+ languages supported
    - Real-time HTTP streaming
    - 24kHz audio output
    - Voice customization (instructions for mars-instruct)
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional

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


# Default configuration
DEFAULT_VOICE_ID = 147320
DEFAULT_LANGUAGE = "en-us"
DEFAULT_MODEL = "mars-flash"  # Faster inference
DEFAULT_BASE_URL = "https://client.camb.ai/apis"
DEFAULT_SAMPLE_RATE = 24000  # 24kHz
DEFAULT_TIMEOUT = 60.0  # Seconds (minimum recommended by Camb.ai)
MIN_TEXT_LENGTH = 3
MAX_TEXT_LENGTH = 3000


def language_to_camb_language(language: Language) -> Optional[str]:
    """Convert a Pipecat Language enum to Camb.ai language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Camb.ai language code (BCP-47 format), or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.EN: "en-us",
        Language.EN_US: "en-us",
        Language.EN_GB: "en-gb",
        Language.EN_AU: "en-au",
        Language.ES: "es-es",
        Language.ES_ES: "es-es",
        Language.ES_MX: "es-mx",
        Language.FR: "fr-fr",
        Language.FR_FR: "fr-fr",
        Language.FR_CA: "fr-ca",
        Language.DE: "de-de",
        Language.DE_DE: "de-de",
        Language.IT: "it-it",
        Language.PT: "pt-pt",
        Language.PT_BR: "pt-br",
        Language.PT_PT: "pt-pt",
        Language.NL: "nl-nl",
        Language.PL: "pl-pl",
        Language.RU: "ru-ru",
        Language.JA: "ja-jp",
        Language.KO: "ko-kr",
        Language.ZH: "zh-cn",
        Language.ZH_CN: "zh-cn",
        Language.ZH_TW: "zh-tw",
        Language.AR: "ar-sa",
        Language.HI: "hi-in",
        Language.TR: "tr-tr",
        Language.VI: "vi-vn",
        Language.TH: "th-th",
        Language.ID: "id-id",
        Language.MS: "ms-my",
        Language.SV: "sv-se",
        Language.DA: "da-dk",
        Language.NO: "no-no",
        Language.FI: "fi-fi",
        Language.CS: "cs-cz",
        Language.EL: "el-gr",
        Language.HE: "he-il",
        Language.HU: "hu-hu",
        Language.RO: "ro-ro",
        Language.SK: "sk-sk",
        Language.UK: "uk-ua",
        Language.BG: "bg-bg",
        Language.HR: "hr-hr",
        Language.SR: "sr-rs",
        Language.SL: "sl-si",
        Language.CA: "ca-es",
        Language.EU: "eu-es",
        Language.GL: "gl-es",
        Language.AF: "af-za",
        Language.SW: "sw-ke",
        Language.TA: "ta-in",
        Language.TE: "te-in",
        Language.BN: "bn-in",
        Language.MR: "mr-in",
        Language.GU: "gu-in",
        Language.KN: "kn-in",
        Language.ML: "ml-in",
        Language.PA: "pa-in",
        Language.UR: "ur-pk",
        Language.FA: "fa-ir",
        Language.TL: "tl-ph",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


class CambTTSService(TTSService):
    """Camb.ai MARS HTTP-based text-to-speech service.

    Converts text to speech using Camb.ai's MARS TTS models with support for
    multiple languages. Provides custom instructions support for the mars-instruct model.

    Example::

        tts = CambTTSService(
            api_key="your-api-key",
            voice_id=147320,
            model="mars-flash",
            aiohttp_session=session,
            params=CambTTSService.InputParams(
                language=Language.EN
            )
        )

        # For mars-instruct with custom instructions:
        tts_instruct = CambTTSService(
            api_key="your-api-key",
            voice_id=147320,
            model="mars-instruct",
            aiohttp_session=session,
            params=CambTTSService.InputParams(
                language=Language.EN,
                user_instructions="Speak with excitement and energy"
            )
        )
    """

    class InputParams(BaseModel):
        """Input parameters for Camb.ai TTS configuration.

        Parameters:
            language: Language for synthesis (BCP-47 format). Defaults to English.
            user_instructions: Custom instructions for mars-instruct model only.
                Ignored for other models. Max 1000 characters.
        """

        language: Optional[Language] = Language.EN
        user_instructions: Optional[str] = Field(
            default=None,
            max_length=1000,
            description="Custom instructions for mars-instruct model only. "
            "Use to control tone, style, or pronunciation. Max 1000 characters.",
        )

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: int = DEFAULT_VOICE_ID,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Camb.ai TTS service.

        Args:
            api_key: Camb.ai API key for authentication.
            aiohttp_session: Shared aiohttp session for making HTTP requests.
            voice_id: Voice ID to use. Defaults to 147320.
            model: TTS model to use. Options: "mars-flash", "mars-pro", "mars-instruct".
                Defaults to "mars-flash" (fastest).
            base_url: Camb.ai API base URL. Defaults to production URL.
            sample_rate: Audio sample rate in Hz. If None, uses Camb.ai default (24000).
            params: Additional voice parameters. If None, uses defaults.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or CambTTSService.InputParams()

        self._api_key = api_key
        self._session = aiohttp_session

        # Remove trailing slash from base URL
        if base_url.endswith("/"):
            logger.warning("Base URL ends with a slash, removing it.")
            base_url = base_url[:-1]

        self._base_url = base_url

        # Build settings
        self._settings = {
            "language": (
                self.language_to_service_language(params.language)
                if params.language
                else DEFAULT_LANGUAGE
            ),
            "user_instructions": params.user_instructions,
        }

        self.set_model_name(model)
        self.set_voice(str(voice_id))
        self._voice_id_int = voice_id

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Camb.ai service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Camb.ai language format.

        Args:
            language: The language to convert.

        Returns:
            The Camb.ai-specific language code, or None if not supported.
        """
        return language_to_camb_language(language)

    async def start(self, frame: StartFrame):
        """Start the Camb.ai TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        # Use Camb.ai's native sample rate if not specified
        if not self._init_sample_rate:
            self._sample_rate = DEFAULT_SAMPLE_RATE
        self._settings["sample_rate"] = self._sample_rate

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings dynamically.

        Args:
            settings: Dictionary of settings to update.
        """
        await super()._update_settings(settings)

        for key, value in settings.items():
            if key in self._settings:
                if key == "language" and isinstance(value, Language):
                    self._settings[key] = language_to_camb_language(value)
                else:
                    self._settings[key] = value
                logger.debug(f"Updated Camb.ai TTS setting {key} to: {value}")
            elif key == "voice_id":
                self._voice_id_int = int(value)
                self.set_voice(str(value))

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Camb.ai's TTS API.

        Args:
            text: The text to synthesize into speech (3-3000 characters).

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Validate text length
        if len(text) < MIN_TEXT_LENGTH:
            logger.warning(f"Text too short for Camb.ai TTS (min {MIN_TEXT_LENGTH} chars): {text}")
            yield TTSStoppedFrame()
            return

        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(
                f"Text too long for Camb.ai TTS (max {MAX_TEXT_LENGTH} chars), truncating"
            )
            text = text[:MAX_TEXT_LENGTH]

        # Build request payload
        payload = {
            "text": text,
            "voice_id": self._voice_id_int,
            "language": self._settings["language"],
            "speech_model": self._model_name,
            "output_configuration": {"format": "pcm_s16le"},
        }

        # Add user instructions if using mars-instruct model
        if self._model_name == "mars-instruct" and self._settings.get("user_instructions"):
            payload["user_instructions"] = self._settings["user_instructions"]

        headers = {
            "x-api-key": self._api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(
                f"{self._base_url}/tts-stream",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = self._format_error_message(response.status, error_text)
                    logger.error(f"{self}: {error_msg}")
                    yield ErrorFrame(error=error_msg)
                    return

                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()

                async for chunk in response.content.iter_chunked(self.chunk_size):
                    if chunk:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                        )

        except aiohttp.ClientError as e:
            error_msg = f"Network error communicating with Camb.ai: {e}"
            logger.error(f"{self}: {error_msg}")
            yield ErrorFrame(error=error_msg)
        except asyncio.TimeoutError:
            error_msg = f"Timeout waiting for Camb.ai TTS response (>{DEFAULT_TIMEOUT}s)"
            logger.error(f"{self}: {error_msg}")
            yield ErrorFrame(error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in Camb.ai TTS: {e}"
            logger.error(f"{self}: {error_msg}")
            yield ErrorFrame(error=error_msg)
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    def _format_error_message(self, status: int, error_text: str) -> str:
        """Format error message based on HTTP status code.

        Args:
            status: HTTP status code.
            error_text: Error response body.

        Returns:
            Formatted, user-friendly error message.
        """
        if status == 401:
            return (
                "Invalid Camb.ai API key. "
                "Set CAMB_API_KEY environment variable with your API key from https://camb.ai"
            )
        elif status == 403:
            return (
                f"Voice ID {self._voice_id_int} is not accessible with your API key. "
                "Use list_voices() to see available voices."
            )
        elif status == 404:
            return (
                f"Invalid voice ID: {self._voice_id_int}. "
                "Use list_voices() to see available voices."
            )
        elif status == 429:
            return "Camb.ai rate limit exceeded. Please wait before making more requests."
        elif status >= 500:
            return f"Camb.ai server error (status {status}): {error_text}"
        else:
            return f"Camb.ai API error (status {status}): {error_text}"

    @staticmethod
    async def list_voices(
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = DEFAULT_BASE_URL,
    ) -> List[Dict[str, Any]]:
        """Fetch available voices from Camb.ai API.

        Args:
            api_key: Camb.ai API key for authentication.
            aiohttp_session: aiohttp ClientSession for making HTTP requests.
            base_url: Camb.ai API base URL.

        Returns:
            List of voice dictionaries with id, name, gender, and language fields.

        Raises:
            Exception: If the API request fails.

        Example::

            async with aiohttp.ClientSession() as session:
                voices = await CambTTSService.list_voices(
                    api_key="your-api-key",
                    aiohttp_session=session,
                )
                for voice in voices:
                    print(f"{voice['id']}: {voice['name']}")
        """
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        headers = {
            "x-api-key": api_key,
            "Accept": "application/json",
        }

        gender_map = {
            0: "Not Specified",
            1: "Male",
            2: "Female",
            9: "Not Applicable",
        }

        async with aiohttp_session.get(
            f"{base_url}/list-voices",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30.0),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to list voices (status {response.status}): {error_text}")

            data = await response.json()
            return [
                {
                    "id": v["id"],
                    "name": v.get("voice_name", "Unknown"),
                    "gender": gender_map.get(v.get("gender"), "Unknown"),
                    "age": v.get("age"),
                    "language": v.get("language"),
                }
                for v in data
            ]
