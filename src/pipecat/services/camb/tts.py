#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Camb.ai MARS text-to-speech service implementation.

This module provides TTS functionality using Camb.ai's MARS model family,
offering high-quality text-to-speech synthesis with streaming support.

Features:
    - MARS models: mars-flash, mars-pro, mars-instruct
    - 140+ languages supported
    - Real-time streaming via official SDK
    - 48kHz audio output
    - Voice customization (instructions for mars-instruct)
"""

from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional

from camb.client import AsyncCambAI
from camb import StreamTtsOutputConfiguration
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
DEFAULT_SAMPLE_RATE = 48000  # 48kHz
DEFAULT_TIMEOUT = 60.0  # Seconds (minimum recommended by Camb.ai)
MIN_TEXT_LENGTH = 3
MAX_TEXT_LENGTH = 3000

# Gender mapping for voice listing
GENDER_MAP = {0: "Not Specified", 1: "Male", 2: "Female", 9: "Not Applicable"}


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
    """Camb.ai MARS text-to-speech service using the official SDK.

    Converts text to speech using Camb.ai's MARS TTS models with support for
    multiple languages. Provides custom instructions support for the mars-instruct model.

    All models output 48kHz audio.

    Example::

        # Basic usage with defaults
        tts = CambTTSService(api_key="your-api-key")

        # With custom voice and model
        tts = CambTTSService(
            api_key="your-api-key",
            voice_id=12345,
            model="mars-pro",
        )

        # mars-instruct with custom instructions
        tts = CambTTSService(
            api_key="your-api-key",
            model="mars-instruct",
            params=CambTTSService.InputParams(
                user_instructions="Speak with excitement and energy"
            ),
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
        voice_id: int = DEFAULT_VOICE_ID,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Camb.ai TTS service.

        Args:
            api_key: Camb.ai API key for authentication.
            voice_id: Voice ID to use. Defaults to DEFAULT_VOICE_ID.
            model: TTS model to use. Options: "mars-flash", "mars-pro", "mars-instruct".
                Defaults to DEFAULT_MODEL (mars-flash, fastest).
            timeout: Request timeout in seconds. Defaults to DEFAULT_TIMEOUT (60s).
            sample_rate: Audio sample rate in Hz. If None, uses DEFAULT_SAMPLE_RATE (48kHz).
            params: Additional voice parameters. If None, uses defaults.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or CambTTSService.InputParams()

        self._client = AsyncCambAI(api_key=api_key, timeout=timeout)

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
        # Use 48kHz sample rate if not explicitly specified
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

        try:
            await self.start_ttfb_metrics()

            # Build SDK parameters
            tts_kwargs: Dict[str, Any] = {
                "text": text,
                "voice_id": self._voice_id_int,
                "language": self._settings["language"],
                "speech_model": self._model_name,
                "output_configuration": StreamTtsOutputConfiguration(format="pcm_s16le"),
            }

            # Add user instructions if using mars-instruct model
            if self._model_name == "mars-instruct" and self._settings.get("user_instructions"):
                tts_kwargs["user_instructions"] = self._settings["user_instructions"]

            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            # Stream audio chunks from SDK
            async for chunk in self._client.text_to_speech.tts(**tts_kwargs):
                if chunk:
                    await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )

        except Exception as e:
            error_msg = f"Camb.ai TTS error: {e}"
            logger.error(f"{self}: {error_msg}")
            yield ErrorFrame(error=error_msg)
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    @staticmethod
    async def list_voices(api_key: str) -> List[Dict[str, Any]]:
        """Fetch available voices from Camb.ai API.

        Args:
            api_key: Camb.ai API key for authentication.

        Returns:
            List of voice dictionaries with id, name, gender, and language fields.

        Raises:
            Exception: If the API request fails.

        Example::

            voices = await CambTTSService.list_voices(api_key="your-api-key")
            for voice in voices:
                print(f"{voice['id']}: {voice['name']}")
        """
        client = AsyncCambAI(api_key=api_key)
        voice_list = await client.voice_cloning.list_voices()

        voices = []
        for voice in voice_list:
            voice_id = voice.get("id")
            # Skip voices without an ID
            if voice_id is None:
                continue

            gender_int = voice.get("gender")
            gender = GENDER_MAP.get(gender_int) if gender_int is not None else None

            voices.append({
                "id": voice_id,
                "name": voice.get("voice_name", ""),
                "gender": gender,
                "age": voice.get("age"),
                "language": voice.get("language"),
            })

        return voices
