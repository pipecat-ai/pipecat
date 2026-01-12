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
    - 24kHz audio output
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
    """Camb.ai MARS text-to-speech service using the official SDK.

    Converts text to speech using Camb.ai's MARS TTS models with support for
    multiple languages. Provides custom instructions support for the mars-instruct model.

    Example::

        # Using API key (creates internal client)
        tts = CambTTSService(
            api_key="your-api-key",
            voice_id=147320,
            model="mars-flash",
            params=CambTTSService.InputParams(
                language=Language.EN
            )
        )

        # Using existing SDK client
        client = AsyncCambAI(api_key="your-api-key")
        tts = CambTTSService(
            client=client,
            voice_id=147320,
            model="mars-flash",
        )

        # For mars-instruct with custom instructions:
        tts_instruct = CambTTSService(
            api_key="your-api-key",
            voice_id=147320,
            model="mars-instruct",
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
        api_key: Optional[str] = None,
        client: Optional[AsyncCambAI] = None,
        voice_id: int = DEFAULT_VOICE_ID,
        model: str = DEFAULT_MODEL,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Camb.ai TTS service.

        Args:
            api_key: Camb.ai API key for authentication. Required if client is not provided.
            client: Existing AsyncCambAI client instance. If provided, api_key is ignored.
            voice_id: Voice ID to use. Defaults to 147320.
            model: TTS model to use. Options: "mars-flash", "mars-pro", "mars-instruct".
                Defaults to "mars-flash" (fastest).
            sample_rate: Audio sample rate in Hz. If None, uses Camb.ai default (24000).
            params: Additional voice parameters. If None, uses defaults.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        if client is None and api_key is None:
            raise ValueError("Either 'api_key' or 'client' must be provided")

        params = params or CambTTSService.InputParams()

        # Use provided client or create one
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = AsyncCambAI(api_key=api_key, timeout=DEFAULT_TIMEOUT)
            self._owns_client = True

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
    async def list_voices(
        api_key: Optional[str] = None,
        client: Optional[AsyncCambAI] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch available voices from Camb.ai API.

        Args:
            api_key: Camb.ai API key for authentication. Required if client is not provided.
            client: Existing AsyncCambAI client instance. If provided, api_key is ignored.

        Returns:
            List of voice dictionaries with id, name, gender, and language fields.

        Raises:
            ValueError: If neither api_key nor client is provided.
            Exception: If the API request fails.

        Example::

            # Using API key
            voices = await CambTTSService.list_voices(api_key="your-api-key")
            for voice in voices:
                print(f"{voice['id']}: {voice['name']}")

            # Using existing client
            client = AsyncCambAI(api_key="your-api-key")
            voices = await CambTTSService.list_voices(client=client)
        """
        if client is None and api_key is None:
            raise ValueError("Either 'api_key' or 'client' must be provided")

        gender_map = {
            0: "Not Specified",
            1: "Male",
            2: "Female",
            9: "Not Applicable",
        }

        # Use provided client or create a temporary one
        if client is not None:
            sdk_client = client
        else:
            sdk_client = AsyncCambAI(api_key=api_key)

        voices = await sdk_client.voice_cloning.list_voices()
        return [
            {
                "id": v.id if hasattr(v, "id") else v.get("id"),
                "name": v.voice_name if hasattr(v, "voice_name") else v.get("voice_name", "Unknown"),
                "gender": gender_map.get(
                    v.gender if hasattr(v, "gender") else v.get("gender"), "Unknown"
                ),
                "age": v.age if hasattr(v, "age") else v.get("age"),
                "language": v.language if hasattr(v, "language") else v.get("language"),
            }
            for v in voices
        ]
