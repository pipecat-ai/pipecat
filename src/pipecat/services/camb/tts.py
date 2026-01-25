#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Camb.ai MARS text-to-speech service implementation.

This module provides TTS functionality using Camb.ai's MARS model family,
offering high-quality text-to-speech synthesis with streaming support.

Features:
    - MARS models: mars-flash (fast), mars-pro (high quality)
    - 140+ languages supported
    - Real-time streaming via official SDK
    - Model-specific sample rates: mars-pro (48kHz), mars-flash (22.05kHz)
"""

from typing import Any, AsyncGenerator, Dict, Optional

from camb import StreamTtsOutputConfiguration
from camb.client import AsyncCambAI
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

# Model-specific sample rates
MODEL_SAMPLE_RATES: Dict[str, int] = {
    "mars-flash": 22050,  # 22.05kHz
    "mars-pro": 48000,  # 48kHz
    "mars-instruct": 22050,  # 22.05kHz
}


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


def _get_aligned_audio(buffer: bytes) -> tuple[bytes, bytes]:
    """Split buffer into aligned audio (2-byte samples) and remainder.

    Args:
        buffer: Raw audio bytes to align.

    Returns:
        Tuple of (aligned audio bytes, remaining bytes).
    """
    aligned_size = (len(buffer) // 2) * 2
    return buffer[:aligned_size], buffer[aligned_size:]


class CambTTSService(TTSService):
    """Camb.ai MARS text-to-speech service using the official SDK.

    Converts text to speech using Camb.ai's MARS TTS models with support for
    multiple languages.

    Models:
        - mars-flash: Fast inference, 22.05kHz output (default)
        - mars-pro: High quality, 48kHz output

    Example::

        # Basic usage with mars-flash (fast)
        tts = CambTTSService(api_key="your-api-key", model="mars-flash")

        # High quality with mars-pro
        tts = CambTTSService(
            api_key="your-api-key",
            voice_id=12345,
            model="mars-pro",
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
        voice_id: int = 147320,
        model: str = "mars-flash",
        timeout: float = 60.0,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Camb.ai TTS service.

        Args:
            api_key: Camb.ai API key for authentication.
            voice_id: Voice ID to use. Defaults to 147320.
            model: TTS model to use. Options: "mars-flash" (fast), "mars-pro" (high quality).
                Defaults to "mars-flash".
            timeout: Request timeout in seconds. Defaults to 60.0 (minimum recommended
                by Camb.ai).
            sample_rate: Audio sample rate in Hz. If None, uses model-specific default.
            params: Additional voice parameters. If None, uses defaults.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._timeout = timeout

        params = params or CambTTSService.InputParams()

        # Warn if sample rate doesn't match model's supported rate
        if sample_rate and sample_rate != MODEL_SAMPLE_RATES.get(model):
            logger.warning(
                f"Camb.ai's {model} model only supports {MODEL_SAMPLE_RATES.get(model)}Hz "
                f"sample rate. Current rate of {sample_rate}Hz may cause issues."
            )

        # Build settings
        self._settings = {
            "language": (
                self.language_to_service_language(params.language) if params.language else "en-us"
            ),
            "user_instructions": params.user_instructions,
        }

        self.set_model_name(model)
        self.set_voice(str(voice_id))
        self._voice_id = voice_id

        self._client = None

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

        self._client = AsyncCambAI(api_key=self._api_key, timeout=self._timeout)

        # Use model-specific sample rate if not explicitly specified
        if not self._init_sample_rate:
            self._sample_rate = MODEL_SAMPLE_RATES.get(self.model_name, 22050)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Camb.ai's TTS API.

        Args:
            text: The text to synthesize into speech (max 3000 characters).

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Validate text length
        if len(text) > 3000:
            logger.warning("Text too long for Camb.ai TTS (max 3000 chars), truncating")
            text = text[:3000]

        try:
            await self.start_ttfb_metrics()

            # Build SDK parameters
            tts_kwargs: Dict[str, Any] = {
                "text": text,
                "voice_id": self._voice_id,
                "language": self._settings["language"],
                "speech_model": self.model_name,
                "output_configuration": StreamTtsOutputConfiguration(format="pcm_s16le"),
            }

            # Add user instructions if using mars-instruct model
            if self._model_name == "mars-instruct" and self._settings.get("user_instructions"):
                tts_kwargs["user_instructions"] = self._settings["user_instructions"]

            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            assert self._client is not None, "Camb.ai TTS service not initialized"

            # Buffer for aligning chunks to 2-byte boundaries (16-bit PCM)
            audio_buffer = b""

            # Stream audio chunks from SDK
            async for chunk in self._client.text_to_speech.tts(**tts_kwargs):
                if chunk:
                    await self.stop_ttfb_metrics()
                    audio_buffer += chunk

                    # Only yield complete 16-bit samples (2 bytes per sample)
                    aligned_audio, audio_buffer = _get_aligned_audio(audio_buffer)
                    if aligned_audio:
                        yield TTSAudioRawFrame(
                            audio=aligned_audio,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                        )

            # Yield any remaining complete samples
            if len(audio_buffer) >= 2:
                aligned_audio, _ = _get_aligned_audio(audio_buffer)
                if aligned_audio:
                    yield TTSAudioRawFrame(
                        audio=aligned_audio,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )

        except Exception as e:
            yield ErrorFrame(error=f"Camb.ai TTS error: {e}")
        finally:
            yield TTSStoppedFrame()
