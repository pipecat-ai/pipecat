#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ModelsLab text-to-speech service implementation.

This module provides integration with ModelsLab's TTS API
for streaming text-to-speech synthesis.
"""

import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Mapping, Optional, Self

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts


def language_to_modelslab_language(language: Language) -> Optional[str]:
    """Convert a Language enum to ModelsLab language format.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding ModelsLab language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.EN: "en",
        Language.ZH: "zh",
        Language.DE: "de",
        Language.FR: "fr",
        Language.ES: "es",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.PT: "pt",
        Language.RU: "ru",
        Language.AR: "ar",
        Language.HI: "hi",
        Language.NL: "nl",
        Language.PL: "pl",
        Language.TR: "tr",
        Language.SV: "sv",
        Language.DA: "da",
        Language.NO: "no",
        Language.FI: "fi",
        Language.HE: "he",
        Language.ID: "id",
        Language.TH: "th",
        Language.VI: "vi",
        Language.CS: "cs",
        Language.EL: "el",
        Language.HU: "hu",
        Language.RO: "ro",
        Language.UK: "uk",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class ModelsLabTTSSettings(TTSSettings):
    """Settings for ModelsLabTTSService.

    Parameters:
        speed: Speech speed (range: 0.5 to 2.0).
        emotion: Emotional tone (options: "neutral", "happy", "sad", "angry", "fearful").
        voice_id: The voice ID to use for synthesis.
    """

    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    emotion: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    voice_id: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> Self:
        """Construct settings from a plain dict."""
        flat = dict(settings)
        return super().from_mapping(flat)


class ModelsLabTTSService(TTSService):
    """Text-to-speech service using ModelsLab's TTS API.

    Provides streaming text-to-speech synthesis using ModelsLab's HTTP API
    with support for various voice settings, emotions, and audio configurations.
    Supports real-time audio streaming with configurable voice parameters.

    Platform documentation:
    https://docs.modelslab.com
    """

    Settings = ModelsLabTTSSettings
    _settings: Settings

    # Default voice ID
    DEFAULT_VOICE_ID = "kylie"

    # Available voices (from ModelsLab API)
    AVAILABLE_VOICES = [
        "adam",
        "adrian",
        "bella",
        "david",
        "james",
        "kylie",
        "nathan",
        "ryan",
        "scarlett",
        "sophie",
    ]

    class InputParams(BaseModel):
        """Configuration parameters for ModelsLab TTS."""

        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0
        emotion: Optional[str] = None

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://modelslab.com/api/v6/voice",
        voice_id: Optional[str] = None,
        aiohttp_session: aiohttp.ClientSession,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the ModelsLab TTS service.

        Args:
            api_key: ModelsLab API key for authentication.
            base_url: API base URL, defaults to ModelsLab's TTS endpoint.
            voice_id: Voice identifier. Defaults to "kylie".
            aiohttp_session: aiohttp.ClientSession for API communication.
            sample_rate: Output audio sample rate in Hz. If None, uses pipeline default.
            params: Additional configuration parameters.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        # Initialize default settings
        default_settings = self.Settings(
            model="flux",
            voice=self.DEFAULT_VOICE_ID,
            language=None,
            speed=1.0,
            emotion=None,
            voice_id=None,
        )

        # Apply params overrides
        if params is not None:
            if params.speed is not None:
                default_settings.speed = params.speed
            if params.emotion is not None:
                default_settings.emotion = params.emotion

        # Apply settings delta (canonical API, always wins)
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

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as ModelsLab service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to ModelsLab service language format.

        Args:
            language: The language to convert.

        Returns:
            The ModelsLab-specific language code, or None if not supported.
        """
        return language_to_modelslab_language(language)

    async def start(self, frame: StartFrame):
        """Start the ModelsLab TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        logger.debug(f"ModelsLab TTS initialized with sample_rate: {self.sample_rate}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio from text using ModelsLab's API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Get voice ID from settings
        voice_id = self._settings.voice or self.DEFAULT_VOICE_ID

        # Build request payload
        payload = {
            "key": self._api_key,
            "voice_id": voice_id,
            "text": text,
            "output_format": "wav",
            "sample_rate": self.sample_rate,
        }

        # Add optional parameters
        if self._settings.speed is not None:
            payload["speed"] = self._settings.speed

        if self._settings.emotion is not None:
            payload["emotion"] = self._settings.emotion

        # Add language if specified
        service_lang = self.language_to_service_language(self._settings.language or Language.EN)
        if service_lang:
            payload["language"] = service_lang

        try:
            async with self._session.post(
                self._base_url, json=payload
            ) as response:
                if response.status != 200:
                    error_message = f"ModelsLab TTS error: HTTP {response.status}"
                    try:
                        error_data = await response.json()
                        error_message += f" - {error_data.get('message', 'Unknown error')}"
                    except Exception:
                        pass
                    yield ErrorFrame(error=error_message)
                    return

                await self.start_tts_usage_metrics(text)

                # Parse response
                data = await response.json()

                # Check for processing status (async API)
                if data.get("status") == "processing":
                    # Poll for completion
                    request_id = data.get("request_id")
                    if request_id:
                        audio_url = await self._poll_for_result(request_id)
                        if audio_url:
                            # Fetch and yield audio
                            async with self._session.get(audio_url) as audio_response:
                                if audio_response.status == 200:
                                    audio_data = await audio_response.read()
                                    await self.stop_ttfb_metrics()
                                    yield TTSAudioRawFrame(
                                        audio=audio_data,
                                        sample_rate=self.sample_rate,
                                        num_channels=1,
                                        context_id=context_id,
                                    )
                        return
                    else:
                        yield ErrorFrame(error="ModelsLab TTS: No request_id returned")
                        return

                # Direct response with audio
                if "output" in data and "audio" in data["output"]:
                    audio_data = await response.read()
                    await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
                else:
                    yield ErrorFrame(error=f"ModelsLab TTS: Unexpected response format")
                    return

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()

    async def _poll_for_result(self, request_id: str) -> Optional[str]:
        """Poll for async TTS result completion.

        Args:
            request_id: The request ID to poll for.

        Returns:
            The audio URL when ready, or None if still processing.
        """
        poll_url = f"{self._base_url}/status"
        max_attempts = 30
        poll_interval = 1.0

        for attempt in range(max_attempts):
            await self._session.sleep(poll_interval)

            try:
                payload = {"key": self._api_key, "request_id": request_id}
                async with self._session.post(poll_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get("status")

                        if status == "success" and "audio" in data:
                            return data["audio"]
                        elif status == "processing":
                            continue
                        else:
                            logger.warning(f"ModelsLab polling got status: {status}")
                            return None
            except Exception as e:
                logger.error(f"Error polling ModelsLab: {e}")
                return None

        logger.warning("ModelsLab polling timed out")
        return None