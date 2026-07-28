#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Maya Research text-to-speech service implementation.

This module provides integration with Maya Research's Maya 2 TTS API for
streaming text-to-speech synthesis across ten Indian languages.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

# Maya 2 always returns raw PCM, 16-bit signed little-endian, mono, 24kHz.
MAYA_SAMPLE_RATE = 24000


def language_to_maya_language(language: Language) -> str:
    """Convert a Language enum to a Maya language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Maya language code. If ``language`` is not in the
        verified mapping, falls back to the base language code and logs a
        warning (via ``resolve_language``).
    """
    LANGUAGE_MAP = {
        Language.BN: "bn",  # Bengali
        Language.BN_IN: "bn",
        Language.GU: "gu",  # Gujarati
        Language.GU_IN: "gu",
        Language.HI: "hi",  # Hindi
        Language.HI_IN: "hi",
        Language.KN: "kn",  # Kannada
        Language.KN_IN: "kn",
        Language.ML: "ml",  # Malayalam
        Language.ML_IN: "ml",
        Language.MR: "mr",  # Marathi
        Language.MR_IN: "mr",
        Language.OR: "or",  # Odia
        Language.OR_IN: "or",
        Language.PA: "pa",  # Punjabi
        Language.PA_IN: "pa",
        Language.TA: "ta",  # Tamil
        Language.TA_IN: "ta",
        Language.TE: "te",  # Telugu
        Language.TE_IN: "te",
    }

    return resolve_language(language, LANGUAGE_MAP)


@dataclass
class MayaTTSSettings(TTSSettings):
    """Settings for MayaHttpTTSService.

    Parameters:
        region: Regional pronunciation profile. One of ``"IN"`` or ``"US"``.
    """

    region: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class MayaHttpTTSService(TTSService):
    """Text-to-speech service using Maya Research's Maya 2 API.

    Streams raw 24kHz PCM audio from Maya's HTTP endpoint, resampling to the
    pipeline sample rate when needed. Two voices (``Ananya`` and ``Arjun``)
    support all ten available Indian languages.

    Platform documentation: https://www.mayaresearch.ai/llm.txt
    """

    Settings = MayaTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://tts.mayaresearch.ai",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Maya TTS service.

        Args:
            api_key: Maya API key for authentication.
            aiohttp_session: aiohttp ClientSession for API communication.
            base_url: API base URL, defaults to Maya's hosted endpoint.
            sample_rate: Output audio sample rate in Hz. If None, uses the
                pipeline default. Maya synthesizes at 24000 Hz; other rates
                are resampled.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model=None,  # Maya 2 exposes a single model; no model parameter.
            voice="Ananya",
            language=None,
            region="IN",
        )

        # 2. (No deprecated init args.)

        # 3. (No params object to apply.)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # Language may arrive as a Language enum; store the service string.
        language = default_settings.language
        if isinstance(language, Language):
            default_settings.language = self.language_to_service_language(language)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._session = aiohttp_session
        self._url = f"{base_url.rstrip('/')}/v1/tts"

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Maya service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a Maya language code.

        Args:
            language: The language to convert.

        Returns:
            The Maya-specific language code, or None if not supported.
        """
        return language_to_maya_language(language)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio from text using Maya's streaming HTTP API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "voice": assert_given(self._settings.voice),
            "text": text,
        }
        language = assert_given(self._settings.language)
        if language:
            payload["language"] = language
        region = assert_given(self._settings.region)
        if region:
            payload["region"] = region

        try:
            async with self._session.post(self._url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    yield ErrorFrame(
                        error=f"{self} error getting audio (status: {response.status}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    in_sample_rate=MAYA_SAMPLE_RATE,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}", exception=e)
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
