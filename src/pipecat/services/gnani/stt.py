#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gnani Speech-to-Text service implementation.

This module provides integration with Gnani's multilingual STT API for speech-to-text
transcription using segmented audio processing.
"""

from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import aiohttp
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Gnani STT, you need to `pip install aiohttp`.")
    raise Exception(f"Missing module: {e}")


def language_to_gnani_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Gnani's language code format.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Gnani language code (e.g., 'en-IN', 'hi-IN'), or None if not supported.
    """
    LANGUAGE_MAP = {
        # English (India)
        Language.EN: "en-IN",
        Language.EN_IN: "en-IN",
        # Hindi
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        # Gujarati
        Language.GU: "gu-IN",
        Language.GU_IN: "gu-IN",
        # Tamil
        Language.TA: "ta-IN",
        Language.TA_IN: "ta-IN",
        # Kannada
        Language.KN: "kn-IN",
        Language.KN_IN: "kn-IN",
        # Telugu
        Language.TE: "te-IN",
        Language.TE_IN: "te-IN",
        # Marathi
        Language.MR: "mr-IN",
        Language.MR_IN: "mr-IN",
        # Bengali
        Language.BN: "bn-IN",
        Language.BN_IN: "bn-IN",
        # Malayalam
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
        # Punjabi
        Language.PA: "pa-IN",
        Language.PA_IN: "pa-IN",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


class GnaniSTTService(SegmentedSTTService):
    """Speech-to-text service using Gnani's multilingual STT API.

    This service uses Gnani's REST API to perform speech-to-text transcription on audio
    segments. It supports multiple Indian languages and inherits from SegmentedSTTService
    to handle audio buffering and speech detection.

    Supported languages:

    - English (India) - en-IN
    - Hindi - hi-IN
    - Gujarati - gu-IN
    - Tamil - ta-IN
    - Kannada - kn-IN
    - Telugu - te-IN
    - Marathi - mr-IN
    - Bengali - bn-IN
    - Malayalam - ml-IN
    - Punjabi - pa-IN
    """

    class InputParams(BaseModel):
        """Configuration parameters for Gnani STT service.

        Parameters:
            language: Language of the audio input. Defaults to Hindi (hi-IN).
            api_request_id: Unique request ID for tracking (optional).
            api_user_id: User/organization identifier.
        """

        language: Optional[Language] = Language.HI_IN
        api_request_id: Optional[str] = None
        api_user_id: str = "pipecat-user"

    def __init__(
        self,
        *,
        api_key: str,
        organization_id: str,
        base_url: str = "https://api.vachana.ai/stt/v3",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the GnaniSTTService with API credentials and parameters.

        Args:
            api_key: Gnani API key for authentication (X-API-Key-ID).
            organization_id: Organization identifier (X-Organization-ID).
            base_url: Base URL for the Gnani STT API. Defaults to production endpoint.
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
            params: Configuration parameters for the STT service.
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or GnaniSTTService.InputParams()

        self._api_key = api_key
        self._organization_id = organization_id
        self._base_url = base_url
        self._api_user_id = params.api_user_id
        self._api_request_id = params.api_request_id

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "hi-IN",
            "api_user_id": self._api_user_id,
        }

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, as Gnani STT service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Gnani's service-specific language code.

        Args:
            language: The language to convert.

        Returns:
            The Gnani-specific language code, or None if not supported.
        """
        return language_to_gnani_language(language)

    async def set_language(self, language: Language):
        """Set the transcription language.

        Args:
            language: The language to use for speech-to-text transcription.
        """
        logger.info(f"Switching STT language to: [{language}]")
        gnani_language = self.language_to_service_language(language)
        if gnani_language:
            self._settings["language"] = gnani_language
        else:
            logger.warning(f"Language {language} not supported by Gnani STT, keeping current language")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing."""
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe an audio segment using Gnani's STT API.

        Args:
            audio: Raw audio bytes in WAV format (already converted by base class).

        Yields:
            Frame: TranscriptionFrame containing the transcribed text, or ErrorFrame on failure.

        Note:
            The audio is already in WAV format from the SegmentedSTTService.
            Only non-empty transcriptions with successful responses are yielded.
        """
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            # Prepare headers
            headers = {
                "X-API-Key-ID": self._api_key,
                "X-Organization-ID": self._organization_id,
                "X-API-User-ID": self._settings["api_user_id"],
            }

            # Add optional request ID if provided
            if self._api_request_id:
                headers["X-API-Request-ID"] = self._api_request_id

            # Prepare form data
            form_data = aiohttp.FormData()
            form_data.add_field("language_code", self._settings["language"])
            form_data.add_field(
                "audio_file",
                audio,
                filename="audio.wav",
                content_type="audio/wav",
            )

            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._base_url,
                    headers=headers,
                    data=form_data,
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Check if the request was successful
                        if result.get("success", False):
                            transcript = result.get("transcript", "").strip()

                            if transcript:
                                await self._handle_transcription(
                                    transcript, True, self._settings["language"]
                                )
                                logger.debug(f"Gnani transcription: [{transcript}]")

                                # Try to convert language code to Language enum
                                try:
                                    # Convert language code like 'hi-IN' to Language enum
                                    language_code = self._settings["language"]
                                    # Try direct conversion first
                                    language_enum = Language(language_code)
                                except (ValueError, KeyError):
                                    # If direct conversion fails, try to find a matching enum
                                    language_enum = None
                                    for lang in Language:
                                        if self.language_to_service_language(lang) == language_code:
                                            language_enum = lang
                                            break

                                yield TranscriptionFrame(
                                    transcript,
                                    self._user_id,
                                    time_now_iso8601(),
                                    language_enum,
                                    result=result,
                                )
                            else:
                                logger.debug("Gnani returned empty transcript")
                        else:
                            error_msg = f"Gnani API returned success=false: {result}"
                            logger.error(error_msg)
                            yield ErrorFrame(error=error_msg)
                    else:
                        error_text = await response.text()
                        error_msg = f"Gnani API error (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        yield ErrorFrame(error=error_msg)

        except aiohttp.ClientError as e:
            error_msg = f"Network error calling Gnani API: {e}"
            logger.error(error_msg)
            yield ErrorFrame(error=error_msg)
        except Exception as e:
            error_msg = f"Unknown error occurred: {e}"
            logger.error(error_msg)
            yield ErrorFrame(error=error_msg)

