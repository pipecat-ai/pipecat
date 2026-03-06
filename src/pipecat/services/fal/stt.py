#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fal speech-to-text service implementation.

This module provides integration with Fal's Wizper API for speech-to-text
transcription using segmented audio processing.
"""

import base64
import os
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings, _warn_deprecated_param
from pipecat.services.stt_latency import FAL_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


def language_to_fal_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Fal's Wizper language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Fal Wizper language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.AF: "af",
        Language.AM: "am",
        Language.AR: "ar",
        Language.AS: "as",
        Language.AZ: "az",
        Language.BA: "ba",
        Language.BE: "be",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.BO: "bo",
        Language.BR: "br",
        Language.BS: "bs",
        Language.CA: "ca",
        Language.CS: "cs",
        Language.CY: "cy",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.EU: "eu",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FO: "fo",
        Language.FR: "fr",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HA: "ha",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HT: "ht",
        Language.HU: "hu",
        Language.HY: "hy",
        Language.ID: "id",
        Language.IS: "is",
        Language.IT: "it",
        Language.JA: "ja",
        Language.JW: "jw",
        Language.KA: "ka",
        Language.KK: "kk",
        Language.KM: "km",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.LA: "la",
        Language.LB: "lb",
        Language.LN: "ln",
        Language.LO: "lo",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MG: "mg",
        Language.MI: "mi",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MN: "mn",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.MT: "mt",
        Language.MY: "my",
        Language.NE: "ne",
        Language.NL: "nl",
        Language.NN: "nn",
        Language.NO: "no",
        Language.OC: "oc",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PS: "ps",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SA: "sa",
        Language.SD: "sd",
        Language.SI: "si",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SN: "sn",
        Language.SO: "so",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SU: "su",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TG: "tg",
        Language.TH: "th",
        Language.TK: "tk",
        Language.TL: "tl",
        Language.TR: "tr",
        Language.TT: "tt",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.UZ: "uz",
        Language.VI: "vi",
        Language.YI: "yi",
        Language.YO: "yo",
        Language.ZH: "zh",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class FalSTTSettings(STTSettings):
    """Settings for FalSTTService."""

    pass


class FalSTTService(SegmentedSTTService):
    """Speech-to-text service using Fal's Wizper API.

    This service uses Fal's Wizper API to perform speech-to-text transcription on audio
    segments. It inherits from SegmentedSTTService to handle audio buffering and speech detection.
    """

    _settings: FalSTTSettings

    class InputParams(BaseModel):
        """Configuration parameters for Fal's Wizper API.

        .. deprecated:: 0.0.105
            Use ``settings=FalSTTSettings(...)`` instead.

        Parameters:
            language: Language of the audio input. Defaults to English.
            task: Task to perform ('transcribe' or 'translate'). Defaults to 'transcribe'.
            chunk_level: Level of chunking ('segment'). Defaults to 'segment'.
            version: Version of Wizper model to use. Defaults to '3'.
        """

        language: Optional[Language] = Language.EN
        task: str = "transcribe"
        chunk_level: str = "segment"
        version: str = "3"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        task: str = "transcribe",
        chunk_level: str = "segment",
        version: str = "3",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        settings: Optional[FalSTTSettings] = None,
        ttfs_p99_latency: Optional[float] = FAL_TTFS_P99,
        **kwargs,
    ):
        """Initialize the FalSTTService with API key and parameters.

        Args:
            api_key: Fal API key. If not provided, will check FAL_KEY environment variable.
            aiohttp_session: Optional aiohttp ClientSession for HTTP requests.
                If not provided, a session will be created and managed internally.
            task: Task to perform (``"transcribe"`` or ``"translate"``).
                Defaults to ``"transcribe"``.
            chunk_level: Level of chunking (``"segment"``). Defaults to ``"segment"``.
            version: Version of Wizper model to use. Defaults to ``"3"``.
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
            params: Configuration parameters for the Wizper API.

                .. deprecated:: 0.0.105
                    Use ``settings=FalSTTSettings(...)`` for model/language and
                    direct init parameters for task/chunk_level/version instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = FalSTTSettings(
            model=None,
            language=language_to_fal_language(Language.EN),
        )

        # 2. (no deprecated direct args for this service)

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            _warn_deprecated_param("params", FalSTTSettings)
            if not settings:
                if params.language is not None:
                    default_settings.language = language_to_fal_language(params.language)
                if params.task != "transcribe":
                    task = params.task
                if params.chunk_level != "segment":
                    chunk_level = params.chunk_level
                if params.version != "3":
                    version = params.version

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._task = task
        self._chunk_level = chunk_level
        self._version = version

        self._api_key = api_key or os.getenv("FAL_KEY", "")
        if not self._api_key:
            raise ValueError(
                "FAL_KEY must be provided either through api_key parameter or environment variable"
            )

        self._session: aiohttp.ClientSession | None = aiohttp_session
        self._owns_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, as Fal STT service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Fal's service-specific language code.

        Args:
            language: The language to convert.

        Returns:
            The Fal-specific language code, or None if not supported.
        """
        return language_to_fal_language(language)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing."""
        await self.stop_processing_metrics()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribes an audio segment using Fal's Wizper API.

        Args:
            audio: Raw audio bytes in WAV format (already converted by base class).

        Yields:
            Frame: TranscriptionFrame containing the transcribed text, or ErrorFrame on failure.

        Note:
            The audio is already in WAV format from the SegmentedSTTService.
            Only non-empty transcriptions are yielded.
        """
        try:
            await self.start_processing_metrics()

            if not self._session:
                self._session = aiohttp.ClientSession()

            data_uri = f"data:audio/x-wav;base64,{base64.b64encode(audio).decode()}"
            payload: dict = {"audio_url": data_uri}
            if self._settings.language is not None:
                payload["language"] = self._settings.language
            if self._task is not None:
                payload["task"] = self._task
            if self._chunk_level is not None:
                payload["chunk_level"] = self._chunk_level
            if self._version is not None:
                payload["version"] = self._version
            headers = {
                "Authorization": f"Key {self._api_key}",
                "Content-Type": "application/json",
            }

            async with self._session.post(
                "https://fal.run/fal-ai/wizper",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    yield ErrorFrame(error=f"Fal API error ({resp.status}): {error_text}")
                    return
                response = await resp.json()

            if response and "text" in response:
                text = response["text"].strip()
                if text:  # Only yield non-empty text
                    await self._handle_transcription(text, True, self._settings.language)
                    logger.debug(f"Transcription: [{text}]")
                    yield TranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        Language(self._settings.language),
                        result=response,
                    )

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
