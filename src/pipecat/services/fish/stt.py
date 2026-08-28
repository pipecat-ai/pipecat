#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio speech-to-text service implementation.

This module provides integration with Fish Audio's ASR API. The API transcribes
a complete audio file rather than a live stream, so transcription runs per
speech segment: audio is buffered while the user speaks and uploaded once VAD
reports the turn has ended.
"""

import io
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, is_given
from pipecat.services.stt_latency import FISH_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


def language_to_fish_language(language: Language) -> str:
    """Convert a Language enum to a Fish Audio language hint.

    Fish Audio takes a plain ISO 639-1 code, so regional variants are reduced to
    their base language (e.g. ``Language.EN_US`` becomes ``"en"``).

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding ISO 639-1 language code.
    """
    return language.value.split("-")[0].lower()


@dataclass
class FishAudioSTTSettings(STTSettings):
    """Settings for FishAudioSTTService.

    Parameters:
        ignore_timestamps: Whether to skip word-level timestamps. Leaving this
            enabled is faster; the service does not use timestamps.
    """

    ignore_timestamps: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class FishAudioSTTService(SegmentedSTTService):
    """Speech-to-text service using Fish Audio's file-based ASR API.

    Inherits from SegmentedSTTService, which buffers audio between VAD speech
    start and stop events and hands over one complete segment at a time. Each
    segment is uploaded as a WAV file and transcribed in a single request, so a
    transcription only arrives once the speaker has finished a turn — there are
    no interim results. VAD must be enabled in the pipeline.
    """

    Settings = FishAudioSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession | None = None,
        base_url: str = "https://api.fish.audio",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = FISH_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Fish Audio STT service.

        Args:
            api_key: Fish Audio API key for authentication.
            aiohttp_session: Optional aiohttp ClientSession for HTTP requests.
                If not provided, a session is created and managed internally.
            base_url: Base URL for the Fish Audio API.
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
            settings: Runtime-updatable settings.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        # Fish Audio's ASR endpoint exposes no model selection, so the inherited
        # model setting stays unset.
        default_settings = self.Settings(
            model=None,
            language=Language.EN,
            ignore_timestamps=True,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session
        self._owns_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, as Fish Audio STT service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a Fish Audio language code.

        Args:
            language: The language to convert.

        Returns:
            The Fish Audio language code.
        """
        return language_to_fish_language(language)

    async def cleanup(self):
        """Close the internally created HTTP session, if any."""
        await super().cleanup()
        if self._owns_session and self._session:
            await self._session.close()
            self._session = None

    async def _transcribe_audio(self, audio: bytes) -> dict:
        """Upload a WAV segment to Fish Audio and return the transcription result.

        Args:
            audio: Raw audio bytes in WAV format.

        Returns:
            The decoded JSON transcription result.

        Raises:
            Exception: If the API responds with a non-200 status.
        """
        if not self._session:
            self._session = aiohttp.ClientSession()

        data = aiohttp.FormData()
        data.add_field(
            "audio",
            io.BytesIO(audio),
            filename="audio.wav",
            content_type="audio/wav",
        )
        if self._settings.language:
            data.add_field("language", self._settings.language)
        ignore_timestamps = self._settings.ignore_timestamps
        if is_given(ignore_timestamps) and ignore_timestamps is not None:
            data.add_field("ignore_timestamps", str(ignore_timestamps).lower())

        async with self._session.post(
            f"{self._base_url}/v1/asr",
            data=data,
            headers={"Authorization": f"Bearer {self._api_key}"},
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Fish Audio API error ({response.status}): {error_text}")
            return await response.json()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: str | None = None
    ):
        """Handle a transcription result with tracing."""
        await self.stop_processing_metrics()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Transcribe an audio segment using Fish Audio's ASR API.

        Args:
            audio: Raw audio bytes in WAV format (already converted by the base class).

        Yields:
            Frame: TranscriptionFrame containing the transcribed text, or ErrorFrame on failure.
        """
        try:
            await self.start_processing_metrics()

            result = await self._transcribe_audio(audio)

            text = (result.get("text") or "").strip()
            if not text:
                return

            language = result.get("language_code") or self._settings.language
            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")

            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=result,
            )
        except Exception as e:
            logger.error(f"{self} error transcribing audio: {e}")
            yield ErrorFrame(error=f"Fish Audio transcription failed: {e}")
