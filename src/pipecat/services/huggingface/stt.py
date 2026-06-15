#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Hugging Face speech-to-text service implementation.

This module integrates with Hugging Face Inference Providers for hosted,
usage-based automatic speech recognition.
"""

import base64
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import aiohttp
from loguru import logger

from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, is_given
from pipecat.services.stt_latency import HUGGINGFACE_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


def language_to_huggingface_language(language: Language) -> str:
    """Convert a Language enum to a Hugging Face language code.

    Hugging Face ASR models generally accept base language codes when a language
    hint is used through model-specific generation parameters.

    Args:
        language: The Language enum value to convert.

    Returns:
        The base language code, for example ``"en"`` from ``"en-US"``.
    """
    return resolve_language(language, {}, use_base_code=True)


@dataclass
class HuggingFaceSTTSettings(STTSettings):
    """Settings for HuggingFaceSTTService.

    Parameters:
        return_timestamps: Whether Hugging Face should include chunk timestamps.
        generation_parameters: Optional provider/model generation parameters.
    """

    return_timestamps: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    generation_parameters: dict[str, Any] | None | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class HuggingFaceSTTService(SegmentedSTTService):
    """Speech-to-text service using Hugging Face Inference Providers.

    The service posts VAD-segmented WAV audio to the Hugging Face router and
    emits one finalized transcription per speech segment.
    """

    Settings = HuggingFaceSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://router.huggingface.co/hf-inference",
        aiohttp_session: aiohttp.ClientSession | None = None,
        bill_to: str | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = HUGGINGFACE_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Hugging Face STT service.

        Args:
            api_key: Hugging Face user access token with Inference Providers permission.
            base_url: Hugging Face router base URL.
            aiohttp_session: Optional shared aiohttp ClientSession.
            bill_to: Optional Hugging Face organization/user to bill via X-HF-Bill-To.
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
            settings: Runtime-updatable settings.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        default_settings = self.Settings(
            model="openai/whisper-large-v3-turbo",
            language=None,
            return_timestamps=None,
            generation_parameters=None,
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
        self._session_owner = aiohttp_session is None
        self._bill_to = bill_to

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Hugging Face language format.

        Args:
            language: The language to convert.

        Returns:
            The Hugging Face language code.
        """
        return language_to_huggingface_language(language)

    async def start(self, frame):
        """Start the Hugging Face STT service."""
        await super().start(frame)
        await self._ensure_session()

    async def stop(self, frame: EndFrame):
        """Stop the Hugging Face STT service."""
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Hugging Face STT service."""
        await super().cancel(frame)
        await self._close_session()

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

    async def _close_session(self):
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()
        if self._session_owner:
            self._session = None

    def _build_url(self) -> str:
        model = quote(str(self._settings.model), safe="/")
        return f"{self._base_url}/models/{model}"

    def _build_payload(self, audio_data: bytes) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "inputs": base64.b64encode(audio_data).decode("utf-8"),
        }
        parameters: dict[str, Any] = {}

        return_timestamps = self._settings.return_timestamps
        if is_given(return_timestamps) and return_timestamps is not None:
            parameters["return_timestamps"] = return_timestamps

        generation_parameters = self._settings.generation_parameters
        if is_given(generation_parameters) and generation_parameters:
            parameters["generation_parameters"] = generation_parameters

        if parameters:
            payload["parameters"] = parameters

        return payload

    async def _transcribe_audio(self, audio_data: bytes) -> dict[str, Any]:
        """Upload audio data to Hugging Face and get transcription result.

        Args:
            audio_data: Raw audio bytes in WAV format.

        Returns:
            The transcription result data.

        Raises:
            Exception: If transcription fails or returns an error.
        """
        await self._ensure_session()
        assert self._session is not None

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._bill_to:
            headers["X-HF-Bill-To"] = self._bill_to

        async with self._session.post(
            self._build_url(), json=self._build_payload(audio_data), headers=headers
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Hugging Face transcription error: {error_text}")
                raise Exception(f"Transcription failed with status {response.status}: {error_text}")

            result = await response.json()
            if not isinstance(result, dict):
                raise Exception("Transcription response was not a JSON object")
            return result

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: str | None = None
    ):
        """Handle a transcription result with tracing."""
        await self.stop_processing_metrics()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Transcribe an audio segment using Hugging Face Inference Providers.

        Args:
            audio: Raw audio bytes in WAV format from SegmentedSTTService.

        Yields:
            Frame: TranscriptionFrame containing text, or ErrorFrame on failure.
        """
        try:
            await self.start_processing_metrics()
            result = await self._transcribe_audio(audio)

            text = str(result.get("text") or "").strip()
            if not text:
                await self.stop_processing_metrics()
                return

            language = (
                str(self._settings.language)
                if is_given(self._settings.language) and self._settings.language
                else None
            )

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
            await self.stop_processing_metrics()
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
