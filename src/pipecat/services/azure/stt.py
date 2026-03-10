#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure Speech-to-Text service implementation for Pipecat.

This module provides speech-to-text functionality using Azure Cognitive Services
Speech SDK for real-time audio transcription.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.azure.common import language_to_azure_language
from pipecat.services.settings import STTSettings, _warn_deprecated_param
from pipecat.services.stt_latency import AZURE_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from azure.cognitiveservices.speech import (
        CancellationReason,
        ResultReason,
        SpeechConfig,
        SpeechRecognizer,
    )
    from azure.cognitiveservices.speech.audio import (
        AudioStreamFormat,
        PushAudioInputStream,
    )
    from azure.cognitiveservices.speech.dialog import AudioConfig
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Azure, you need to `pip install pipecat-ai[azure]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class AzureSTTSettings(STTSettings):
    """Settings for AzureSTTService."""

    pass


class AzureSTTService(STTService):
    """Azure Speech-to-Text service for real-time audio transcription.

    This service uses Azure Cognitive Services Speech SDK to convert speech
    audio into text transcriptions. It supports continuous recognition and
    provides real-time transcription results with timing information.
    """

    Settings = AzureSTTSettings
    _settings: AzureSTTSettings

    def __init__(
        self,
        *,
        api_key: str,
        region: Optional[str] = None,
        language: Optional[Language] = Language.EN_US,
        sample_rate: Optional[int] = None,
        private_endpoint: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        settings: Optional[AzureSTTSettings] = None,
        ttfs_p99_latency: Optional[float] = AZURE_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Azure STT service.

        Args:
            api_key: Azure Cognitive Services subscription key.
            region: Azure region for the Speech service (e.g., 'eastus').
                Required unless ``private_endpoint`` is provided.
            language: Language for speech recognition. Defaults to English (US).

                .. deprecated:: 0.0.105
                    Use ``settings=AzureSTTSettings(language=...)`` instead.

            sample_rate: Audio sample rate in Hz. If None, uses service default.
            private_endpoint: Private endpoint for STT behind firewall.
                See https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-services-private-link?tabs=portal
            endpoint_id: Custom model endpoint id.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to parent STTService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = AzureSTTSettings(
            model=None,
            language=language_to_azure_language(Language.EN_US),
        )

        # 2. Apply direct init arg overrides (deprecated)
        if language is not None and language != Language.EN_US:
            _warn_deprecated_param("language", AzureSTTSettings, "language")
            default_settings.language = language_to_azure_language(language)

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        recognition_language = default_settings.language or language_to_azure_language(
            Language.EN_US
        )

        if not region and not private_endpoint:
            raise ValueError("Either 'region' or 'private_endpoint' must be provided.")

        if private_endpoint:
            if region:
                logger.warning(
                    "Both 'region' and 'private_endpoint' provided; 'region' will be ignored."
                )
            self._speech_config = SpeechConfig(
                subscription=api_key,
                endpoint=private_endpoint,
                speech_recognition_language=recognition_language,
            )
        else:
            self._speech_config = SpeechConfig(
                subscription=api_key,
                region=region,
                speech_recognition_language=recognition_language,
            )

        if endpoint_id:
            self._speech_config.endpoint_id = endpoint_id

        self._audio_stream = None
        self._speech_recognizer = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate performance metrics.

        Returns:
            True as this service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Azure service-specific language code.

        Args:
            language: The language to convert.

        Returns:
            The Azure-specific language identifier, or None if not supported.
        """
        return language_to_azure_language(language)

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if language changed."""
        changed = await super()._update_settings(delta)

        if "language" in changed:
            self._speech_config.speech_recognition_language = (
                self._settings.language or language_to_azure_language(Language.EN_US)
            )
            if self._audio_stream:
                await self._disconnect()
                await self._connect()

        return changed

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data for speech-to-text conversion.

        Feeds audio data to the Azure speech recognizer for processing.
        Recognition results are handled asynchronously through callbacks.

        Args:
            audio: Raw audio bytes to process.

        Yields:
            Frame: Either None for successful processing or ErrorFrame on failure.
        """
        try:
            await self.start_processing_metrics()
            if self._audio_stream:
                self._audio_stream.write(audio)
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

    async def start(self, frame: StartFrame):
        """Start the speech recognition service.

        Args:
            frame: Frame indicating the start of processing.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the speech recognition service.

        Args:
            frame: Frame indicating the end of processing.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the speech recognition service.

        Args:
            frame: Frame indicating cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Initialize the Azure speech recognizer and begin continuous recognition."""
        if self._audio_stream:
            return

        try:
            stream_format = AudioStreamFormat(samples_per_second=self.sample_rate, channels=1)
            self._audio_stream = PushAudioInputStream(stream_format)

            audio_config = AudioConfig(stream=self._audio_stream)

            self._speech_recognizer = SpeechRecognizer(
                speech_config=self._speech_config, audio_config=audio_config
            )
            self._speech_recognizer.recognizing.connect(self._on_handle_recognizing)
            self._speech_recognizer.recognized.connect(self._on_handle_recognized)
            self._speech_recognizer.canceled.connect(self._on_handle_canceled)
            self._speech_recognizer.start_continuous_recognition_async()
        except Exception as e:
            await self.push_error(
                error_msg=f"Uncaught exception during initialization: {e}", exception=e
            )

    async def _disconnect(self):
        """Stop recognition and close audio streams."""
        if self._speech_recognizer:
            self._speech_recognizer.stop_continuous_recognition_async()
            self._speech_recognizer = None

        if self._audio_stream:
            self._audio_stream.close()
            self._audio_stream = None

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        await self.stop_processing_metrics()

    def _on_handle_recognized(self, event):
        if event.result.reason == ResultReason.RecognizedSpeech and len(event.result.text) > 0:
            language = getattr(event.result, "language", None) or self._settings.language
            frame = TranscriptionFrame(
                event.result.text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=event,
            )
            asyncio.run_coroutine_threadsafe(
                self._handle_transcription(event.result.text, True, language), self.get_event_loop()
            )
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())

    def _on_handle_recognizing(self, event):
        if event.result.reason == ResultReason.RecognizingSpeech and len(event.result.text) > 0:
            language = getattr(event.result, "language", None) or self._settings.language
            frame = InterimTranscriptionFrame(
                event.result.text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=event,
            )
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())

    def _on_handle_canceled(self, event):
        details = event.result.cancellation_details
        if details.reason == CancellationReason.Error:
            error_msg = f"Azure STT recognition canceled: {details.reason}"
            if details.error_details:
                error_msg += f" - {details.error_details}"
            asyncio.run_coroutine_threadsafe(
                self.push_error(error_msg=error_msg), self.get_event_loop()
            )
