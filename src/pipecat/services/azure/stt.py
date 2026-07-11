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
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

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
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, assert_given
from pipecat.services.stt_latency import AZURE_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from azure.cognitiveservices.speech import (
        CancellationReason,
        ProfanityOption,
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
    logger.error('In order to use Azure, you need to `uv add "pipecat-ai[azure]"`.')
    raise ImportError(f"Missing module: {e}") from e


AzureProfanity = Literal["raw", "masked", "removed"]
"""How Azure handles profanity in transcripts.

* ``"raw"`` — return the text as recognized, no masking.
* ``"masked"`` — replace profane words with ``****`` (Azure default).
* ``"removed"`` — drop profane words from the output.
"""

_PROFANITY_OPTIONS: dict[AzureProfanity, ProfanityOption] = {
    "raw": ProfanityOption.Raw,
    "masked": ProfanityOption.Masked,
    "removed": ProfanityOption.Removed,
}


@dataclass
class AzureSTTSettings(STTSettings):
    """Settings for AzureSTTService.

    ``model`` and ``language`` are inherited from ``STTSettings`` /
    ``ServiceSettings``.

    Parameters:
        profanity: How Azure handles profanity in transcripts. One of
            ``"raw"``, ``"masked"``, or ``"removed"`` (see ``AzureProfanity``).
            Store-mode default is ``None`` (Azure SDK default = ``"masked"``).
            Use ``"raw"`` for non-English deployments where Azure's profanity
            list is over-eager and masks ordinary words (e.g. Italian names
            containing common substrings), which breaks downstream fuzzy
            matching and LLM reasoning. See `SpeechConfig.set_profanity
            <https://learn.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/azure.cognitiveservices.speech.speechconfig#azure-cognitiveservices-speech-speechconfig-set-profanity>`_.
    """

    profanity: AzureProfanity | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class AzureSTTService(STTService):
    """Azure Speech-to-Text service for real-time audio transcription.

    This service uses Azure Cognitive Services Speech SDK to convert speech
    audio into text transcriptions. It supports continuous recognition and
    provides real-time transcription results with timing information.
    """

    Settings = AzureSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        region: str | None = None,
        language: Language | None = Language.EN_US,
        sample_rate: int | None = None,
        private_endpoint: str | None = None,
        endpoint_id: str | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = AZURE_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Azure STT service.

        Args:
            api_key: Azure Cognitive Services subscription key.
            region: Azure region for the Speech service (e.g., 'eastus').
                Required unless ``private_endpoint`` is provided.
            language: Language for speech recognition. Defaults to English (US).

                .. deprecated:: 0.0.105
                    Use ``settings=AzureSTTService.Settings(language=...)`` instead.
                    Will be removed in 2.0.0.

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
        default_settings = self.Settings(
            model=None,
            language=Language.EN_US,
            profanity=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if language is not None and language != Language.EN_US:
            self._warn_init_param_moved_to_settings("language", "language")
            default_settings.language = language

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

        recognition_language = assert_given(
            default_settings.language
        ) or language_to_azure_language(Language.EN_US)

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

        self._apply_profanity()

        self._audio_stream = None
        self._speech_recognizer = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate performance metrics.

        Returns:
            True as this service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Azure service-specific language code.

        Args:
            language: The language to convert.

        Returns:
            The Azure-specific language identifier, or None if not supported.
        """
        return language_to_azure_language(language)

    def _apply_profanity(self):
        """Apply the current ``profanity`` setting to the speech config.

        A no-op when profanity is ``None`` (keeps the Azure SDK default of
        ``"masked"``).
        """
        # Annotate the local so pyright solves ``assert_given``'s TypeVar to the
        # literal instead of widening it to ``str`` (which wouldn't be a valid
        # ``_PROFANITY_OPTIONS`` key).
        profanity: AzureProfanity | None = assert_given(self._settings.profanity)
        if profanity is not None:
            self._speech_config.set_profanity(_PROFANITY_OPTIONS[profanity])

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if language or profanity changed."""
        changed = await super()._update_settings(delta)

        if "language" in changed:
            self._speech_config.speech_recognition_language = assert_given(
                self._settings.language
            ) or language_to_azure_language(Language.EN_US)

        if "profanity" in changed:
            self._apply_profanity()

        # Both settings are baked into the recognizer at connect time, so a
        # live change only takes effect after a reconnect.
        if ("language" in changed or "profanity" in changed) and self._audio_stream:
            await self._disconnect()
            await self._connect()

        return changed

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
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

    async def cleanup(self):
        """Release resources at pipeline teardown."""
        await super().cleanup()
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
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        await self.stop_processing_metrics()

    def _on_handle_recognized(self, event):
        if event.result.reason == ResultReason.RecognizedSpeech and len(event.result.text) > 0:
            # Technically either source could be a raw string, but Language is
            # a StrEnum so downstream handles either.
            language = cast(
                "Language | None",
                getattr(event.result, "language", None) or assert_given(self._settings.language),
            )
            # Azure's ``RecognizedSpeech`` event is by definition the final
            # recognition for an utterance — mark the frame as such so that
            # downstream turn-stop strategies (``SpeechTimeoutUserTurnStop``
            # and friends) can take their finalized fast-path instead of
            # waiting for VAD events that may never arrive on short replies.
            frame = TranscriptionFrame(
                event.result.text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=event,
                finalized=True,
            )
            asyncio.run_coroutine_threadsafe(
                self._handle_transcription(event.result.text, True, language), self.get_event_loop()
            )
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())

    def _on_handle_recognizing(self, event):
        if event.result.reason == ResultReason.RecognizingSpeech and len(event.result.text) > 0:
            # Technically either source could be a raw string, but Language is
            # a StrEnum so downstream handles either.
            language = cast(
                "Language | None",
                getattr(event.result, "language", None) or assert_given(self._settings.language),
            )
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
