#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure Speech-to-Text service implementation for Pipecat.

This module provides speech-to-text functionality using Azure Cognitive Services
Speech SDK for real-time audio transcription.
"""

import asyncio
import json
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.azure.common import language_to_azure_language
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from azure.cognitiveservices.speech import (
        OutputFormat,
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


class AzureSTTService(STTService):
    """Azure Speech-to-Text service for real-time audio transcription.

    This service uses Azure Cognitive Services Speech SDK to convert speech
    audio into text transcriptions. It supports continuous recognition and
    provides real-time transcription results with timing information.
    """

    def __init__(
        self,
        *,
        api_key: str,
        region: str,
        language: Language = Language.EN_US,
        sample_rate: Optional[int] = None,
        endpoint_id: Optional[str] = None,
        enable_confidence: bool = False,
        **kwargs,
    ):
        """Initialize the Azure STT service.

        Args:
            api_key: Azure Cognitive Services subscription key.
            region: Azure region for the Speech service (e.g., 'eastus').
            language: Language for speech recognition. Defaults to English (US).
            sample_rate: Audio sample rate in Hz. If None, uses service default.
            endpoint_id: Custom model endpoint id.
            enable_confidence: If True, extract confidence scores.
            **kwargs: Additional arguments passed to parent STTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._speech_config = SpeechConfig(
            subscription=api_key,
            region=region,
            speech_recognition_language=language_to_azure_language(language),
        )

        if endpoint_id:
            self._speech_config.endpoint_id = endpoint_id

        if enable_confidence:
            self._speech_config.output_format = OutputFormat.Detailed

        self._enable_confidence = enable_confidence
        self._audio_stream = None
        self._speech_recognizer = None
        self._settings = {
            "region": region,
            "language": language_to_azure_language(language),
            "sample_rate": sample_rate,
        }

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate performance metrics.

        Returns:
            True as this service supports metrics generation.
        """
        return True

    def _extract_nbest_from_result(self, result) -> Optional[dict]:
        """Extract NBest data from Azure result JSON.

        Args:
            result: Azure SDK recognition result.

        Returns:
            Dictionary containing NBest data, or None if unavailable.
        """
        if not self._enable_confidence:
            return None
        try:
            data = json.loads(result.json)
            nbest = data.get("NBest", [])
            if nbest and len(nbest) > 0:
                # Return the top result with useful info
                top = nbest[0]
                logger.debug(f"NBest details: {json.dumps(nbest)}")
                return {
                    "confidence": top.get("Confidence"),
                    "lexical": top.get("Lexical"),
                    "itn": top.get("ITN"),
                    "masked_itn": top.get("MaskedITN"),
                    "display": top.get("Display"),
                }
        except Exception as e:
            logger.error(f"Could not extract NBest data: {e}")
        return None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data for speech-to-text conversion.

        Feeds audio data to the Azure speech recognizer for processing.
        Recognition results are handled asynchronously through callbacks.

        Args:
            audio: Raw audio bytes to process.

        Yields:
            None - actual transcription frames are pushed via callbacks.
        """
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        if self._audio_stream:
            self._audio_stream.write(audio)
        yield None

    async def start(self, frame: StartFrame):
        """Start the speech recognition service.

        Initializes the Azure speech recognizer with audio stream configuration
        and begins continuous speech recognition.

        Args:
            frame: Frame indicating the start of processing.
        """
        await super().start(frame)

        if self._audio_stream:
            return

        stream_format = AudioStreamFormat(samples_per_second=self.sample_rate, channels=1)
        self._audio_stream = PushAudioInputStream(stream_format)

        audio_config = AudioConfig(stream=self._audio_stream)

        self._speech_recognizer = SpeechRecognizer(
            speech_config=self._speech_config, audio_config=audio_config
        )
        self._speech_recognizer.recognizing.connect(self._on_handle_recognizing)
        self._speech_recognizer.recognized.connect(self._on_handle_recognized)
        self._speech_recognizer.start_continuous_recognition_async()

    async def stop(self, frame: EndFrame):
        """Stop the speech recognition service.

        Cleanly shuts down the Azure speech recognizer and closes audio streams.

        Args:
            frame: Frame indicating the end of processing.
        """
        await super().stop(frame)

        if self._speech_recognizer:
            self._speech_recognizer.stop_continuous_recognition_async()

        if self._audio_stream:
            self._audio_stream.close()

    async def cancel(self, frame: CancelFrame):
        """Cancel the speech recognition service.

        Immediately stops recognition and closes resources.

        Args:
            frame: Frame indicating cancellation.
        """
        await super().cancel(frame)

        if self._speech_recognizer:
            self._speech_recognizer.stop_continuous_recognition_async()

        if self._audio_stream:
            self._audio_stream.close()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None, **kwargs
    ):
        """Handle a transcription result with tracing."""
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    def _on_handle_recognized(self, event):
        if event.result.reason == ResultReason.RecognizedSpeech and len(event.result.text) > 0:
            language = getattr(event.result, "language", None) or self._settings.get("language")

            # Prepare additional tracing attributes
            trace_kwargs = {}
            if self._enable_confidence:
                nbest_data = self._extract_nbest_from_result(event.result)

                if nbest_data:
                    confidence = nbest_data.get("confidence")
                    if confidence is not None:
                        trace_kwargs["confidence"] = confidence
                        logger.debug(
                            f"Final transcription confidence: {confidence} for text: '{event.result.text}'"
                        )
                    # Serialize NBest data for tracing as a JSON string
                    trace_kwargs["nbest"] = json.dumps(nbest_data)

            frame = TranscriptionFrame(
                event.result.text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=event,
            )
            asyncio.run_coroutine_threadsafe(
                self._handle_transcription(event.result.text, True, language, **trace_kwargs),
                self.get_event_loop(),
            )
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())

    def _on_handle_recognizing(self, event):
        if event.result.reason == ResultReason.RecognizingSpeech and len(event.result.text) > 0:
            language = getattr(event.result, "language", None) or self._settings.get("language")

            # Log confidence for interim transcriptions if enabled
            if self._enable_confidence:
                nbest_data = self._extract_nbest_from_result(event.result)
                if nbest_data:
                    confidence = nbest_data.get("confidence")
                    if confidence is not None:
                        logger.debug(
                            f"Interim transcription confidence: {confidence} for text: '{event.result.text}'"
                        )

            frame = InterimTranscriptionFrame(
                event.result.text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=event,
            )
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())
