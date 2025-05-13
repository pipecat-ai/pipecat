#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
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
    def __init__(
        self,
        *,
        api_key: str,
        region: str,
        language: Language = Language.EN_US,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._speech_config = SpeechConfig(
            subscription=api_key,
            region=region,
            speech_recognition_language=language_to_azure_language(language),
        )

        self._audio_stream = None
        self._speech_recognizer = None
        self._settings = {
            "region": region,
            "language": language_to_azure_language(language),
            "sample_rate": sample_rate,
        }

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        if self._audio_stream:
            self._audio_stream.write(audio)
        yield None

    async def start(self, frame: StartFrame):
        await super().start(frame)

        if self._audio_stream:
            return

        stream_format = AudioStreamFormat(samples_per_second=self.sample_rate, channels=1)
        self._audio_stream = PushAudioInputStream(stream_format)

        audio_config = AudioConfig(stream=self._audio_stream)

        self._speech_recognizer = SpeechRecognizer(
            speech_config=self._speech_config, audio_config=audio_config
        )
        self._speech_recognizer.recognized.connect(self._on_handle_recognized)
        self._speech_recognizer.start_continuous_recognition_async()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)

        if self._speech_recognizer:
            self._speech_recognizer.stop_continuous_recognition_async()

        if self._audio_stream:
            self._audio_stream.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

        if self._speech_recognizer:
            self._speech_recognizer.stop_continuous_recognition_async()

        if self._audio_stream:
            self._audio_stream.close()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    def _on_handle_recognized(self, event):
        if event.result.reason == ResultReason.RecognizedSpeech and len(event.result.text) > 0:
            language = getattr(event.result, "language", None) or self._settings.get("language")
            frame = TranscriptionFrame(event.result.text, "", time_now_iso8601(), language)
            asyncio.run_coroutine_threadsafe(
                self._handle_transcription(event.result.text, True, language), self.get_event_loop()
            )
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())
