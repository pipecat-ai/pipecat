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
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import assemblyai as aai
    from assemblyai import AudioEncoding
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use AssemblyAI, you need to `pip install pipecat-ai[assemblyai]`.")
    raise Exception(f"Missing module: {e}")


class AssemblyAISTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: Optional[int] = None,
        encoding: AudioEncoding = AudioEncoding("pcm_s16le"),
        language=Language.EN,  # Only English is supported for Realtime
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        aai.settings.api_key = api_key
        self._transcriber: Optional[aai.RealtimeTranscriber] = None

        self._settings = {
            "encoding": encoding,
            "language": language,
        }

    async def set_language(self, language: Language):
        logger.info(f"Switching STT language to: [{language}]")
        self._settings["language"] = language

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process an audio chunk for STT transcription.

        This method streams the audio data to AssemblyAI for real-time transcription.
        Transcription results are handled asynchronously via callback functions.

        :param audio: Audio data as bytes
        :yield: None (transcription frames are pushed via self.push_frame in callbacks)
        """
        if self._transcriber:
            await self.start_processing_metrics()
            self._transcriber.stream(audio)
            await self.stop_processing_metrics()
        yield None

    async def _connect(self):
        """Establish a connection to the AssemblyAI real-time transcription service.

        This method sets up the necessary callback functions and initializes the
        AssemblyAI transcriber.
        """

        if self._transcriber:
            return

        def on_open(session_opened: aai.RealtimeSessionOpened):
            """Callback for when the connection to AssemblyAI is opened."""
            logger.info(f"{self}: Connected to AssemblyAI")

        def on_data(transcript: aai.RealtimeTranscript):
            """Callback for handling incoming transcription data.

            This function runs in a separate thread from the main asyncio event loop.
            It creates appropriate transcription frames and schedules them to be
            pushed to the next stage of the pipeline in the main event loop.
            """
            if not transcript.text:
                return

            timestamp = time_now_iso8601()

            if isinstance(transcript, aai.RealtimeFinalTranscript):
                frame = TranscriptionFrame(
                    transcript.text, "", timestamp, self._settings["language"]
                )
            else:
                frame = InterimTranscriptionFrame(
                    transcript.text, "", timestamp, self._settings["language"]
                )

            # Schedule the coroutine to run in the main event loop
            # This is necessary because this callback runs in a different thread
            asyncio.run_coroutine_threadsafe(self.push_frame(frame), self.get_event_loop())

        def on_error(error: aai.RealtimeError):
            """Callback for handling errors from AssemblyAI.

            Like on_data, this runs in a separate thread and schedules error
            handling in the main event loop.
            """
            logger.error(f"{self}: An error occurred: {error}")
            # Schedule the coroutine to run in the main event loop
            asyncio.run_coroutine_threadsafe(
                self.push_frame(ErrorFrame(str(error))), self.get_event_loop()
            )

        def on_close():
            """Callback for when the connection to AssemblyAI is closed."""
            logger.info(f"{self}: Disconnected from AssemblyAI")

        self._transcriber = aai.RealtimeTranscriber(
            sample_rate=self.sample_rate,
            encoding=self._settings["encoding"],
            on_data=on_data,
            on_error=on_error,
            on_open=on_open,
            on_close=on_close,
        )
        self._transcriber.connect()

    async def _disconnect(self):
        """Disconnect from the AssemblyAI service and clean up resources."""
        if self._transcriber:
            self._transcriber.close()
            self._transcriber = None
