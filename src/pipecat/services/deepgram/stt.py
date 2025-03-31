#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import AsyncGenerator, Dict, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    from deepgram import (
        AsyncListenWebSocketClient,
        DeepgramClient,
        DeepgramClientOptions,
        ErrorResponse,
        LiveOptions,
        LiveResultResponse,
        LiveTranscriptionEvents,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`.")
    raise Exception(f"Missing module: {e}")


class DeepgramSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        url: str = "",
        sample_rate: Optional[int] = None,
        live_options: Optional[LiveOptions] = None,
        addons: Optional[Dict] = None,
        **kwargs,
    ):
        sample_rate = sample_rate or (live_options.sample_rate if live_options else None)
        super().__init__(sample_rate=sample_rate, **kwargs)

        default_options = LiveOptions(
            encoding="linear16",
            language=Language.EN,
            model="nova-3-general",
            channels=1,
            interim_results=True,
            smart_format=True,
            punctuate=True,
            profanity_filter=True,
            vad_events=False,
        )

        merged_options = default_options
        if live_options:
            merged_options = LiveOptions(**{**default_options.to_dict(), **live_options.to_dict()})

        # deepgram connection requires language to be a string
        if isinstance(merged_options.language, Language) and hasattr(
            merged_options.language, "value"
        ):
            merged_options.language = merged_options.language.value

        self._settings = merged_options.to_dict()
        self._addons = addons

        self._client = DeepgramClient(
            api_key,
            config=DeepgramClientOptions(
                url=url,
                options={"keepalive": "true"},  # verbose=logging.DEBUG
            ),
        )

        if self.vad_enabled:
            self._register_event_handler("on_speech_started")
            self._register_event_handler("on_utterance_end")

    @property
    def vad_enabled(self):
        return self._settings["vad_events"]

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        await super().set_model(model)
        logger.info(f"Switching STT model to: [{model}]")
        self._settings["model"] = model
        await self._disconnect()
        await self._connect()

    async def set_language(self, language: Language):
        logger.info(f"Switching STT language to: [{language}]")
        self._settings["language"] = language
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self._connection.send(audio)
        yield None

    async def _connect(self):
        logger.debug("Connecting to Deepgram")

        self._connection: AsyncListenWebSocketClient = self._client.listen.asyncwebsocket.v("1")

        self._connection.on(
            LiveTranscriptionEvents(LiveTranscriptionEvents.Transcript), self._on_message
        )
        self._connection.on(LiveTranscriptionEvents(LiveTranscriptionEvents.Error), self._on_error)

        if self.vad_enabled:
            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.SpeechStarted),
                self._on_speech_started,
            )
            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.UtteranceEnd),
                self._on_utterance_end,
            )

        if not await self._connection.start(options=self._settings, addons=self._addons):
            logger.error(f"{self}: unable to connect to Deepgram")

    async def _disconnect(self):
        if self._connection.is_connected:
            logger.debug("Disconnecting from Deepgram")
            await self._connection.finish()

    async def start_metrics(self):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def _on_error(self, *args, **kwargs):
        error: ErrorResponse = kwargs["error"]
        logger.warning(f"{self} connection error, will retry: {error}")
        await self.stop_all_metrics()
        # NOTE(aleix): we don't disconnect (i.e. call finish on the connection)
        # because this triggers more errors internally in the Deepgram SDK. So,
        # we just forget about the previous connection and create a new one.
        await self._connect()

    async def _on_speech_started(self, *args, **kwargs):
        await self.start_metrics()
        await self._call_event_handler("on_speech_started", *args, **kwargs)

    async def _on_utterance_end(self, *args, **kwargs):
        await self._call_event_handler("on_utterance_end", *args, **kwargs)

    async def _on_message(self, *args, **kwargs):
        result: LiveResultResponse = kwargs["result"]
        if len(result.channel.alternatives) == 0:
            return
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        language = None
        if result.channel.alternatives[0].languages:
            language = result.channel.alternatives[0].languages[0]
            language = Language(language)
        if len(transcript) > 0:
            await self.stop_ttfb_metrics()
            if is_final:
                await self.push_frame(
                    TranscriptionFrame(transcript, "", time_now_iso8601(), language)
                )
                await self.stop_processing_metrics()
            else:
                await self.push_frame(
                    InterimTranscriptionFrame(transcript, "", time_now_iso8601(), language)
                )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame) and not self.vad_enabled:
            # Start metrics if Deepgram VAD is disabled & pipeline VAD has detected speech
            await self.start_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # https://developers.deepgram.com/docs/finalize
            await self._connection.finalize()
            logger.trace(f"Triggered finalize event on: {frame.name=}, {direction=}")
