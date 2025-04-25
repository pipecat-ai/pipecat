#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import riva.client

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use NVIDIA Riva STT, you need to `pip install pipecat-ai[riva]`.")
    raise Exception(f"Missing module: {e}")


class ParakeetSTTService(STTService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN_US

    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        function_id: str = "1598d209-5e27-4d3c-8079-4751568b1081",
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_key = api_key
        self._profanity_filter = False
        self._automatic_punctuation = False
        self._no_verbatim_transcripts = False
        self._language_code = params.language
        self._boosted_lm_words = None
        self._boosted_lm_score = 4.0
        self._start_history = -1
        self._start_threshold = -1.0
        self._stop_history = -1
        self._stop_threshold = -1.0
        self._stop_history_eou = -1
        self._stop_threshold_eou = -1.0
        self._custom_configuration = ""

        self.set_model_name("parakeet-ctc-1.1b-asr")

        metadata = [
            ["function-id", function_id],
            ["authorization", f"Bearer {api_key}"],
        ]
        auth = riva.client.Auth(None, True, server, metadata)

        self._asr_service = riva.client.ASRService(auth)

        self._queue = asyncio.Queue()
        self._config = None
        self._thread_task = None
        self._response_task = None

    def can_generate_metrics(self) -> bool:
        return False

    async def start(self, frame: StartFrame):
        await super().start(frame)

        if self._config:
            return

        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self._language_code,
                model="",
                max_alternatives=1,
                profanity_filter=self._profanity_filter,
                enable_automatic_punctuation=self._automatic_punctuation,
                verbatim_transcripts=not self._no_verbatim_transcripts,
                sample_rate_hertz=self.sample_rate,
                audio_channel_count=1,
            ),
            interim_results=True,
        )

        riva.client.add_word_boosting_to_config(
            config, self._boosted_lm_words, self._boosted_lm_score
        )

        riva.client.add_endpoint_parameters_to_config(
            config,
            self._start_history,
            self._start_threshold,
            self._stop_history,
            self._stop_history_eou,
            self._stop_threshold,
            self._stop_threshold_eou,
        )
        riva.client.add_custom_configuration_to_config(config, self._custom_configuration)

        self._config = config

        if not self._thread_task:
            self._thread_task = self.create_task(self._thread_task_handler())

        if not self._response_task:
            self._response_queue = asyncio.Queue()
            self._response_task = self.create_task(self._response_task_handler())

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop_tasks()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop_tasks()

    async def _stop_tasks(self):
        if self._thread_task:
            await self.cancel_task(self._thread_task)
            self._thread_task = None

        if self._response_task:
            await self.cancel_task(self._response_task)
            self._response_task = None

    def _response_handler(self):
        responses = self._asr_service.streaming_response_generator(
            audio_chunks=self,
            streaming_config=self._config,
        )
        for response in responses:
            if not response.results:
                continue
            asyncio.run_coroutine_threadsafe(
                self._response_queue.put(response), self.get_event_loop()
            )

    async def _thread_task_handler(self):
        try:
            self._thread_running = True
            await asyncio.to_thread(self._response_handler)
        except asyncio.CancelledError:
            self._thread_running = False
            raise

    async def _handle_response(self, response):
        for result in response.results:
            if result and not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            if transcript and len(transcript) > 0:
                await self.stop_ttfb_metrics()
                if result.is_final:
                    await self.stop_processing_metrics()
                    await self.push_frame(
                        TranscriptionFrame(transcript, "", time_now_iso8601(), None)
                    )
                else:
                    await self.push_frame(
                        InterimTranscriptionFrame(transcript, "", time_now_iso8601(), None)
                    )

    async def _response_task_handler(self):
        while True:
            response = await self._response_queue.get()
            await self._handle_response(response)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self._queue.put(audio)
        yield None

    def __next__(self) -> bytes:
        if not self._thread_running:
            raise StopIteration
        future = asyncio.run_coroutine_threadsafe(self._queue.get(), self.get_event_loop())
        return future.result()

    def __iter__(self):
        return self
