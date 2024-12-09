#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic.main import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import STTService, TTSService
from pipecat.utils.time import time_now_iso8601

try:
    import riva.client

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use nvidia riva TTS or STT, you need to `pip install pipecat-ai[riva]`. Also, set `NVIDIA_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class FastpitchTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[str] = "en-US"

    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice_id: str = "English-US.Female-1",
        sample_rate_hz: int = 24000,
        # nvidia riva calls this 'function-id'
        model: str = "0149dedb-2be8-4195-b9a0-e57e0e14f972",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate_hz, **kwargs)
        self._api_key = api_key

        self.set_model_name("fastpitch-hifigan-tts")
        self.set_voice(voice_id)

        self.voice_id = voice_id
        self.sample_rate_hz = sample_rate_hz
        self.language_code = params.language
        self.quality = None

        metadata = [
            ["function-id", model],
            ["authorization", f"Bearer {api_key}"],
        ]
        auth = riva.client.Auth(None, True, server, metadata)

        self.service = riva.client.SpeechSynthesisService(auth)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        def read_audio_responses():
            try:
                custom_dictionary_input = {}
                responses = self.service.synthesize_online(
                    text,
                    self.voice_id,
                    self.language_code,
                    sample_rate_hz=self.sample_rate_hz,
                    audio_prompt_file=None,
                    quality=20 if self.quality is None else self.quality,
                    custom_dictionary=custom_dictionary_input,
                )
                return responses
            except Exception as e:
                logger.error(f"{self} exception: {e}")
                return []

        logger.debug(f"Generating TTS: [{text}]")

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        responses = await asyncio.to_thread(read_audio_responses)

        for resp in responses:
            await self.stop_ttfb_metrics()

            frame = TTSAudioRawFrame(
                audio=resp.audio,
                sample_rate=self.sample_rate_hz,
                num_channels=1,
            )
            yield frame

        await self.start_tts_usage_metrics(text)
        yield TTSStoppedFrame()


class ParakeetSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        # nvidia calls this 'function-id'
        model: str = "1598d209-5e27-4d3c-8079-4751568b1081",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key

        self.set_model_name("parakeet-ctc-1.1b-asr")

        profanity_filter = False
        automatic_punctuation = False
        no_verbatim_transcripts = False
        language_code = "en-US"
        boosted_lm_words = None
        boosted_lm_score = 4.0
        start_history = -1
        start_threshold = -1.0
        stop_history = -1
        stop_threshold = -1.0
        stop_history_eou = -1
        stop_threshold_eou = -1.0
        custom_configuration = ""
        sample_rate_hz: int = 16000

        metadata = [
            ["function-id", model],
            ["authorization", f"Bearer {api_key}"],
        ]
        auth = riva.client.Auth(None, True, server, metadata)

        self.asr_service = riva.client.ASRService(auth)

        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=language_code,
                model="",
                max_alternatives=1,
                profanity_filter=profanity_filter,
                enable_automatic_punctuation=automatic_punctuation,
                verbatim_transcripts=not no_verbatim_transcripts,
                sample_rate_hertz=sample_rate_hz,
                audio_channel_count=1,
            ),
            interim_results=True,
        )
        self.config = config
        riva.client.add_word_boosting_to_config(config, boosted_lm_words, boosted_lm_score)
        riva.client.add_endpoint_parameters_to_config(
            config,
            start_history,
            start_threshold,
            stop_history,
            stop_history_eou,
            stop_threshold,
            stop_threshold_eou,
        )
        riva.client.add_custom_configuration_to_config(config, custom_configuration)

        # this doesn't work, but something like this perhaps? part 1
        self._queue = asyncio.Queue()

    def can_generate_metrics(self) -> bool:
        return False

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._thread_task = self.get_event_loop().create_task(self._thread_task_handler())
        self._response_task = self.get_event_loop().create_task(self._response_task_handler())
        self._response_queue = asyncio.Queue()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop_tasks()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop_tasks()

    async def _stop_tasks(self):
        self._thread_task.cancel()
        await self._thread_task
        self._response_task.cancel()
        await self._response_task

    def _response_handler(self):
        responses = self.asr_service.streaming_response_generator(
            audio_chunks=self,
            streaming_config=self.config,
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
            pass

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
            try:
                response = await self._response_queue.get()
                await self._handle_response(response)
            except asyncio.CancelledError:
                break

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
