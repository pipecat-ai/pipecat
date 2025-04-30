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
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import riva.client

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use NVIDIA Riva STT, you need to `pip install pipecat-ai[riva]`. Also set NVIDIA_API_KEY env var.")
    raise Exception(f"Missing module: {e}")


class RivaSTTService(STTService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN_US

    def __init__(
        self,
        *,
        api_key: str = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        function_id: str = "1598d209-5e27-4d3c-8079-4751568b1081",
        model_name: str = "parakeet-ctc-1.1b-asr",
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

        self.set_model_name(model_name)

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

class RivaOfflineSTTService(SegmentedSTTService):
    """Speech-to-text service using Fal's Wizper API.

    This service uses Fal's Wizper API to perform speech-to-text transcription on audio
    segments. It inherits from SegmentedSTTService to handle audio buffering and speech detection.

    Args:
        api_key: NVIDIA_API_KEY.
        sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
        params: Configuration parameters for Riva.
        **kwargs: Additional arguments passed to SegmentedSTTService.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Fal's Wizper API.

        Attributes:
            language: Language of the audio input. Defaults to English.
            task: Task to perform ('transcribe' or 'translate'). Defaults to 'transcribe'.
            chunk_level: Level of chunking ('segment'). Defaults to 'segment'.
            version: Version of Wizper model to use. Defaults to '3'.
        """

        language: Optional[Language] = Language.EN
        task: str = "transcribe"
        chunk_level: str = "segment"
        version: str = "3"

    def __init__(
        self,
        *,
        api_key: str = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        function_id: str = "ee8dc628-76de-4acc-8595-1836e7e857bd",
        model_name: str = "canary-1b-asr",
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

        self.set_model_name(model_name)

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

        # config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            # encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code=self._language_code,
            # model="",
            max_alternatives=1,
            profanity_filter=self._profanity_filter,
            enable_automatic_punctuation=self._automatic_punctuation,
            verbatim_transcripts=not self._no_verbatim_transcripts,
                # sample_rate_hertz=self.sample_rate,
                # audio_channel_count=1,
                # enable_word_time_offsets=args.word_time_offsets or args.speaker_diarization,??
            # ),
            # interim_results=True,
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



    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe an audio segment

        Args:
            audio: Raw audio bytes in WAV format (already converted by base class).

        Yields:
            Frame: TranscriptionFrame containing the transcribed text.

        Note:
            The audio is already in WAV format from the SegmentedSTTService.
            Only non-empty transcriptions are yielded.
        """
        try:
            response = self._asr_service.offline_recognize(audio, self._config)
            # response = riva.client.print_offline(response=self._asr_service.offline_recognize(audio, self._config))
            print(f"_____stt.py * response: {response}")
            # # Send to Fal directly (audio is already in WAV format from base class)
            # data_uri = fal_client.encode(audio, "audio/x-wav")
            # response = await self._fal_client.run(
            #     "fal-ai/wizper",
            #     arguments={"audio_url": data_uri, **self._settings},
            # )

            if response and "text" in response:
                text = response["text"].strip()
                if text:  # Only yield non-empty text
                    logger.debug(f"Transcription: [{text}]")
                    yield TranscriptionFrame(
                        text, "", time_now_iso8601(), Language(self._settings["language"])
                    )

        except Exception as e:
            logger.error(f"Riva Offline STT error: {e}")
            yield ErrorFrame(f"Riva Offline STT error: {str(e)}")

    def __next__(self) -> bytes:
        if not self._thread_running:
            raise StopIteration
        future = asyncio.run_coroutine_threadsafe(self._queue.get(), self.get_event_loop())
        return future.result()

    def __iter__(self):
        return self

class ParakeetSTTService(RivaSTTService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN_US

    def __init__(
        self,
        *,
        api_key: str = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        function_id: str = "1598d209-5e27-4d3c-8079-4751568b1081",
        model_name: str = "parakeet-ctc-1.1b-asr",
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            server=server,
            function_id=function_id,
            model_name=model_name,
            sample_rate=sample_rate,
            params=params,
            **kwargs,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`ParakeetSTTService` is deprecated, use `RivaSTTService` instead.",
                DeprecationWarning,
            )

