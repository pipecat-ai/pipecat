#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import AsyncGenerator, List, Mapping, Optional

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
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import riva.client

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use NVIDIA Riva STT, you need to `pip install pipecat-ai[riva]`.")
    raise Exception(f"Missing module: {e}")


def language_to_riva_language(language: Language) -> Optional[str]:
    """Maps Language enum to Riva ASR language codes.

    Source:
    https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-riva-build-table.html?highlight=fr%20fr

    Args:
        language: Language enum value.

    Returns:
        Optional[str]: Riva language code or None if not supported.
    """
    language_map = {
        # Arabic
        Language.AR: "ar-AR",
        # English
        Language.EN: "en-US",  # Default to US
        Language.EN_US: "en-US",
        Language.EN_GB: "en-GB",
        # French
        Language.FR: "fr-FR",
        Language.FR_FR: "fr-FR",
        # German
        Language.DE: "de-DE",
        Language.DE_DE: "de-DE",
        # Hindi
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        # Italian
        Language.IT: "it-IT",
        Language.IT_IT: "it-IT",
        # Japanese
        Language.JA: "ja-JP",
        Language.JA_JP: "ja-JP",
        # Korean
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
        # Portuguese
        Language.PT: "pt-BR",  # Default to Brazilian
        Language.PT_BR: "pt-BR",
        # Russian
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        # Spanish
        Language.ES: "es-ES",  # Default to Spain
        Language.ES_ES: "es-ES",
        Language.ES_US: "es-US",  # US Spanish
    }

    return language_map.get(language)


class RivaSTTService(STTService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN_US

    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        model_function_map: Mapping[str, str] = {
            "function_id": "1598d209-5e27-4d3c-8079-4751568b1081",
            "model_name": "parakeet-ctc-1.1b-asr",
        },
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_key = api_key
        self._profanity_filter = False
        self._automatic_punctuation = True
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
        self._function_id = model_function_map.get("function_id")

        self.set_model_name(model_function_map.get("model_name"))

        metadata = [
            ["function-id", self._function_id],
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

    async def set_model(self, model: str):
        logger.warning(f"Cannot set model after initialization. Set model and function id like so:")
        example = {"function_id": "<UUID>", "model_name": "<model_name>"}
        logger.warning(
            f"{self.__class__.__name__}(api_key=<api_key>, model_function_map={example})"
        )

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


class RivaSegmentedSTTService(SegmentedSTTService):
    """Speech-to-text service using NVIDIA Riva's offline/batch models.

    By default, his service uses NVIDIA's Riva Canary ASR API to perform speech-to-text
    transcription on audio segments. It inherits from SegmentedSTTService to handle
    audio buffering and speech detection.

    Args:
        api_key: NVIDIA API key for authentication
        server: Riva server address (defaults to NVIDIA Cloud Function endpoint)
        model_function_map: Mapping of model name and its corresponding NVIDIA Cloud Function ID
        sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate
        params: Additional configuration parameters for Riva
        **kwargs: Additional arguments passed to SegmentedSTTService
    """

    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN_US
        profanity_filter: bool = False
        automatic_punctuation: bool = True
        verbatim_transcripts: bool = False
        boosted_lm_words: Optional[List[str]] = None
        boosted_lm_score: float = 4.0

    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        model_function_map: Mapping[str, str] = {
            "function_id": "ee8dc628-76de-4acc-8595-1836e7e857bd",
            "model_name": "canary-1b-asr",
        },
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Set model name
        self.set_model_name(model_function_map.get("model_name"))

        # Initialize Riva settings
        self._api_key = api_key
        self._server = server
        self._function_id = model_function_map.get("function_id")
        self._model_name = model_function_map.get("model_name")

        # Store the language as a Language enum and as a string
        self._language_enum = params.language or Language.EN_US
        self._language = self.language_to_service_language(self._language_enum) or "en-US"

        # Configure transcription parameters
        self._profanity_filter = params.profanity_filter
        self._automatic_punctuation = params.automatic_punctuation
        self._verbatim_transcripts = params.verbatim_transcripts
        self._boosted_lm_words = params.boosted_lm_words
        self._boosted_lm_score = params.boosted_lm_score

        # Voice activity detection thresholds (use Riva defaults)
        self._start_history = -1
        self._start_threshold = -1.0
        self._stop_history = -1
        self._stop_threshold = -1.0
        self._stop_history_eou = -1
        self._stop_threshold_eou = -1.0
        self._custom_configuration = ""

        # Create Riva client
        self._config = None
        self._asr_service = None
        self._settings = {"language": self._language_enum}

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert pipecat Language enum to Riva's language code."""
        return language_to_riva_language(language)

    def _initialize_client(self):
        """Initialize the Riva ASR client with authentication metadata."""
        if self._asr_service is not None:
            return

        # Set up authentication metadata for NVIDIA Cloud Functions
        metadata = [
            ["function-id", self._function_id],
            ["authorization", f"Bearer {self._api_key}"],
        ]

        # Create authenticated client
        auth = riva.client.Auth(None, True, self._server, metadata)
        self._asr_service = riva.client.ASRService(auth)

        logger.info(f"Initialized RivaSegmentedSTTService with model: {self.model_name}")

    def _create_recognition_config(self):
        """Create the Riva ASR recognition configuration."""
        # Create base configuration
        config = riva.client.RecognitionConfig(
            language_code=self._language,  # Now using the string, not a tuple
            max_alternatives=1,
            profanity_filter=self._profanity_filter,
            enable_automatic_punctuation=self._automatic_punctuation,
            verbatim_transcripts=self._verbatim_transcripts,
        )

        # Add word boosting if specified
        if self._boosted_lm_words:
            riva.client.add_word_boosting_to_config(
                config, self._boosted_lm_words, self._boosted_lm_score
            )

        # Add voice activity detection parameters
        riva.client.add_endpoint_parameters_to_config(
            config,
            self._start_history,
            self._start_threshold,
            self._stop_history,
            self._stop_history_eou,
            self._stop_threshold,
            self._stop_threshold_eou,
        )

        # Add any custom configuration
        if self._custom_configuration:
            riva.client.add_custom_configuration_to_config(config, self._custom_configuration)

        return config

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate processing metrics."""
        return True

    async def set_model(self, model: str):
        logger.warning(f"Cannot set model after initialization. Set model and function id like so:")
        example = {"function_id": "<UUID>", "model_name": "<model_name>"}
        logger.warning(
            f"{self.__class__.__name__}(api_key=<api_key>, model_function_map={example})"
        )

    async def start(self, frame: StartFrame):
        """Initialize the service when the pipeline starts."""
        await super().start(frame)
        self._initialize_client()
        self._config = self._create_recognition_config()

    async def set_language(self, language: Language):
        """Set the language for the STT service."""
        logger.info(f"Switching STT language to: [{language}]")
        self._language_enum = language
        self._language = self.language_to_service_language(language) or "en-US"
        self._settings["language"] = language

        # Update configuration with new language
        if self._config:
            self._config.language_code = self._language

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe an audio segment.

        Args:
            audio: Raw audio bytes in WAV format (already converted by base class).

        Yields:
            Frame: TranscriptionFrame containing the transcribed text.
        """
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            # Make sure the client is initialized
            if self._asr_service is None:
                self._initialize_client()

            # Make sure the config is created
            if self._config is None:
                self._config = self._create_recognition_config()

            # Type assertion to satisfy the IDE
            assert self._asr_service is not None, "ASR service not initialized"
            assert self._config is not None, "Recognition config not created"

            # Process audio with Riva ASR - explicitly request non-future response
            raw_response = self._asr_service.offline_recognize(audio, self._config, future=False)

            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

            # Process the response - handle different possible return types
            try:
                # If it's a future-like object, get the result
                if hasattr(raw_response, "result"):
                    response = raw_response.result()
                else:
                    response = raw_response

                # Process transcription results
                transcription_found = False

                # Now we can safely check results
                # Type hint for the IDE
                results = getattr(response, "results", [])

                for result in results:
                    alternatives = getattr(result, "alternatives", [])
                    if alternatives:
                        text = alternatives[0].transcript.strip()
                        if text:
                            logger.debug(f"Transcription: [{text}]")
                            yield TranscriptionFrame(
                                text, "", time_now_iso8601(), self._language_enum
                            )
                            transcription_found = True

                if not transcription_found:
                    logger.debug("No transcription results found in Riva response")

            except AttributeError as ae:
                logger.error(f"Unexpected response structure from Riva: {ae}")
                yield ErrorFrame(f"Unexpected Riva response format: {str(ae)}")

        except Exception as e:
            logger.exception(f"Riva Canary ASR error: {e}")
            yield ErrorFrame(f"Riva Canary ASR error: {str(e)}")


class ParakeetSTTService(RivaSTTService):
    """Deprecated: Use RivaSTTService instead."""

    def __init__(
        self,
        *,
        api_key: str,
        server: str = "grpc.nvcf.nvidia.com:443",
        model_function_map: Mapping[str, str] = {
            "function_id": "1598d209-5e27-4d3c-8079-4751568b1081",
            "model_name": "parakeet-ctc-1.1b-asr",
        },
        sample_rate: Optional[int] = None,
        params: RivaSTTService.InputParams = RivaSTTService.InputParams(),  # Use parent class's type
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            server=server,
            model_function_map=model_function_map,
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
