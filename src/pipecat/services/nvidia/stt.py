#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Riva Speech-to-Text service implementations for real-time and batch transcription."""

import asyncio
from concurrent.futures import CancelledError as FuturesCancelledError
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, List, Mapping, Optional

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
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_latency import NVIDIA_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import riva.client

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use NVIDIA Riva STT, you need to `pip install pipecat-ai[nvidia]`.")
    raise Exception(f"Missing module: {e}")


def language_to_nvidia_riva_language(language: Language) -> Optional[str]:
    """Maps Language enum to NVIDIA Riva ASR language codes.

    Source:
    https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-riva-build-table.html?highlight=fr%20fr

    Args:
        language: Language enum value.

    Returns:
        Optional[str]: NVIDIA Riva language code or None if not supported.
    """
    LANGUAGE_MAP = {
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

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


@dataclass
class NvidiaSTTSettings(STTSettings):
    """Settings for the NVIDIA Riva streaming STT service."""

    pass


@dataclass
class NvidiaSegmentedSTTSettings(STTSettings):
    """Settings for the NVIDIA Riva segmented STT service.

    Parameters:
        profanity_filter: Whether to filter profanity from results.
        automatic_punctuation: Whether to add automatic punctuation.
        verbatim_transcripts: Whether to return verbatim transcripts.
        boosted_lm_words: List of words to boost in language model.
        boosted_lm_score: Score boost for specified words.
    """

    profanity_filter: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    automatic_punctuation: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    verbatim_transcripts: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    boosted_lm_words: List[str] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    boosted_lm_score: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class NvidiaSTTService(STTService):
    """Real-time speech-to-text service using NVIDIA Riva streaming ASR.

    Provides real-time transcription capabilities using NVIDIA's Riva ASR models
    through streaming recognition. Supports interim results and continuous audio
    processing for low-latency applications.
    """

    _settings: NvidiaSTTSettings

    class InputParams(BaseModel):
        """Configuration parameters for NVIDIA Riva STT service.

        Parameters:
            language: Target language for transcription. Defaults to EN_US.
        """

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
        params: Optional[InputParams] = None,
        use_ssl: bool = True,
        ttfs_p99_latency: Optional[float] = NVIDIA_TTFS_P99,
        **kwargs,
    ):
        """Initialize the NVIDIA Riva STT service.

        Args:
            api_key: NVIDIA API key for authentication.
            server: NVIDIA Riva server address. Defaults to NVIDIA Cloud Function endpoint.
            model_function_map: Mapping containing 'function_id' and 'model_name' for the ASR model.
            sample_rate: Audio sample rate in Hz. If None, uses pipeline default.
            params: Additional configuration parameters for NVIDIA Riva.
            use_ssl: Whether to use SSL for the NVIDIA Riva server. Defaults to True.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to STTService.
        """
        params = params or NvidiaSTTService.InputParams()

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=NvidiaSTTSettings(
                model=model_function_map.get("model_name"),
                language=params.language,
            ),
            **kwargs,
        )

        self._server = server
        self._api_key = api_key
        self._use_ssl = use_ssl
        self._start_history = -1
        self._start_threshold = -1.0
        self._stop_history = -1
        self._stop_threshold = -1.0
        self._stop_history_eou = -1
        self._stop_threshold_eou = -1.0
        self._custom_configuration = ""
        self._function_id = model_function_map.get("function_id")

        self._asr_service = None
        self._queue = None
        self._config = None
        self._thread_task = None

    def _initialize_client(self):
        metadata = [
            ["function-id", self._function_id],
            ["authorization", f"Bearer {self._api_key}"],
        ]
        auth = riva.client.Auth(None, self._use_ssl, self._server, metadata)

        self._asr_service = riva.client.ASRService(auth)

    def _create_recognition_config(self):
        """Create the NVIDIA Riva ASR recognition configuration."""
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self._settings.language,
                model="",
                max_alternatives=1,
                profanity_filter=False,
                enable_automatic_punctuation=True,
                verbatim_transcripts=True,
                sample_rate_hertz=self.sample_rate,
                audio_channel_count=1,
            ),
            interim_results=True,
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

        return config

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            False - this service does not support metrics generation.
        """
        return False

    async def set_model(self, model: str):
        """Set the ASR model for transcription.

        .. deprecated:: 0.0.104
            Model cannot be changed after initialization for NVIDIA Riva streaming STT.
            Set model and function id in the constructor instead, e.g.::

                NvidiaSTTService(
                    api_key=...,
                    model_function_map={"function_id": "<UUID>", "model_name": "<model_name>"},
                )

        Args:
            model: Model name to set.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "'set_model' is deprecated. Model cannot be changed after initialization"
                " for NVIDIA Riva streaming STT. Set model and function id in the"
                " constructor instead, e.g.:"
                " NvidiaSTTService(api_key=..., model_function_map="
                "{'function_id': '<UUID>', 'model_name': '<model_name>'})",
                DeprecationWarning,
                stacklevel=2,
            )

    async def start(self, frame: StartFrame):
        """Start the NVIDIA Riva STT service and initialize streaming configuration.

        Args:
            frame: StartFrame indicating pipeline start.
        """
        await super().start(frame)
        self._initialize_client()
        self._config = self._create_recognition_config()

        self._queue = asyncio.Queue()

        if not self._thread_task:
            self._thread_task = self.create_task(self._thread_task_handler())

        logger.debug(f"Initialized NvidiaSTTService with model: {self._settings.model}")

    async def stop(self, frame: EndFrame):
        """Stop the NVIDIA Riva STT service and clean up resources.

        Args:
            frame: EndFrame indicating pipeline stop.
        """
        await super().stop(frame)
        await self._stop_tasks()

    async def cancel(self, frame: CancelFrame):
        """Cancel the NVIDIA Riva STT service operation.

        Args:
            frame: CancelFrame indicating operation cancellation.
        """
        await super().cancel(frame)
        await self._stop_tasks()

    async def _stop_tasks(self):
        if self._thread_task:
            await self.cancel_task(self._thread_task)
            self._thread_task = None

    def _response_handler(self):
        responses = self._asr_service.streaming_response_generator(
            audio_chunks=self,
            streaming_config=self._config,
        )
        for response in responses:
            if not response.results:
                continue
            asyncio.run_coroutine_threadsafe(self._handle_response(response), self.get_event_loop())

    async def _thread_task_handler(self):
        try:
            self._thread_running = True
            await asyncio.to_thread(self._response_handler)
        except asyncio.CancelledError:
            self._thread_running = False
            raise

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _handle_response(self, response):
        for result in response.results:
            if result and not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            if transcript and len(transcript) > 0:
                if result.is_final:
                    await self.stop_processing_metrics()
                    await self.push_frame(
                        TranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            self._settings.language,
                            result=result,
                        )
                    )
                    await self._handle_transcription(
                        transcript=transcript,
                        is_final=result.is_final,
                        language=self._settings.language,
                    )
                else:
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            self._settings.language,
                            result=result,
                        )
                    )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data for speech-to-text transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            None - transcription results are pushed to the pipeline via frames.
        """
        await self.start_processing_metrics()
        await self._queue.put(audio)
        yield None

    def __next__(self) -> bytes:
        """Get the next audio chunk for NVIDIA Riva processing.

        Returns:
            Audio bytes from the queue.

        Raises:
            StopIteration: When the thread is no longer running.
        """
        if not self._thread_running:
            raise StopIteration

        try:
            future = asyncio.run_coroutine_threadsafe(self._queue.get(), self.get_event_loop())
            return future.result()
        except FuturesCancelledError:
            raise StopIteration

    def __iter__(self):
        """Return iterator for audio chunk processing.

        Returns:
            Self as iterator.
        """
        return self


class NvidiaSegmentedSTTService(SegmentedSTTService):
    """Speech-to-text service using NVIDIA Riva's offline/batch models.

    By default, his service uses NVIDIA's Riva Canary ASR API to perform speech-to-text
    transcription on audio segments. It inherits from SegmentedSTTService to handle
    audio buffering and speech detection.
    """

    _settings: NvidiaSegmentedSTTSettings

    class InputParams(BaseModel):
        """Configuration parameters for NVIDIA Riva segmented STT service.

        Parameters:
            language: Target language for transcription. Defaults to EN_US.
            profanity_filter: Whether to filter profanity from results.
            automatic_punctuation: Whether to add automatic punctuation.
            verbatim_transcripts: Whether to return verbatim transcripts.
            boosted_lm_words: List of words to boost in language model.
            boosted_lm_score: Score boost for specified words.
        """

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
        params: Optional[InputParams] = None,
        use_ssl: bool = True,
        ttfs_p99_latency: Optional[float] = NVIDIA_TTFS_P99,
        **kwargs,
    ):
        """Initialize the NVIDIA Riva segmented STT service.

        Args:
            api_key: NVIDIA API key for authentication
            server: NVIDIA Riva server address (defaults to NVIDIA Cloud Function endpoint)
            model_function_map: Mapping of model name and its corresponding NVIDIA Cloud Function ID
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate
            params: Additional configuration parameters for NVIDIA Riva
            use_ssl: Whether to use SSL for the NVIDIA Riva server. Defaults to True.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to SegmentedSTTService
        """
        params = params or NvidiaSegmentedSTTService.InputParams()

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=NvidiaSegmentedSTTSettings(
                model=model_function_map.get("model_name"),
                language=self.language_to_service_language(params.language or Language.EN_US)
                or "en-US",
                profanity_filter=params.profanity_filter,
                automatic_punctuation=params.automatic_punctuation,
                verbatim_transcripts=params.verbatim_transcripts,
                boosted_lm_words=params.boosted_lm_words,
                boosted_lm_score=params.boosted_lm_score,
            ),
            **kwargs,
        )

        # Initialize NVIDIA Riva settings
        self._api_key = api_key
        self._server = server
        self._use_ssl = use_ssl
        self._function_id = model_function_map.get("function_id")

        # Voice activity detection thresholds (use NVIDIA Riva defaults)
        self._start_history = -1
        self._start_threshold = -1.0
        self._stop_history = -1
        self._stop_threshold = -1.0
        self._stop_history_eou = -1
        self._stop_threshold_eou = -1.0
        self._custom_configuration = ""

        self._config = None
        self._asr_service = None

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert pipecat Language enum to NVIDIA Riva's language code.

        Args:
            language: Language enum value.

        Returns:
            NVIDIA Riva language code or None if not supported.
        """
        return language_to_nvidia_riva_language(language)

    def _initialize_client(self):
        """Initialize the NVIDIA Riva ASR client with authentication metadata."""
        if self._asr_service is not None:
            return

        # Set up authentication metadata for NVIDIA Cloud Functions
        metadata = [
            ["function-id", self._function_id],
            ["authorization", f"Bearer {self._api_key}"],
        ]

        # Create authenticated client
        auth = riva.client.Auth(None, self._use_ssl, self._server, metadata)
        self._asr_service = riva.client.ASRService(auth)

    def _get_language_code(self) -> str:
        """Get the current NVIDIA Riva language code string."""
        return self._settings.language or "en-US"

    def _create_recognition_config(self):
        """Create the NVIDIA Riva ASR recognition configuration."""
        # Create base configuration
        config = riva.client.RecognitionConfig(
            language_code=self._get_language_code(),
            max_alternatives=1,
            profanity_filter=self._settings.profanity_filter,
            enable_automatic_punctuation=self._settings.automatic_punctuation,
            verbatim_transcripts=self._settings.verbatim_transcripts,
        )

        # Add word boosting if specified
        if self._settings.boosted_lm_words:
            riva.client.add_word_boosting_to_config(
                config, self._settings.boosted_lm_words, self._settings.boosted_lm_score
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
        """Check if this service can generate processing metrics.

        Returns:
            True - this service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Initialize the service when the pipeline starts.

        Args:
            frame: StartFrame indicating pipeline start.
        """
        await super().start(frame)
        self._initialize_client()
        self._config = self._create_recognition_config()
        logger.debug(f"Initialized NvidiaSegmentedSTTService with model: {self._settings.model}")

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and sync internal state.

        Args:
            delta: A :class:`STTSettings` (or ``NvidiaSegmentedSTTSettings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if changed:
            self._config = self._create_recognition_config()

        return changed

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe an audio segment.

        Args:
            audio: Raw audio bytes in WAV format (already converted by base class).

        Yields:
            Frame: TranscriptionFrame containing the transcribed text.
        """
        try:
            assert self._asr_service is not None, "ASR service not initialized"
            assert self._config is not None, "Recognition config not created"

            await self.start_processing_metrics()

            # Process audio with NVIDIA Riva ASR - explicitly request non-future response
            raw_response = self._asr_service.offline_recognize(audio, self._config, future=False)

            await self.stop_processing_metrics()

            # Process the response - handle different possible return types
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
                            text,
                            self._user_id,
                            time_now_iso8601(),
                            self._settings.language,
                        )
                        transcription_found = True

                        await self._handle_transcription(text, True, self._settings.language)

            if not transcription_found:
                logger.debug(f"{self}: No transcription results found in NVIDIA Riva response")
        except AttributeError as ae:
            logger.error(f"{self}: Unexpected response structure from NVIDIA Riva: {ae}")
            yield ErrorFrame(f"{self}: Unexpected NVIDIA Riva response format: {str(ae)}")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
