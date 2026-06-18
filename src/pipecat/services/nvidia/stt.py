#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#

"""NVIDIA Nemotron Speech-to-Text service implementations for real-time and batch transcription.

Refer to the NVIDIA ASR NIM documentation for usage, customization,
and local deployment steps:
https://docs.nvidia.com/nim/speech/latest/asr/
"""

import asyncio
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import CancelledError as FuturesCancelledError
from dataclasses import dataclass, field
from typing import Any

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
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, assert_given
from pipecat.services.stt_latency import NVIDIA_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.deprecation import deprecated
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import grpc
    import riva.client

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use NVIDIA Nemotron Speech STT, you need to `uv add "pipecat-ai[nvidia]"`.'
    )
    raise ImportError(f"Missing module: {e}") from e


def language_to_nvidia_nemotron_speech_language(language: Language) -> str:
    """Maps Language enum to NVIDIA Nemotron Speech ASR language codes.

    Source:
    https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html#supported-languages-by-model-type

    Args:
        language: Language enum value.

    Returns:
        The NVIDIA Nemotron Speech language code. If ``language`` is not in
        the verified mapping, falls back to the full language code string and
        logs a warning (via ``resolve_language(..., use_base_code=False)``).
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
class _NvidiaBaseSTTSettings(STTSettings):
    """Shared settings for NVIDIA Nemotron Speech STT services.

    Parameters:
        profanity_filter: Whether to filter profanity from results.
        automatic_punctuation: Whether to add automatic punctuation.
        verbatim_transcripts: Whether to return verbatim transcripts.
        boosted_lm_words: List of words to boost in language model.
        boosted_lm_score: Score boost for specified words.
        max_alternatives: Maximum number of recognition alternatives.
        word_time_offsets: Whether to include word-level time offsets.
        speaker_diarization: Whether to enable speaker diarization.
        diarization_max_speakers: Maximum number of speakers for diarization.
    """

    profanity_filter: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    automatic_punctuation: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    verbatim_transcripts: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    boosted_lm_words: list[str] | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    boosted_lm_score: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_alternatives: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    word_time_offsets: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speaker_diarization: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    diarization_max_speakers: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class NvidiaSTTSettings(_NvidiaBaseSTTSettings):
    """Settings for NvidiaSTTService.

    Parameters:
        interim_results: Whether to return interim (partial) results.
    """

    interim_results: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class NvidiaSegmentedSTTSettings(_NvidiaBaseSTTSettings):
    """Settings for NvidiaSegmentedSTTService."""

    pass


class AudioChunkIterator:
    """Per-stream iterator that feeds audio chunks to NVIDIA's gRPC stream.

    The NVIDIA client consumes audio synchronously from a worker thread, while
    Pipecat produces audio asynchronously on the event loop. This iterator
    bridges those two worlds and owns the logic for unblocking and closing a
    single stream cleanly.
    """

    _QUEUE_SENTINEL = object()

    def __init__(self, event_loop: asyncio.AbstractEventLoop):
        """Initialize the iterator for a single streaming session.

        Args:
            event_loop: Event loop used to await queue operations from the
                worker thread.
        """
        self._event_loop = event_loop
        self._queue: asyncio.Queue[bytes | object] = asyncio.Queue()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the iterator has been closed intentionally."""
        return self._closed

    async def put(self, audio: bytes) -> None:
        """Enqueue audio for the active stream.

        Args:
            audio: Raw PCM audio bytes to send to the server.
        """
        if self._closed:
            return
        await self._queue.put(audio)

    async def close(self) -> None:
        """Close the iterator and unblock any pending read."""
        if self._closed:
            return
        self._closed = True
        await self._queue.put(self._QUEUE_SENTINEL)

    def __iter__(self):
        """Return the iterator instance."""
        return self

    def __next__(self) -> bytes:
        """Get the next audio chunk for the active stream.

        Returns:
            Audio bytes from the queue.

        Raises:
            StopIteration: When the iterator has been closed.
        """
        if self._closed:
            raise StopIteration

        try:
            future = asyncio.run_coroutine_threadsafe(self._queue.get(), self._event_loop)
            audio = future.result()
        except FuturesCancelledError:
            raise StopIteration

        if audio is self._QUEUE_SENTINEL:
            self._closed = True
            raise StopIteration

        return audio


class NvidiaSTTService(STTService):
    """Real-time speech-to-text service using NVIDIA Nemotron Speech streaming ASR.

    Provides real-time transcription capabilities using NVIDIA's Nemotron Speech ASR models
    through streaming recognition. Supports interim results and continuous audio
    processing for low-latency applications.
    """

    Settings = NvidiaSTTSettings
    _settings: Settings

    @deprecated(
        "`NvidiaSTTService.InputParams` is deprecated since 0.0.105 and will be removed in 2.0.0. "
        "Use `NvidiaSTTService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Configuration parameters for NVIDIA Nemotron Speech STT service.

        .. deprecated:: 0.0.105
            Use ``settings=NvidiaSTTService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Target language for transcription. Defaults to EN_US.
        """

        language: Language | None = Language.EN_US

    def __init__(
        self,
        *,
        api_key: str | None = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        model_function_map: Mapping[str, str] = {
            "function_id": "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa",
            "model_name": "nemotron-asr-streaming",
        },
        sample_rate: int | None = None,
        params: InputParams | None = None,
        use_ssl: bool = True,
        audio_channel_count: int = 1,
        start_history: int = -1,
        start_threshold: float = -1.0,
        stop_history: int = 320,
        stop_threshold: float = -1.0,
        stop_history_eou: int = -1,
        stop_threshold_eou: float = -1.0,
        custom_configuration: str = "",
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = NVIDIA_TTFS_P99,
        **kwargs,
    ):
        """Initialize the NVIDIA Nemotron Speech STT service.

        Args:
            api_key: NVIDIA API key for authentication. Required when using the
                cloud endpoint. Not needed for local deployments.
            server: NVIDIA Nemotron Speech server address. Defaults to NVIDIA Cloud Function endpoint.
                For local deployments, pass the local address (e.g. ``localhost:50051``).
            model_function_map: Mapping containing 'function_id' and 'model_name' for the ASR model.
            sample_rate: Audio sample rate in Hz. If None, uses pipeline default.
            params: Additional configuration parameters for NVIDIA Nemotron Speech.

                .. deprecated:: 0.0.105
                    Use ``settings=NvidiaSTTService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            use_ssl: Whether to use SSL for the gRPC connection. Defaults to True
                for the NVIDIA cloud endpoint. Set to False for local deployments.
            audio_channel_count: Number of audio channels.
            start_history: VAD start history in frames. Use -1 for Nemotron Speech default.
            start_threshold: VAD start threshold. Use -1.0 for Nemotron Speech default.
            stop_history: VAD stop history in frames. Use -1 for Nemotron Speech default.
            stop_threshold: VAD stop threshold. Use -1.0 for Nemotron Speech default.
            stop_history_eou: End-of-utterance stop history in frames. Use -1 for Nemotron Speech default.
            stop_threshold_eou: End-of-utterance stop threshold. Use -1.0 for Nemotron Speech default.
            custom_configuration: Custom Nemotron Speech configuration string
                (e.g. ``"enable_vad_endpointing:true,neural_vad.onset:0.65"``).
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to STTService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model=model_function_map.get("model_name"),
            language=Language.EN_US,
            profanity_filter=False,
            automatic_punctuation=True,
            verbatim_transcripts=True,
            boosted_lm_words=None,
            boosted_lm_score=4.0,
            max_alternatives=1,
            interim_results=True,
            word_time_offsets=False,
            speaker_diarization=False,
            diarization_max_speakers=0,
        )

        # 2. (no deprecated direct args for this service)

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.language = params.language

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._server = server
        self._api_key = api_key
        self._use_ssl = use_ssl
        self._audio_channel_count = audio_channel_count
        self._start_history = start_history
        self._start_threshold = start_threshold
        self._stop_history = stop_history
        self._stop_threshold = stop_threshold
        self._stop_history_eou = stop_history_eou
        self._stop_threshold_eou = stop_threshold_eou
        self._custom_configuration = custom_configuration
        self._function_id = model_function_map.get("function_id")

        self._asr_service = None
        self._audio_iterator: AudioChunkIterator | None = None
        self._config = None
        self._thread_task = None

    def _initialize_client(self):
        """Initialize the NVIDIA Nemotron Speech ASR client with authentication metadata."""
        metadata = []
        if self._function_id:
            metadata.append(["function-id", self._function_id])
        if self._api_key:
            metadata.append(["authorization", f"Bearer {self._api_key}"])
        auth = riva.client.Auth(None, self._use_ssl, self._server, metadata)

        self._asr_service = riva.client.ASRService(auth)

    def _create_recognition_config(self):
        """Create the NVIDIA Nemotron Speech ASR recognition configuration."""
        s = self._settings
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=s.language,
                model="",
                max_alternatives=s.max_alternatives,
                profanity_filter=s.profanity_filter,
                enable_automatic_punctuation=s.automatic_punctuation,
                verbatim_transcripts=s.verbatim_transcripts,
                sample_rate_hertz=self.sample_rate,
                audio_channel_count=self._audio_channel_count,
                enable_word_time_offsets=s.word_time_offsets,
            ),
            interim_results=s.interim_results,
        )

        boosted_lm_words = assert_given(s.boosted_lm_words)
        if boosted_lm_words:
            riva.client.add_word_boosting_to_config(
                config, boosted_lm_words, assert_given(s.boosted_lm_score)
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

        if self._custom_configuration:
            riva.client.add_custom_configuration_to_config(config, self._custom_configuration)

        speaker_diarization = assert_given(s.speaker_diarization)
        if speaker_diarization:
            riva.client.add_speaker_diarization_to_config(
                config, speaker_diarization, assert_given(s.diarization_max_speakers)
            )

        return config

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True - this service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and sync internal state.

        Args:
            delta: A :class:`STTSettings` (or ``NvidiaSTTService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if changed and self._config is not None:
            self._config = self._create_recognition_config()

        return changed

    @deprecated(
        "`NvidiaSTTService.set_model` is deprecated since 0.0.104 and will be removed in 2.0.0. "
        "No replacement."
    )
    async def set_model(self, model: str):
        """Set the ASR model for transcription.

        .. deprecated:: 0.0.104
            No replacement. Model cannot be changed after initialization for NVIDIA Nemotron
            Speech streaming STT; set model and function id in the constructor instead.
            Will be removed in 2.0.0.

            Example::

                NvidiaSTTService(
                    api_key=...,
                    model_function_map={"function_id": "<UUID>", "model_name": "<model_name>"},
                )

        Args:
            model: Model name to set.
        """

    async def start(self, frame: StartFrame):
        """Start the NVIDIA Nemotron Speech STT service and initialize streaming configuration.

        Args:
            frame: StartFrame indicating pipeline start.
        """
        await super().start(frame)
        self._initialize_client()
        self._config = self._create_recognition_config()
        self._audio_iterator = AudioChunkIterator(self.get_event_loop())

        if not self._thread_task:
            self._thread_task = self.create_task(self._thread_task_handler())

        logger.debug(f"Initialized NvidiaSTTService with model: {self._settings.model}")

    async def stop(self, frame: EndFrame):
        """Stop the NVIDIA Nemotron Speech STT service and clean up resources.

        Args:
            frame: EndFrame indicating pipeline stop.
        """
        await super().stop(frame)
        await self._stop_tasks()

    async def cancel(self, frame: CancelFrame):
        """Cancel the NVIDIA Nemotron Speech STT service operation.

        Args:
            frame: CancelFrame indicating operation cancellation.
        """
        await super().cancel(frame)
        await self._stop_tasks()

    async def _stop_tasks(self, close_iterator: bool = True):
        """Stop the active stream thread and optionally close the shared iterator.

        ``close_iterator=False`` is used during reconnect so we can tear down the
        current gRPC stream but keep buffering audio on the same iterator until
        the replacement stream starts consuming it.
        """
        iterator = self._audio_iterator
        if close_iterator:
            self._audio_iterator = None
        if close_iterator and iterator is not None:
            await iterator.close()

        if self._thread_task:
            await self.cancel_task(self._thread_task)
            self._thread_task = None

    async def _do_reconnect(self):
        iterator = self._audio_iterator
        if iterator is None or iterator.closed:
            return

        await self._stop_tasks(close_iterator=False)
        self._initialize_client()
        self._config = self._create_recognition_config()
        self._thread_task = self.create_task(self._thread_task_handler())

    async def _handle_stream_drop(self, iterator: AudioChunkIterator, reason: str):
        if iterator.closed or iterator is not self._audio_iterator:
            return

        logger.warning(f"{self} stream dropped: {reason}")
        await self._request_reconnect()

    def _response_handler(self, iterator: AudioChunkIterator):
        drop_reason = None
        try:
            responses = self._asr_service.streaming_response_generator(
                audio_chunks=iterator,
                streaming_config=self._config,
            )
            for response in responses:
                if not response.results:
                    continue
                asyncio.run_coroutine_threadsafe(
                    self._handle_response(response), self.get_event_loop()
                )
            drop_reason = "server closed the gRPC stream"
        except grpc.RpcError as e:
            status = e.code().name if hasattr(e, "code") else "UNKNOWN"
            details = e.details() if hasattr(e, "details") else str(e)
            drop_reason = f"gRPC {status}: {details}"
        except Exception as e:
            drop_reason = str(e)
            logger.error(f"{self} unexpected streaming error: {e}")

        if drop_reason:
            asyncio.run_coroutine_threadsafe(
                self._handle_stream_drop(iterator, drop_reason),
                self.get_event_loop(),
            )

    async def _thread_task_handler(self):
        iterator = self._audio_iterator
        if iterator is None:
            return

        try:
            await asyncio.to_thread(self._response_handler, iterator)
        except asyncio.CancelledError:
            raise

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _handle_response(self, response):
        for result in response.results:
            if result and not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            if transcript and len(transcript) > 0:
                language = assert_given(self._settings.language)
                if result.is_final:
                    await self.stop_processing_metrics()
                    logger.debug(f"Transcription: [{transcript}]")
                    await self.push_frame(
                        TranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            language,
                            result=result,
                            finalized=True,
                        )
                    )
                    await self._handle_transcription(
                        transcript=transcript,
                        is_final=result.is_final,
                        language=language,
                    )
                else:
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            language,
                            result=result,
                        )
                    )
                    logger.trace(f"Interim Transcription: [{transcript}]")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Process audio data for speech-to-text transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            None - transcription results are pushed to the pipeline via frames.
        """
        await self.start_processing_metrics()
        iterator = self._audio_iterator
        if iterator is not None and not iterator.closed:
            await iterator.put(audio)
        yield None


class NvidiaSegmentedSTTService(SegmentedSTTService):
    """Speech-to-text service using NVIDIA Nemotron Speech's offline/batch models.

    By default, this service uses NVIDIA's Nemotron Speech Canary ASR API to perform speech-to-text
    transcription on audio segments. It inherits from SegmentedSTTService to handle
    audio buffering and speech detection.
    """

    Settings = NvidiaSegmentedSTTSettings
    _settings: Settings

    @deprecated(
        "`NvidiaSegmentedSTTService.InputParams` is deprecated since 0.0.105 and will be removed "
        "in 2.0.0. Use `NvidiaSegmentedSTTService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Configuration parameters for NVIDIA Nemotron Speech segmented STT service.

        .. deprecated:: 0.0.105
            Use ``settings=NvidiaSegmentedSTTService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Target language for transcription. Defaults to EN_US.
            profanity_filter: Whether to filter profanity from results.
            automatic_punctuation: Whether to add automatic punctuation.
            verbatim_transcripts: Whether to return verbatim transcripts.
            boosted_lm_words: List of words to boost in language model.
            boosted_lm_score: Score boost for specified words.
        """

        language: Language | None = Language.EN_US
        profanity_filter: bool = False
        automatic_punctuation: bool = True
        verbatim_transcripts: bool = False
        boosted_lm_words: list[str] | None = None
        boosted_lm_score: float = 4.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        server: str = "grpc.nvcf.nvidia.com:443",
        model_function_map: Mapping[str, str] = {
            "function_id": "ee8dc628-76de-4acc-8595-1836e7e857bd",
            "model_name": "canary-1b-asr",
        },
        sample_rate: int | None = None,
        params: InputParams | None = None,
        use_ssl: bool = True,
        custom_configuration: str = "",
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = NVIDIA_TTFS_P99,
        **kwargs,
    ):
        """Initialize the NVIDIA Nemotron Speech segmented STT service.

        Args:
            api_key: NVIDIA API key for authentication. Required when using the
                cloud endpoint. Not needed for local deployments.
            server: NVIDIA Nemotron Speech server address. Defaults to NVIDIA Cloud Function endpoint.
                For local deployments, pass the local address (e.g. ``localhost:50051``).
            model_function_map: Mapping of model name and its corresponding NVIDIA Cloud Function ID.
            sample_rate: Audio sample rate in Hz. If not provided, uses the pipeline's rate.
            params: Additional configuration parameters for NVIDIA Nemotron Speech.

                .. deprecated:: 0.0.105
                    Use ``settings=NvidiaSegmentedSTTService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            use_ssl: Whether to use SSL for the gRPC connection. Defaults to True
                for the NVIDIA cloud endpoint. Set to False for local deployments.
            custom_configuration: Custom Nemotron Speech configuration string
                (e.g. ``"enable_vad_endpointing:true,neural_vad.onset:0.65"``).
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model=model_function_map.get("model_name"),
            language=Language.EN_US,
            profanity_filter=False,
            automatic_punctuation=True,
            verbatim_transcripts=False,
            boosted_lm_words=None,
            boosted_lm_score=4.0,
            max_alternatives=1,
            word_time_offsets=False,
        )

        # 2. (no deprecated direct args for this service)

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.language = params.language or Language.EN_US
                default_settings.profanity_filter = params.profanity_filter
                default_settings.automatic_punctuation = params.automatic_punctuation
                default_settings.verbatim_transcripts = params.verbatim_transcripts
                default_settings.boosted_lm_words = params.boosted_lm_words
                default_settings.boosted_lm_score = params.boosted_lm_score

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        # Initialize NVIDIA Nemotron Speech settings
        self._api_key = api_key
        self._server = server
        self._use_ssl = use_ssl
        self._function_id = model_function_map.get("function_id")
        self._custom_configuration = custom_configuration

        self._config = None
        self._asr_service = None

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert pipecat Language enum to NVIDIA Nemotron Speech's language code.

        Args:
            language: Language enum value.

        Returns:
            NVIDIA Nemotron Speech language code or None if not supported.
        """
        return language_to_nvidia_nemotron_speech_language(language)

    def _initialize_client(self):
        """Initialize the NVIDIA Nemotron Speech ASR client with authentication metadata."""
        if self._asr_service is not None:
            return

        # Set up authentication metadata for NVIDIA Cloud Functions
        metadata = []
        if self._function_id:
            metadata.append(["function-id", self._function_id])
        if self._api_key:
            metadata.append(["authorization", f"Bearer {self._api_key}"])

        # Create authenticated client
        auth = riva.client.Auth(None, self._use_ssl, self._server, metadata)
        self._asr_service = riva.client.ASRService(auth)

    def _get_language_code(self) -> str:
        """Get the current NVIDIA Nemotron Speech language code string."""
        return assert_given(self._settings.language) or "en-US"

    def _create_recognition_config(self):
        """Create the NVIDIA Nemotron Speech ASR recognition configuration."""
        # Create base configuration
        s = self._settings
        config = riva.client.RecognitionConfig(
            language_code=self._get_language_code(),
            max_alternatives=s.max_alternatives,
            profanity_filter=s.profanity_filter,
            enable_automatic_punctuation=s.automatic_punctuation,
            verbatim_transcripts=s.verbatim_transcripts,
            enable_word_time_offsets=s.word_time_offsets,
        )

        # Add word boosting if specified
        boosted_lm_words = assert_given(s.boosted_lm_words)
        if boosted_lm_words:
            riva.client.add_word_boosting_to_config(
                config, boosted_lm_words, assert_given(s.boosted_lm_score)
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
            delta: A :class:`STTSettings` (or ``NvidiaSegmentedSTTService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if changed:
            self._config = self._create_recognition_config()

        return changed

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
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

            # Process audio with NVIDIA Nemotron Speech ASR - explicitly request non-future response
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
                        language = assert_given(self._settings.language)
                        yield TranscriptionFrame(
                            text,
                            self._user_id,
                            time_now_iso8601(),
                            language,
                        )
                        transcription_found = True

                        await self._handle_transcription(text, True, language)

            if not transcription_found:
                logger.debug(
                    f"{self}: No transcription results found in NVIDIA Nemotron Speech response"
                )
        except AttributeError as ae:
            logger.error(f"{self}: Unexpected response structure from NVIDIA Nemotron Speech: {ae}")
            yield ErrorFrame(
                error=f"{self}: Unexpected NVIDIA Nemotron Speech response format: {str(ae)}"
            )
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
