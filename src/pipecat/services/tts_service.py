#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base classes for Text-to-speech services."""

import asyncio
import uuid
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregationType,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMAssistantPushAggregationFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.settings import TTSSettings, is_given
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.utils.time import seconds_to_nanoseconds


@dataclass
class TTSContext:
    """Context information for a TTS request.

    Attributes:
        append_to_context: Whether this TTS output should be appended to the
            conversation context after it is spoken.
        push_assistant_aggregation: Whether to push an
            ``LLMAssistantPushAggregationFrame`` after the TTS has finished
            speaking, forcing the assistant aggregator to commit its current
            text buffer to the conversation context.
    """

    append_to_context: bool = True
    push_assistant_aggregation: Optional[bool] = False


class TextAggregationMode(str, Enum):
    """Controls how incoming text is aggregated before TTS synthesis.

    Parameters:
        SENTENCE: Buffer text until sentence boundaries are detected before synthesis.
            Produces more natural speech but adds latency (~200-300ms per sentence).
        TOKEN: Stream text tokens directly to TTS as they arrive.
            Reduces latency but may affect speech quality depending on the TTS provider.
    """

    SENTENCE = "sentence"
    TOKEN = "token"

    def __str__(self):
        return self.value


class TTSService(AIService):
    """Base class for text-to-speech services.

    Provides common functionality for TTS services including text aggregation,
    filtering, audio generation, and frame management. Supports configurable
    sentence aggregation, silence insertion, and frame processing control.

    Event handlers:
        on_connected: Called when connected to the TTS service.
        on_disconnected: Called when disconnected from the TTS service.
        on_connection_error: Called when a connection to the TTS service error occurs.
        on_tts_request: Called before a TTS request is made, with the context ID and text.

    Example::

        @tts.event_handler("on_connected")
        async def on_connected(tts: TTSService):
            logger.debug(f"TTS connected")

        @tts.event_handler("on_disconnected")
        async def on_disconnected(tts: TTSService):
            logger.debug(f"TTS disconnected")

        @tts.event_handler("on_connection_error")
        async def on_connection_error(tts: TTSService, error: str):
            logger.error(f"TTS connection error: {error}")

        @tts.event_handler("on_tts_request")
        async def on_tts_request(tts: TTSService, context_id: str, text: str):
            logger.debug(f"TTS request: {context_id} - {text}")
    """

    _settings: TTSSettings

    def __init__(
        self,
        *,
        text_aggregation_mode: Optional[TextAggregationMode] = None,
        aggregate_sentences: Optional[bool] = None,
        # if True, TTSService will push TextFrames and LLMFullResponseEndFrames,
        # otherwise subclass must do it
        push_text_frames: bool = True,
        # if True, TTSService will push TTSStoppedFrames, otherwise subclass must do it
        push_stop_frames: bool = False,
        # if push_stop_frames is True, wait for this idle period before pushing TTSStoppedFrame
        stop_frame_timeout_s: float = 2.0,
        # if True, TTSService will push silence audio frames after TTSStoppedFrame
        push_silence_after_stop: bool = False,
        # if push_silence_after_stop is True, send this amount of audio silence
        silence_time_s: float = 2.0,
        # if True, we will pause processing frames while we are receiving audio
        pause_frame_processing: bool = False,
        # if True, append a trailing space to text before sending to TTS
        # (helps prevent some TTS services from vocalizing trailing punctuation)
        append_trailing_space: bool = False,
        # TTS output sample rate
        sample_rate: Optional[int] = None,
        # if True, enables word-level timestamp tracking and synchronization
        supports_word_timestamps: bool = False,
        # Text aggregator to aggregate incoming tokens and decide when to push to the TTS.
        text_aggregator: Optional[BaseTextAggregator] = None,
        # Types of text aggregations that should not be spoken.
        skip_aggregator_types: Optional[List[str]] = [],
        # A list of callables to transform text before just before sending it to TTS.
        # Each callable takes the aggregated text and its type, and returns the transformed text.
        # To register, provide a list of tuples of (aggregation_type | '*', transform_function).
        text_transforms: Optional[
            List[
                Tuple[AggregationType | str, Callable[[str, str | AggregationType], Awaitable[str]]]
            ]
        ] = None,
        # Text filter executed after text has been aggregated.
        text_filters: Optional[Sequence[BaseTextFilter]] = None,
        text_filter: Optional[BaseTextFilter] = None,
        # Audio transport destination of the generated frames.
        transport_destination: Optional[str] = None,
        settings: Optional[TTSSettings] = None,
        **kwargs,
    ):
        """Initialize the TTS service.

        Args:
            text_aggregation_mode: How to aggregate incoming text before synthesis.
                TextAggregationMode.SENTENCE (default) buffers until sentence boundaries,
                TextAggregationMode.TOKEN streams tokens directly for lower latency.
            aggregate_sentences: Whether to aggregate text into sentences before synthesis.

                .. deprecated:: 0.0.104
                    Use ``text_aggregation_mode`` instead. Set to ``TextAggregationMode.SENTENCE``
                    to aggregate text into sentences before synthesis, or
                    ``TextAggregationMode.TOKEN`` to stream tokens directly for lower latency.

            push_text_frames: Whether to push TextFrames and LLMFullResponseEndFrames.
            push_stop_frames: Whether to automatically push TTSStoppedFrames.
            stop_frame_timeout_s: Idle time before pushing TTSStoppedFrame when push_stop_frames is True.
            push_silence_after_stop: Whether to push silence audio after TTSStoppedFrame.
            silence_time_s: Duration of silence to push when push_silence_after_stop is True.
            pause_frame_processing: Whether to pause frame processing during audio generation.
            append_trailing_space: Whether to append a trailing space to text before sending to TTS.
                This helps prevent some TTS services from vocalizing trailing punctuation (e.g., "dot").
            sample_rate: Output sample rate for generated audio.
            supports_word_timestamps: Whether this service supports word-level timestamp tracking.
                When True, enables synchronization of audio with spoken words so only spoken words
                are added to the conversation context.
            text_aggregator: Custom text aggregator for processing incoming text.

                .. deprecated:: 0.0.95
                    Use an LLMTextProcessor before the TTSService for custom text aggregation.

            skip_aggregator_types: List of aggregation types that should not be spoken.
            text_transforms: A list of callables to transform text before just before sending it
                to TTS. Each callable takes the aggregated text and its type, and returns the
                transformed text. To register, provide a list of tuples of
                (aggregation_type | '*', transform_function).

            text_filters: Sequence of text filters to apply after aggregation.
            text_filter: Single text filter (deprecated, use text_filters).

                .. deprecated:: 0.0.59
                    Use `text_filters` instead, which allows multiple filters.

            transport_destination: Destination for generated audio frames.
            settings: The runtime-updatable settings for the TTS service.
            **kwargs: Additional arguments passed to the parent AIService.
        """
        super().__init__(
            settings=settings
            # Here in case subclass doesn't implement more specific settings
            # (which hopefully should be rare)
            or TTSSettings(),
            **kwargs,
        )

        # Resolve text_aggregation_mode from the new param or deprecated aggregate_sentences
        if aggregate_sentences is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'aggregate_sentences' is deprecated. "
                    "Use 'text_aggregation_mode=TextAggregationMode.SENTENCE' or "
                    "'text_aggregation_mode=TextAggregationMode.TOKEN' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if text_aggregation_mode is None:
                text_aggregation_mode = (
                    TextAggregationMode.SENTENCE
                    if aggregate_sentences
                    else TextAggregationMode.TOKEN
                )

        if text_aggregation_mode is None:
            text_aggregation_mode = TextAggregationMode.SENTENCE

        self._text_aggregation_mode: TextAggregationMode = text_aggregation_mode
        self._push_text_frames: bool = push_text_frames
        self._push_stop_frames: bool = push_stop_frames
        self._stop_frame_timeout_s: float = stop_frame_timeout_s
        self._push_silence_after_stop: bool = push_silence_after_stop
        self._silence_time_s: float = silence_time_s
        self._pause_frame_processing: bool = pause_frame_processing
        self._append_trailing_space: bool = append_trailing_space
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._text_aggregator: BaseTextAggregator = text_aggregator or SimpleTextAggregator(
            aggregation_type=self._text_aggregation_mode
        )
        if text_aggregator:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'text_aggregator' is deprecated. Use an LLMTextProcessor before the TTSService for custom text aggregation.",
                    DeprecationWarning,
                )

        self._skip_aggregator_types: List[str] = skip_aggregator_types or []
        self._text_transforms: List[
            Tuple[AggregationType | str, Callable[[str, AggregationType | str], Awaitable[str]]]
        ] = text_transforms or []
        # TODO: Deprecate _text_filters when added to LLMTextProcessor
        self._text_filters: Sequence[BaseTextFilter] = text_filters or []
        self._transport_destination: Optional[str] = transport_destination
        if text_filter:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'text_filter' is deprecated, use 'text_filters' instead.",
                    DeprecationWarning,
                )
            self._text_filters = [text_filter]

        self._resampler = create_stream_resampler()

        self._stop_frame_task: Optional[asyncio.Task] = None
        self._stop_frame_queue: asyncio.Queue = asyncio.Queue()

        self._processing_text: bool = False
        self._tts_contexts: Dict[str, TTSContext] = {}
        self._streamed_text: str = ""
        self._text_aggregation_metrics_started: bool = False

        # Word timestamp state (active when supports_word_timestamps=True)
        self._supports_word_timestamps: bool = supports_word_timestamps
        self._initial_word_timestamp: int = -1
        self._initial_word_times: List[Tuple[str, float, Optional[str]]] = []
        self._words_task: Optional[asyncio.Task] = None
        self._llm_response_started: bool = False

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_connection_error")
        self._register_event_handler("on_tts_request")

    @property
    def _is_streaming_tokens(self) -> bool:
        """Whether the service is streaming tokens directly without sentence aggregation."""
        return self._text_aggregation_mode == TextAggregationMode.TOKEN

    async def start_tts_usage_metrics(self, text: str):
        """Record TTS usage metrics.

        When streaming tokens, usage metrics are aggregated and reported at
        flush time instead of per token, so individual calls are skipped.

        Args:
            text: The text being processed by TTS.
        """
        if self._is_streaming_tokens:
            return
        await super().start_tts_usage_metrics(text)

    async def start_text_aggregation_metrics(self):
        """Start text aggregation metrics if not already started.

        Only starts the metric once per LLM response. Skipped when streaming
        tokens since per-token aggregation time is not meaningful.
        """
        if self._is_streaming_tokens or self._text_aggregation_metrics_started:
            return
        self._text_aggregation_metrics_started = True
        await super().start_text_aggregation_metrics()

    async def stop_text_aggregation_metrics(self):
        """Stop text aggregation metrics and reset the started flag."""
        self._text_aggregation_metrics_started = False
        await super().stop_text_aggregation_metrics()

    @property
    def sample_rate(self) -> int:
        """Get the current sample rate for audio output.

        Returns:
            The sample rate in Hz.
        """
        return self._sample_rate

    @property
    def chunk_size(self) -> int:
        """Get the recommended chunk size for audio streaming.

        This property indicates how much audio we download (from TTS services
        that require chunking) before we start pushing the first audio
        frame. This will make sure we download the rest of the audio while audio
        is being played without causing audio glitches (specially at the
        beginning). Of course, this will also depend on how fast the TTS service
        generates bytes.

        Returns:
            The recommended chunk size in bytes.
        """
        CHUNK_SECONDS = 0.5
        return int(self.sample_rate * CHUNK_SECONDS * 2)  # 2 bytes/sample

    async def set_model(self, model: str):
        """Set the TTS model to use.

        .. deprecated:: 0.0.104
            Use ``TTSUpdateSettingsFrame(model=...)`` instead.

        Args:
            model: The name of the TTS model.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "'set_model' is deprecated, use 'TTSUpdateSettingsFrame(model=...)' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        logger.info(f"Switching TTS model to: [{model}]")
        settings_cls = type(self._settings)
        await self._update_settings(settings_cls(model=model))

    async def set_voice(self, voice: str):
        """Set the voice for speech synthesis.

        .. deprecated:: 0.0.104
            Use ``TTSUpdateSettingsFrame(voice=...)`` instead.

        Args:
            voice: The voice identifier or name.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "'set_voice' is deprecated, use 'TTSUpdateSettingsFrame(voice=...)' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        logger.info(f"Switching TTS voice to: [{voice}]")
        settings_cls = type(self._settings)
        await self._update_settings(settings_cls(voice=voice))

    def create_context_id(self) -> str:
        """Generate a unique context ID for a TTS request.

        This method can be overridden by subclasses to provide custom context ID generation.

        Returns:
            A unique string identifier for the TTS context.
        """
        return str(uuid.uuid4())

    # Converts the text to audio.
    @abstractmethod
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Run text-to-speech synthesis on the provided text.

        This method must be implemented by subclasses to provide actual TTS functionality.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        pass

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a language to the service-specific language format.

        Args:
            language: The language to convert.

        Returns:
            The service-specific language identifier, or None if not supported.
        """
        return Language(language)

    def _prepare_text_for_tts(self, text: str) -> str:
        """Prepare text for TTS by applying any transformations required by the TTS service.

        Args:
            text: The text to prepare.

        Returns:
            The prepared text with transformations applied.
        """
        if self._append_trailing_space and not text.endswith(" "):
            return text + " "
        return text

    async def flush_audio(self):
        """Flush any buffered audio data."""
        pass

    async def start(self, frame: StartFrame):
        """Start the TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._sample_rate = self._init_sample_rate or frame.audio_out_sample_rate
        if self._push_stop_frames and not self._stop_frame_task:
            self._stop_frame_task = self.create_task(self._stop_frame_handler())
        if self._supports_word_timestamps:
            self._create_words_task()

    async def stop(self, frame: EndFrame):
        """Stop the TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        if self._stop_frame_task:
            await self.cancel_task(self._stop_frame_task)
            self._stop_frame_task = None
        if self._words_task:
            await self._stop_words_task()

    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        if self._stop_frame_task:
            await self.cancel_task(self._stop_frame_task)
            self._stop_frame_task = None
        if self._words_task:
            await self._stop_words_task()

    def add_text_transformer(
        self,
        transform_function: Callable[[str, AggregationType | str], Awaitable[str]],
        aggregation_type: AggregationType | str = "*",
    ):
        """Transform text for a specific aggregation type.

        Args:
            transform_function: The function to apply for transformation. This function should take
                the text and aggregation type as input and return the transformed text.
                Ex.: async def my_transform(text: str, aggregation_type: str) -> str:
            aggregation_type: The type of aggregation to transform. This value defaults to "*" indicating
                the function should handle all text before sending to TTS.
        """
        self._text_transforms.append((aggregation_type, transform_function))

    def remove_text_transformer(
        self,
        transform_function: Callable[[str, AggregationType | str], Awaitable[str]],
        aggregation_type: AggregationType | str = "*",
    ):
        """Remove a text transformer for a specific aggregation type.

        Args:
            transform_function: The function to remove.
            aggregation_type: The type of aggregation to remove the transformer for.
        """
        self._text_transforms = [
            (agg_type, func)
            for agg_type, func in self._text_transforms
            if not (agg_type == aggregation_type and func == transform_function)
        ]

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a TTS settings delta.

        Translates language to service-specific value before applying.

        Args:
            delta: A TTS settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        # Translate language *before* applying so the stored value is canonical
        if is_given(delta.language) and isinstance(delta.language, Language):
            converted = self.language_to_service_language(delta.language)
            if converted is not None:
                delta.language = converted

        changed = await super()._update_settings(delta)

        return changed

    async def say(self, text: str):
        """Immediately speak the provided text.

        .. deprecated:: 0.0.79
            Push a `TTSSpeakFrame` instead to ensure frame ordering is maintained.

        Args:
            text: The text to speak.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`TTSService.say()` is deprecated. Push a `TTSSpeakFrame` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        await self.queue_frame(TTSSpeakFrame(text))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for text-to-speech conversion.

        Handles TextFrames for synthesis, interruption frames, settings updates,
        and various control frames.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, (TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame))
            and frame.skip_tts
        ):
            await self.push_frame(frame, direction)
        elif isinstance(frame, AggregatedTextFrame):
            await self._push_tts_frames(frame)
        elif (
            isinstance(frame, TextFrame)
            and not isinstance(frame, InterimTranscriptionFrame)
            and not isinstance(frame, TranscriptionFrame)
        ):
            await self.start_text_aggregation_metrics()
            await self._process_text_frame(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruption(frame, direction)
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_response_started = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            # We pause processing incoming frames if the LLM response included
            # text (it might be that it's only a function calling response). We
            # pause to avoid audio overlapping.
            await self._maybe_pause_frame_processing()

            # Flush any remaining text (including text waiting for lookahead)
            remaining = await self._text_aggregator.flush()
            # Stop the aggregation metric (no-op if already stopped on first sentence).
            await self.stop_text_aggregation_metrics()
            if remaining:
                await self._push_tts_frames(AggregatedTextFrame(remaining.text, remaining.type))

            # Log accumulated streamed text and emit aggregated usage metric.
            if self._streamed_text:
                logger.debug(f"{self}: Generating TTS [{self._streamed_text}]")
                await super().start_tts_usage_metrics(self._streamed_text)
                self._streamed_text = ""

            # Reset aggregator state
            self._processing_text = False
            if isinstance(frame, LLMFullResponseEndFrame):
                if self._push_text_frames:
                    await self.push_frame(frame, direction)
            else:
                await self.push_frame(frame, direction)
            # Flush any pending audio so the TTS service closes the current context.
            if self._supports_word_timestamps:
                await self.flush_audio()
        elif isinstance(frame, TTSSpeakFrame):
            # Store if we were processing text or not so we can set it back.
            processing_text = self._processing_text
            # If we are not receiving text from the LLM, we can assume that the SpeakFrame should be automatically added to the context
            push_assistant_aggregation = frame.append_to_context and not self._llm_response_started
            # Assumption: text in TTSSpeakFrame does not include inter-frame spaces
            await self._push_tts_frames(
                AggregatedTextFrame(frame.text, AggregationType.SENTENCE),
                append_tts_text_to_context=frame.append_to_context,
                push_assistant_aggregation=push_assistant_aggregation,
            )
            # We pause processing incoming frames because we are sending data to
            # the TTS. We pause to avoid audio overlapping.
            await self._maybe_pause_frame_processing()
            await self.flush_audio()
            self._processing_text = processing_text
        elif isinstance(frame, TTSUpdateSettingsFrame):
            if frame.delta is not None:
                await self._update_settings(frame.delta)
            elif frame.settings:
                # Backward-compatible path: convert legacy dict to settings object.
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(
                        "Passing a dict via TTSUpdateSettingsFrame(settings={...}) is deprecated "
                        "since 0.0.104, use TTSUpdateSettingsFrame(delta=TTSSettings(...)) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                delta = type(self._settings).from_mapping(frame.settings)
                await self._update_settings(delta)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._maybe_resume_frame_processing()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream with TTS-specific handling.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        # Clean up context when we see TTSStoppedFrame
        if isinstance(frame, TTSStoppedFrame) and frame.context_id:
            if frame.context_id in self._tts_contexts:
                logger.debug(f"{self} cleaning up TTS context {frame.context_id}")
                del self._tts_contexts[frame.context_id]

        if self._push_silence_after_stop and isinstance(frame, TTSStoppedFrame):
            silence_num_bytes = int(self._silence_time_s * self.sample_rate * 2)  # 16-bit
            silence_frame = TTSAudioRawFrame(
                audio=b"\x00" * silence_num_bytes,
                sample_rate=self.sample_rate,
                num_channels=1,
            )
            silence_frame.transport_destination = self._transport_destination
            await self.push_frame(silence_frame)

        if isinstance(frame, (TTSStartedFrame, TTSStoppedFrame, TTSAudioRawFrame, TTSTextFrame)):
            frame.transport_destination = self._transport_destination

        await super().push_frame(frame, direction)

        if self._push_stop_frames and (
            isinstance(frame, InterruptionFrame)
            or isinstance(frame, TTSStartedFrame)
            or isinstance(frame, TTSAudioRawFrame)
            or isinstance(frame, TTSStoppedFrame)
        ):
            await self._stop_frame_queue.put(frame)

    async def _stream_audio_frames_from_iterator(
        self,
        iterator: AsyncIterator[bytes],
        *,
        strip_wav_header: bool = False,
        in_sample_rate: Optional[int] = None,
        context_id: Optional[str] = None,
    ) -> AsyncGenerator[Frame, None]:
        """Stream audio frames from an async byte iterator with optional resampling.

        For WAV data, use `strip_wav_header=True` to strip the header and
        auto-detect the source sample rate. For raw PCM data, pass
        `in_sample_rate` directly. Audio is resampled to `self.sample_rate` when
        the source rate differs.

        Args:
            iterator: Async iterator yielding audio bytes.
            strip_wav_header: Strip WAV header and parse source sample rate from it.
            in_sample_rate: Source sample rate for raw PCM data. Overrides
                WAV-detected rate if both are provided.
            context_id: Unique identifier for this TTS context.

        """
        buffer = bytearray()
        source_sample_rate = in_sample_rate
        need_to_strip_wav_header = strip_wav_header

        async def maybe_resample(audio: bytes) -> bytes:
            if source_sample_rate and source_sample_rate != self.sample_rate:
                return await self._resampler.resample(audio, source_sample_rate, self.sample_rate)
            return audio

        async for chunk in iterator:
            if need_to_strip_wav_header and chunk.startswith(b"RIFF"):
                # Parse sample rate from WAV header (bytes 24-28, little-endian uint32).
                if len(chunk) >= 44 and source_sample_rate is None:
                    source_sample_rate = int.from_bytes(chunk[24:28], "little")
                chunk = chunk[44:]
                need_to_strip_wav_header = False

            # Append to current buffer.
            buffer.extend(chunk)

            # Round to nearest even number.
            aligned_length = len(buffer) & ~1  # 111111111...11110
            if aligned_length > 0:
                aligned_chunk = await maybe_resample(bytes(buffer[:aligned_length]))
                buffer = buffer[aligned_length:]  # keep any leftover byte

                if len(aligned_chunk) > 0:
                    frame = TTSAudioRawFrame(
                        bytes(aligned_chunk), self.sample_rate, 1, context_id=context_id
                    )
                    yield frame

        if len(buffer) > 0:
            # Make sure we don't need an extra padding byte.
            if len(buffer) % 2 == 1:
                buffer.extend(b"\x00")
            audio = await maybe_resample(bytes(buffer))
            yield TTSAudioRawFrame(audio, self.sample_rate, 1)

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        self._processing_text = False
        await self._text_aggregator.handle_interruption()
        for filter in self._text_filters:
            await filter.handle_interruption()

        self._llm_response_started = False
        self._streamed_text = ""
        self._text_aggregation_metrics_started = False
        if self._supports_word_timestamps:
            await self.reset_word_timestamps()

    async def _maybe_pause_frame_processing(self):
        if self._processing_text and self._pause_frame_processing:
            await self.pause_processing_frames()

    async def _maybe_resume_frame_processing(self):
        if self._pause_frame_processing:
            await self.resume_processing_frames()

    async def _process_text_frame(self, frame: TextFrame):
        async for aggregate in self._text_aggregator.aggregate(frame.text):
            includes_inter_frame_spaces = (
                frame.includes_inter_frame_spaces
                if aggregate.type == AggregationType.TOKEN
                else False
            )
            if aggregate.type != AggregationType.TOKEN:
                # Stop the aggregation metric on the first sentence only.
                await self.stop_text_aggregation_metrics()
            await self._push_tts_frames(
                AggregatedTextFrame(aggregate.text, aggregate.type), includes_inter_frame_spaces
            )

    async def _push_tts_frames(
        self,
        src_frame: AggregatedTextFrame,
        includes_inter_frame_spaces: Optional[bool] = False,
        append_tts_text_to_context: Optional[bool] = True,
        push_assistant_aggregation: Optional[bool] = False,
    ):
        type = src_frame.aggregated_by
        text = src_frame.text

        # Skip sending to TTS if the aggregation type is in the skip list. Simply
        # push the original frame downstream.
        if type in self._skip_aggregator_types:
            await self.push_frame(src_frame)
            return

        # Remove leading newlines only
        text = text.lstrip("\n")

        # Don't send only whitespace. This causes problems for some TTS models. But also don't
        # strip all whitespace, as whitespace can influence prosody.
        if not text.strip():
            return

        # This is just a flag that indicates if we sent something to the TTS
        # service. It will be cleared if we sent text because of a TTSSpeakFrame
        # or when we received an LLMFullResponseEndFrame
        self._processing_text = True

        # Accumulate text for a single debug log at flush time when streaming tokens.
        if self._is_streaming_tokens:
            self._streamed_text += text

        # Skip per-token processing metrics when streaming. The per-token
        # processing time is just websocket send overhead (~0.1ms) and not
        # meaningful. TTFB captures the important timing for streaming TTS.
        if not self._is_streaming_tokens:
            await self.start_processing_metrics()

        # Process all filters.
        for filter in self._text_filters:
            await filter.reset_interruption()
            text = await filter.filter(text)

        if not text.strip():
            if not self._is_streaming_tokens:
                await self.stop_processing_metrics()
            return

        # Create context ID and store metadata
        context_id = self.create_context_id()

        # To support use cases that may want to know the text before it's spoken, we
        # push the AggregatedTextFrame version before transforming and sending to TTS.
        # However, we do not want to add this text to the assistant context until it
        # is spoken, so we set append_to_context to False.
        src_frame.append_to_context = False
        src_frame.context_id = context_id
        await self.push_frame(src_frame)

        # Note: Text transformations are meant to only affect the text sent to the TTS for
        # TTS-specific purposes. This allows for explicit TTS modifications (e.g., inserting
        # TTS supported tags for spelling or emotion or replacing an @ with "at"). For TTS
        # services that support word-level timestamps, this CAN affect the resulting context
        # since the TTSTextFrames are generated from the TTS output stream
        transformed_text = text
        for aggregation_type, transform in self._text_transforms:
            if aggregation_type == type or aggregation_type == "*":
                transformed_text = await transform(transformed_text, type)

        self._tts_contexts[context_id] = TTSContext(
            append_to_context=append_tts_text_to_context
            if append_tts_text_to_context is not None
            else True,
            push_assistant_aggregation=push_assistant_aggregation,
        )

        # Apply any final text preparation (e.g., trailing space)
        prepared_text = self._prepare_text_for_tts(transformed_text)

        # Trigger event before starting TTS
        await self._call_event_handler("on_tts_request", context_id, prepared_text)

        await self.process_generator(self.run_tts(prepared_text, context_id))

        if not self._is_streaming_tokens:
            await self.stop_processing_metrics()

        if self._push_text_frames:
            # In TTS services that support word timestamps, the TTSTextFrames
            # are pushed as words are spoken. However, in the case where the TTS service
            # does not support word timestamps (i.e. _push_text_frames is True), we send
            # the original (non-transformed) text after the TTS generation has completed.
            # This way, if we are interrupted, the text is not added to the assistant
            # context and the context that IS added does not include TTS-specific tags
            # or transformations.
            frame = TTSTextFrame(text, aggregated_by=type)
            frame.includes_inter_frame_spaces = includes_inter_frame_spaces
            frame.context_id = context_id
            # Only override append_to_context if explicitly set
            if append_tts_text_to_context is not None:
                frame.append_to_context = append_tts_text_to_context
            await self.push_frame(frame)
            if push_assistant_aggregation:
                await self.push_frame(LLMAssistantPushAggregationFrame())

    async def _stop_frame_handler(self):
        has_started = False
        while True:
            try:
                frame = await asyncio.wait_for(
                    self._stop_frame_queue.get(), timeout=self._stop_frame_timeout_s
                )
                if isinstance(frame, TTSStartedFrame):
                    has_started = True
                elif isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
                    has_started = False
            except asyncio.TimeoutError:
                if has_started:
                    await self.push_frame(TTSStoppedFrame())
                    has_started = False

    #
    # Word timestamp methods (active when supports_word_timestamps=True)
    #

    async def start_word_timestamps(self):
        """Start tracking word timestamps from the current time."""
        if self._initial_word_timestamp == -1:
            self._initial_word_timestamp = self.get_clock().get_time()
            # If we cached some initial word times (because we didn't receive
            # audio), let's add them now.
            if self._initial_word_times:
                await self._add_word_timestamps(self._initial_word_times)
                self._initial_word_times = []

    async def reset_word_timestamps(self):
        """Reset word timestamp tracking."""
        self._initial_word_timestamp = -1

    async def add_word_timestamps(
        self, word_times: List[Tuple[str, float]], context_id: Optional[str] = None
    ):
        """Add word timestamps to the processing queue.

        Args:
            word_times: List of (word, timestamp) tuples where timestamp is in seconds.
            context_id: Unique identifier for the TTS context.
        """
        # Transform to include context_id in each tuple
        word_times_with_context = [(word, timestamp, context_id) for word, timestamp in word_times]

        if self._initial_word_timestamp == -1:
            # Cache word timestamps and don't add them until we have started
            # (i.e. we have some audio).
            self._initial_word_times.extend(word_times_with_context)
        else:
            await self._add_word_timestamps(word_times_with_context)

    def _create_words_task(self):
        if not self._words_task:
            self._words_queue: asyncio.Queue = asyncio.Queue()
            self._words_task = self.create_task(self._words_task_handler())

    async def _stop_words_task(self):
        if self._words_task:
            await self.cancel_task(self._words_task)
            self._words_task = None

    async def _add_word_timestamps(self, word_times_with_context: List[Tuple[str, float, str]]):
        for word, timestamp, context_id in word_times_with_context:
            await self._words_queue.put((word, seconds_to_nanoseconds(timestamp), context_id))

    async def _words_task_handler(self):
        last_pts = 0
        while True:
            frame = None
            (word, timestamp, context_id) = await self._words_queue.get()
            if word == "Reset" and timestamp == 0:
                await self.reset_word_timestamps()
                if self._llm_response_started:
                    self._llm_response_started = False
                    frame = LLMFullResponseEndFrame()
                    frame.pts = last_pts
            elif word == "TTSStoppedFrame" and timestamp == 0:
                frame = TTSStoppedFrame()
                frame.pts = last_pts
                frame.context_id = context_id
                if context_id in self._tts_contexts:
                    if self._tts_contexts[context_id].push_assistant_aggregation:
                        await self.push_frame(LLMAssistantPushAggregationFrame())
            else:
                # Assumption: word-by-word text frames don't include spaces, so
                # we can rely on the default includes_inter_frame_spaces=False
                frame = TTSTextFrame(word, aggregated_by=AggregationType.WORD)
                frame.pts = self._initial_word_timestamp + timestamp
                frame.context_id = context_id
                # Look up append_to_context from context metadata
                if context_id in self._tts_contexts:
                    frame.append_to_context = self._tts_contexts[context_id].append_to_context
            if frame:
                last_pts = frame.pts
                await self.push_frame(frame)
            self._words_queue.task_done()


class WordTTSService(TTSService):
    """Deprecated. Use TTSService with supports_word_timestamps=True instead.

    .. deprecated:: 0.0.104
        Word timestamp functionality has been moved to TTSService. Pass
        ``supports_word_timestamps=True`` to TTSService (or any subclass) instead.
    """

    def __init__(self, **kwargs):
        """Initialize the Word TTS service.

        Args:
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(supports_word_timestamps=True, **kwargs)


class WebsocketTTSService(TTSService, WebsocketService):
    """Base class for websocket-based TTS services.

    Combines TTS functionality with websocket connectivity, providing automatic
    error handling and reconnection capabilities.

    Event handlers:
        on_connection_error: Called when a websocket connection error occurs.

    Example::

        @tts.event_handler("on_connection_error")
        async def on_connection_error(tts: TTSService, error: str):
            logger.error(f"TTS connection error: {error}")
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the Websocket TTS service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        TTSService.__init__(self, **kwargs)
        WebsocketService.__init__(self, reconnect_on_error=reconnect_on_error, **kwargs)

    async def _report_error(self, error: ErrorFrame):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error_frame(error)


class InterruptibleTTSService(WebsocketTTSService):
    """Websocket-based TTS service that handles interruptions without word timestamps.

    Designed for TTS services that don't support word timestamps. Handles interruptions
    by reconnecting the websocket when the bot is speaking and gets interrupted.
    """

    def __init__(self, **kwargs):
        """Initialize the Interruptible TTS service.

        Args:
            **kwargs: Additional arguments passed to the parent WebsocketTTSService.
        """
        super().__init__(**kwargs)

        # Indicates if the bot is speaking. If the bot is not speaking we don't
        # need to reconnect when the user speaks. If the bot is speaking and the
        # user interrupts we need to reconnect.
        self._bot_speaking = False

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        if self._bot_speaking:
            await self._disconnect()
            await self._connect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with bot speaking state tracking.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False


class WebsocketWordTTSService(WebsocketTTSService):
    """Deprecated. Use WebsocketTTSService with supports_word_timestamps=True instead.

    .. deprecated:: 0.0.104
        Word timestamp functionality has been moved to TTSService. Pass
        ``supports_word_timestamps=True`` to WebsocketTTSService instead.
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the Websocket Word TTS service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(
            supports_word_timestamps=True, reconnect_on_error=reconnect_on_error, **kwargs
        )


class InterruptibleWordTTSService(InterruptibleTTSService):
    """Deprecated. Use InterruptibleTTSService with supports_word_timestamps=True instead.

    .. deprecated:: 0.0.104
        Word timestamp functionality has been moved to TTSService. Pass
        ``supports_word_timestamps=True`` to InterruptibleTTSService instead.
    """

    def __init__(self, **kwargs):
        """Initialize the Interruptible Word TTS service.

        Args:
            **kwargs: Additional arguments passed to the parent InterruptibleTTSService.
        """
        super().__init__(supports_word_timestamps=True, **kwargs)


class AudioContextTTSService(WebsocketTTSService):
    """Base class for websocket-based TTS services with audio context management.

    This is a base class for websocket-based TTS services that allow correlating
    the generated audio with the requested text through audio contexts.

    Each request could be multiple sentences long which are grouped by
    context. For this to work, the TTS service needs to support handling
    multiple requests at once (i.e. multiple simultaneous contexts).

    The audio received from the TTS will be played in context order. That is, if
    we requested audio for a context "A" and then audio for context "B", the
    audio from context ID "A" will be played first.
    """

    _CONTEXT_KEEPALIVE = object()

    def __init__(
        self,
        *,
        reuse_context_id_within_turn: bool = True,
        reconnect_on_error: bool = True,
        **kwargs,
    ):
        """Initialize the Audio Context TTS service.

        Args:
            reuse_context_id_within_turn: Whether the service should reuse context IDs within the same turn.
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to the parent WebsocketTTSService.
        """
        super().__init__(reconnect_on_error=reconnect_on_error, **kwargs)
        self._reuse_context_id_within_turn = reuse_context_id_within_turn
        self._context_id = None
        self._contexts: Dict[str, asyncio.Queue] = {}
        self._audio_context_task = None

    async def create_audio_context(self, context_id: str):
        """Create a new audio context for grouping related audio.

        Args:
            context_id: Unique identifier for the audio context.
        """
        # Set the context ID if not already set
        if not self._context_id:
            self._context_id = context_id

        await self._contexts_queue.put(context_id)
        self._contexts[context_id] = asyncio.Queue()
        logger.trace(f"{self} created audio context {context_id}")

    async def append_to_audio_context(self, context_id: str, frame: TTSAudioRawFrame):
        """Append audio to an existing context.

        Args:
            context_id: The context to append audio to.
            frame: The audio frame to append.
        """
        if self.audio_context_available(context_id):
            logger.trace(f"{self} appending audio {frame} to audio context {context_id}")
            await self._contexts[context_id].put(frame)
        else:
            logger.warning(f"{self} unable to append audio to context {context_id}")

    async def remove_audio_context(self, context_id: str):
        """Remove an existing audio context.

        Args:
            context_id: The context to remove.
        """
        if self.audio_context_available(context_id):
            # We just mark the audio context for deletion by appending
            # None. Once we reach None while handling audio we know we can
            # safely remove the context.
            logger.trace(f"{self} marking audio context {context_id} for deletion")
            await self._contexts[context_id].put(None)
        else:
            logger.warning(f"{self} unable to remove context {context_id}")

    def has_active_audio_context(self) -> bool:
        """Check if there is an active audio context.

        Returns:
            True if an active audio context exists, False otherwise.
        """
        return self._context_id is not None and self.audio_context_available(self._context_id)

    def get_active_audio_context_id(self) -> Optional[str]:
        """Get the active audio context ID.

        Returns:
            The active context ID, or None if no context is active.
        """
        return self._context_id

    async def remove_active_audio_context(self):
        """Remove the active audio context."""
        if self._context_id:
            await self.remove_audio_context(self._context_id)
            self.reset_active_audio_context()

    def reset_active_audio_context(self):
        """Reset the active audio context."""
        self._context_id = None

    def audio_context_available(self, context_id: str) -> bool:
        """Check whether the given audio context is registered.

        Args:
            context_id: The context ID to check.

        Returns:
            True if the context exists and is available.
        """
        return context_id in self._contexts

    def create_context_id(self) -> str:
        """Generate or reuse a context ID based on concurrent TTS support.

        If _reuse_context_id_within_turn is False and a context already exists,
        the existing context ID is returned. Otherwise, a new unique context
        ID is generated.

        Returns:
            A context ID string for the TTS request.
        """
        if self._reuse_context_id_within_turn and self._context_id:
            self._refresh_active_audio_context()
            return self._context_id
        return super().create_context_id()

    def _refresh_active_audio_context(self):
        """Signal that the audio context is still in use, resetting the timeout."""
        if self.has_active_audio_context():
            self._contexts[self._context_id].put_nowait(AudioContextTTSService._CONTEXT_KEEPALIVE)

    async def start(self, frame: StartFrame):
        """Start the audio context TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._create_audio_context_task()

    async def stop(self, frame: EndFrame):
        """Stop the audio context TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        if self._audio_context_task:
            # Indicate no more audio contexts are available. this will end the
            # task cleanly after all contexts have been processed.
            await self._contexts_queue.put(None)
            await self._audio_context_task
            self._audio_context_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the audio context TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._stop_audio_context_task()

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self._stop_audio_context_task()
        await self.on_audio_context_interrupted(context_id=self._context_id)
        self.reset_active_audio_context()
        self._create_audio_context_task()

    def _create_audio_context_task(self):
        if not self._audio_context_task:
            self._contexts_queue = asyncio.Queue()
            self._contexts: Dict[str, asyncio.Queue] = {}
            self._audio_context_task = self.create_task(self._audio_context_task_handler())

    async def _stop_audio_context_task(self):
        if self._audio_context_task:
            await self.cancel_task(self._audio_context_task)
            self._audio_context_task = None

    async def _audio_context_task_handler(self):
        """In this task we process audio contexts in order."""
        running = True
        while running:
            context_id = await self._contexts_queue.get()
            self._context_id = context_id

            if context_id:
                # Process the audio context until the context doesn't have more
                # audio available (i.e. we find None).
                await self._handle_audio_context(context_id)

                # We just finished processing the context, so we can safely remove it.
                del self._contexts[context_id]
                await self.on_audio_context_completed(context_id=context_id)
                self.reset_active_audio_context()

                # Append some silence between sentences.
                silence = b"\x00" * self.sample_rate
                frame = TTSAudioRawFrame(
                    audio=silence,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
                await self.push_frame(frame)
            else:
                running = False

            self._contexts_queue.task_done()

    async def _handle_audio_context(self, context_id: str):
        # If we don't receive any audio during this time, we consider the context finished.
        AUDIO_CONTEXT_TIMEOUT = 3.0
        queue = self._contexts[context_id]
        running = True
        while running:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=AUDIO_CONTEXT_TIMEOUT)
                if frame is AudioContextTTSService._CONTEXT_KEEPALIVE:
                    # Context is still in use, reset the timeout.
                    continue

                if frame:
                    await self.push_frame(frame)
                running = frame is not None
            except asyncio.TimeoutError:
                # We didn't get audio, so let's consider this context finished.
                logger.trace(f"{self} time out on audio context {context_id}")
                break

    async def on_audio_context_interrupted(self, context_id: str):
        """Called when an audio context is cancelled due to an interruption.

        Override this in a subclass to perform provider-specific cleanup (e.g.
        sending a cancel/close message over the WebSocket) when the bot is
        interrupted mid-speech.  The audio context task has already been stopped
        and the active context has **not** yet been reset when this is called,
        so ``context_id`` reflects the context that was cut short.

        Args:
            context_id: The ID of the audio context that was interrupted, or
                ``None`` if no context was active at the time.
        """
        pass

    async def on_audio_context_completed(self, context_id: str):
        """Called after an audio context has finished playing all of its audio.

        Override this in a subclass to perform provider-specific cleanup (e.g.
        sending a close-context message to free server-side resources) once an
        audio context has been fully processed.  The context entry has already
        been removed from the internal context map, and the active context has
        **not** yet been reset when this is called.

        Args:
            context_id: The ID of the audio context that finished processing.
        """
        pass


class AudioContextWordTTSService(AudioContextTTSService):
    """Deprecated. Use AudioContextTTSService with supports_word_timestamps=True instead.

    .. deprecated:: 0.0.104
        Word timestamp functionality has been moved to TTSService. Pass
        ``supports_word_timestamps=True`` to AudioContextTTSService instead.
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the Audio Context Word TTS service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(
            supports_word_timestamps=True, reconnect_on_error=reconnect_on_error, **kwargs
        )
