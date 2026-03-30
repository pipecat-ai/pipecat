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
    SystemFrame,
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


@dataclass
class _WordTimestampEntry:
    """Internal: word timestamp routed through an audio context queue."""

    word: str
    timestamp: float
    context_id: str


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

    _CONTEXT_KEEPALIVE = object()

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
        # if True, TTSService will push TTSStartedFrames and create audio contexts automatically
        push_start_frame: bool = False,
        # if push_stop_frames is True, wait for this idle period before pushing TTSStoppedFrame
        stop_frame_timeout_s: float = 3.0,
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
        # if True, the context ID is reused within an LLM turn
        reuse_context_id_within_turn: bool = True,
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
            push_start_frame: Whether to automatically create audio contexts and push TTSStartedFrames.
                When True, the base class handles ``create_audio_context`` and yields ``TTSStartedFrame``
                before each synthesis call, so ``run_tts`` implementations do not need to.
            stop_frame_timeout_s: Idle time before pushing TTSStoppedFrame when push_stop_frames is True.
            push_silence_after_stop: Whether to push silence audio after TTSStoppedFrame.
            silence_time_s: Duration of silence to push when push_silence_after_stop is True.
            pause_frame_processing: Whether to pause frame processing during audio generation.
            append_trailing_space: Whether to append a trailing space to text before sending to TTS.
                This helps prevent some TTS services from vocalizing trailing punctuation (e.g., "dot").
            sample_rate: Output sample rate for generated audio.
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
            reuse_context_id_within_turn: Whether the service should reuse context IDs within the
                same turn.
            **kwargs: Additional arguments passed to the parent AIService.
        """
        super().__init__(
            settings=settings
            # Here in case subclass doesn't implement more specific settings
            # (which hopefully should be rare)
            or TTSSettings(),
            **kwargs,
        )

        # Convert Language enum to service-specific format at init time.
        # Runtime updates are handled by _update_settings(), but init-time
        # settings bypass that path and need explicit conversion.
        # Raw strings (e.g. "de-DE") are first converted to Language enums
        # so they go through the same resolution logic.
        if isinstance(self._settings.language, str) and not isinstance(
            self._settings.language, Language
        ):
            try:
                self._settings.language = Language(self._settings.language)
            except ValueError:
                logger.debug(
                    f"Language string '{self._settings.language}' is not a recognized "
                    f"Language code. It will be passed to the service as-is."
                )
        if isinstance(self._settings.language, Language):
            converted = self.language_to_service_language(self._settings.language)
            if converted is not None:
                self._settings.language = converted

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
        self._push_start_frame: bool = push_start_frame
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

        self._processing_text: bool = False
        self._tts_contexts: Dict[str, TTSContext] = {}
        self._streamed_text: str = ""
        self._text_aggregation_metrics_started: bool = False

        # Word timestamp state
        self._initial_word_timestamp: int = -1
        self._initial_word_times: List[Tuple[str, float, Optional[str]]] = []
        # PTS of the last word frame pushed via _add_word_timestamps, used to assign
        # correct PTS to TTSStoppedFrame and LLMFullResponseEndFrame.
        self._word_last_pts: int = 0
        self._llm_response_started: bool = False
        self._reuse_context_id_within_turn: bool = reuse_context_id_within_turn

        # _turn_context_id:
        #   Set on LLMFullResponseStartFrame and cleared after LLMFullResponseEndFrame
        #   is processed (i.e. after flush). All sentences within one LLM turn share
        #   this ID so the TTS service groups them into a single audio context.
        #   Temporarily set to None for TTSSpeakFrame utterances, which are standalone.
        #
        # _playing_context_id (playback-side cursor):
        #   Set by _audio_context_task_handler as it dequeues contexts for playback.
        #   Cleared by reset_active_audio_context() on interruption. Used by
        #   has_active_audio_context() and get_active_audio_context_id().
        #
        # Both fields may hold the same value during a turn, but
        # they clear at different times: _turn_context_id is cleared when the LLM turn
        # ends (synthesis done) while _playing_context_id remains set until the audio
        # finishes playing. Merging them would null out the playback cursor prematurely.
        self._playing_context_id: Optional[str] = None
        self._turn_context_id: Optional[str] = None
        self._audio_contexts: Dict[str, asyncio.Queue] = {}
        self._audio_context_task: Optional[asyncio.Task] = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_connection_error")
        self._register_event_handler("on_tts_request")

        # Whether the TTS process is currently yielding audio frames synchronously.
        self._is_yielding_frames_synchronously = False

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
        """Generate or reuse a context ID based on concurrent TTS support.

        Returns:
            A context ID string for the TTS request.
        """
        if self._reuse_context_id_within_turn and self._turn_context_id:
            self._refresh_audio_context(self._turn_context_id)
            return self._turn_context_id
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

    async def flush_audio(self, context_id: Optional[str] = None):
        """Flush any buffered audio data.

        Args:
            context_id: The specific context to flush. If None, falls back to the
                currently active context (for non-concurrent services).
        """
        pass

    async def start(self, frame: StartFrame):
        """Start the TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._sample_rate = self._init_sample_rate or frame.audio_out_sample_rate
        self._create_audio_context_task()

    async def stop(self, frame: EndFrame):
        """Stop the TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        if self._audio_context_task:
            # Sentinel None shuts down the serialization queue once all
            # pending contexts and frames have been processed.
            await self._serialization_queue.put(None)
            await self._audio_context_task
            self._audio_context_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._stop_audio_context_task()

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
        # Translate language *before* applying so the stored value is canonical.
        # Raw strings are first converted to Language enums for proper resolution.
        if (
            is_given(delta.language)
            and isinstance(delta.language, str)
            and not isinstance(delta.language, Language)
        ):
            try:
                delta.language = Language(delta.language)
            except ValueError:
                logger.debug(
                    f"Language string '{delta.language}' is not a recognized "
                    f"Language code. It will be passed to the service as-is."
                )
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

    async def on_turn_context_created(self, context_id: str):
        """Called when a new turn context ID has been created.

        Override to perform provider-specific setup (e.g., eagerly opening a
        server-side context) before text starts flowing. This is called from
        ``process_frame`` when an ``LLMFullResponseStartFrame`` or ``TTSSpeakFrame`` arrives.

        Args:
            context_id: The newly created turn context ID.
        """
        pass

    async def on_turn_context_completed(self):
        """Handle the completion of a turn."""
        # For HTTP services they emit the frames synchronously, so close the audio context here
        # once all frames (including TTSTextFrame above) have been enqueued.
        if self._is_yielding_frames_synchronously and self.audio_context_available(
            self._turn_context_id
        ):
            if self._push_stop_frames:
                await self.append_to_audio_context(
                    self._turn_context_id, TTSStoppedFrame(context_id=self._turn_context_id)
                )
            await self.remove_audio_context(self._turn_context_id)

        # Flush any pending audio so the TTS service closes the current context.
        # Only flush if the context was actually opened (text reached run_tts).
        # When an interruption arrives before any text flows, the turn context ID
        # exists but was never registered via create_audio_context, so flushing
        # would send a message for a context the provider never opened.
        if self._turn_context_id and self.audio_context_available(self._turn_context_id):
            await self.flush_audio(context_id=self._turn_context_id)

        # Reset the turn context ID
        self._turn_context_id = None

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
            # New LLM turn → assign a fresh context ID shared by all sentences
            self._turn_context_id = self.create_context_id()
            await self.on_turn_context_created(self._turn_context_id)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            # Flush any remaining text (including text waiting for lookahead)
            remaining = await self._text_aggregator.flush()
            # Stop the aggregation metric (no-op if already stopped on first sentence).
            await self.stop_text_aggregation_metrics()
            if remaining:
                await self._push_tts_frames(AggregatedTextFrame(remaining.text, remaining.type))

            # We pause processing incoming frames if the LLM response included
            # text (it might be that it's only a function calling response). We
            # pause to avoid audio overlapping.
            await self._maybe_pause_frame_processing()

            # Log accumulated streamed text and emit aggregated usage metric.
            if self._streamed_text:
                logger.debug(f"{self}: Generating TTS [{self._streamed_text}]")
                await super().start_tts_usage_metrics(self._streamed_text)
                self._streamed_text = ""

            # Reset aggregator state
            self._processing_text = False
            if isinstance(frame, LLMFullResponseEndFrame):
                if self._push_text_frames:
                    # Route through the serialization queue so the frame is
                    # emitted only after the audio context has been fully
                    # drained (including the final TTSTextFrame).  Pushing
                    # directly would let it race ahead of queued text frames.
                    await self._serialization_queue.put(frame)
            else:
                await self.push_frame(frame, direction)

            await self.on_turn_context_completed()
        elif isinstance(frame, TTSSpeakFrame):
            # Store if we were processing text or not so we can set it back.
            processing_text = self._processing_text
            # TTSSpeakFrame is independent — temporarily clear the turn context
            # so create_context_id() generates a fresh UUID for this utterance.
            saved_turn_context_id = self._turn_context_id
            self._turn_context_id = None
            # Creating a new context_id for the TTS request.
            self._turn_context_id = self.create_context_id()
            await self.on_turn_context_created(self._turn_context_id)
            # If we are not receiving text from the LLM, we can assume that the SpeakFrame should be automatically added to the context
            push_assistant_aggregation = frame.append_to_context and not self._llm_response_started
            # Assumption: text in TTSSpeakFrame does not include inter-frame spaces
            await self._push_tts_frames(
                AggregatedTextFrame(frame.text, AggregationType.SENTENCE),
                append_tts_text_to_context=frame.append_to_context,
                push_assistant_aggregation=push_assistant_aggregation,
            )
            await self.on_turn_context_completed()
            # We pause processing incoming frames because we are sending data to
            # the TTS. We pause to avoid audio overlapping.
            await self._maybe_pause_frame_processing()
            self._turn_context_id = saved_turn_context_id
            self._processing_text = processing_text
        elif isinstance(frame, TTSUpdateSettingsFrame):
            if frame.service is not None and frame.service is not self:
                await self.push_frame(frame, direction)
            elif frame.delta is not None:
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
            if direction == FrameDirection.DOWNSTREAM and not isinstance(frame, SystemFrame):
                # Route non-system downstream frames through the serialization queue so they
                # are emitted in the same order they arrive relative to any audio contexts that
                # are already queued (e.g. a FooFrame sent right after a TTSSpeakFrame must
                # not overtake the TTSStartedFrame / TTSAudioRawFrame / TTSStoppedFrame
                # sequence from that speak frame).
                await self._serialization_queue.put(frame)
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
                if self._tts_contexts[frame.context_id].push_assistant_aggregation:
                    await self.push_frame(LLMAssistantPushAggregationFrame())
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
        await self.reset_word_timestamps()

        await self._stop_audio_context_task()
        audio_contexts = self.get_audio_contexts()
        if audio_contexts:
            for ctx_id in audio_contexts:
                await self.on_audio_context_interrupted(context_id=ctx_id)
        self.reset_active_audio_context()
        self._turn_context_id = None
        self._word_last_pts = 0
        self._create_audio_context_task()

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
        # Route AggregatedTextFrame through the serialization queue so it is emitted
        # immediately before the TTSStartedFrame of the audio context it describes,
        # rather than racing ahead of audio frames from a previous context.
        if not self.audio_context_available(context_id):
            await self._serialization_queue.put(src_frame)
        # Otherwise, if the context already exists, we append the AggregatedTextFrame
        # to the existing context queue.
        else:
            await self.append_to_audio_context(context_id, src_frame)

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

        if self._push_start_frame and not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            await self.start_ttfb_metrics()
            await self.append_to_audio_context(context_id, TTSStartedFrame(context_id=context_id))

        await self.tts_process_generator(context_id, self.run_tts(prepared_text, context_id))

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
            # Appending to the context, so it preserves the ordering.
            await self.append_to_audio_context(context_id, frame)

    async def tts_process_generator(
        self, context_id: str, generator: AsyncGenerator[Frame | None, None]
    ) -> bool:
        """Process frames from an async generator, routing them through the audio context.

        All non-None frames yielded by the generator are appended to the audio context
        identified by context_id. The audio context must be created by run_tts (via
        create_audio_context) before the first frame is yielded.

        WebSocket services yield None to signal that audio will arrive via a separate
        receive loop; those services manage context lifetime themselves (via remove_audio_context
        in the receive loop on "done"). HTTP services never yield None and do NOT call
        remove_audio_context in run_tts — the caller (_synthesize_text) closes the context
        after appending any remaining frames (e.g. TTSTextFrame).

        Args:
            context_id: The audio context to route frames to.
            generator: An async generator yielding Frame objects or None.

        """
        is_yielding_frames = False
        async for frame in generator:
            if frame:
                await self.append_to_audio_context(context_id, frame)
                if isinstance(frame, TTSAudioRawFrame):
                    is_yielding_frames = True

        self._is_yielding_frames_synchronously = is_yielding_frames

    #
    # Word timestamp methods
    #

    async def start_word_timestamps(self):
        """Start tracking word timestamps from the current time."""
        if self._initial_word_timestamp == -1:
            current_time = self.get_clock().get_time()
            # Initialize word timestamp tracking. Use the last emitted timestamp if it's ahead
            # of current time to maintain continuity across overlapping audio contexts.
            self._initial_word_timestamp = (
                self._word_last_pts if self._word_last_pts > current_time else current_time
            )
            # If we cached some initial word times (because we didn't receive
            # audio), let's add them now.
            if self._initial_word_times:
                cached = self._initial_word_times.copy()
                self._initial_word_times = []
                for word, timestamp_seconds, ctx_id in cached:
                    await self._add_word_timestamps([(word, timestamp_seconds)], ctx_id)

    async def reset_word_timestamps(self):
        """Reset word timestamp tracking."""
        self._initial_word_timestamp = -1
        # Discard any pre-audio word timestamps from the interrupted turn so they
        # cannot be flushed into the next context after the audio baseline resets.
        self._initial_word_times = []

    async def add_word_timestamps(
        self, word_times: List[Tuple[str, float]], context_id: Optional[str] = None
    ):
        """Add word timestamps for processing.

        When an audio context exists for this context_id, timestamps are routed into the
        per-context audio queue alongside audio frames so they are processed in strict
        playback order by _handle_audio_context. Otherwise they are processed immediately
        via _add_word_timestamps.

        Args:
            word_times: List of (word, timestamp) tuples where timestamp is in seconds.
            context_id: Unique identifier for the TTS context.
        """
        if context_id and self.audio_context_available(context_id):
            for word, timestamp in word_times:
                await self.append_to_audio_context(
                    context_id,
                    _WordTimestampEntry(
                        word=word,
                        timestamp=timestamp,
                        context_id=context_id,
                    ),
                )
        else:
            await self._add_word_timestamps(word_times=word_times, context_id=context_id)

    async def _add_word_timestamps(
        self, word_times: List[Tuple[str, float]], context_id: Optional[str] = None
    ):
        """Process word timestamps directly, building and pushing TTSTextFrames inline.

        Used both from _handle_audio_context (via _WordTimestampEntry) and from services
        that do not use audio contexts. Each entry emits a TTSTextFrame with a PTS
        relative to the baseline established by start_word_timestamps().

        When the baseline (_initial_word_timestamp) is not yet set, entries are cached
        in _initial_word_times and flushed once start_word_timestamps() is called
        (i.e. when the first audio chunk is received).
        """
        for word, timestamp in word_times:
            ts_ns = seconds_to_nanoseconds(timestamp)
            if self._initial_word_timestamp == -1:
                # Cache until we have audio and can compute PTS.
                self._initial_word_times.append((word, timestamp, context_id))
            else:
                # Assumption: word-by-word text frames don't include spaces, so
                # we can rely on the default includes_inter_frame_spaces=False
                frame = TTSTextFrame(word, aggregated_by=AggregationType.WORD)
                frame.pts = self._initial_word_timestamp + ts_ns
                frame.context_id = context_id
                if context_id in self._tts_contexts:
                    frame.append_to_context = self._tts_contexts[context_id].append_to_context
                self._word_last_pts = frame.pts
                await self.push_frame(frame)

    #
    # Audio context methods (active when using websocket-based TTS with context management)
    #

    async def create_audio_context(self, context_id: str):
        """Create a new audio context for grouping related audio.

        Args:
            context_id: Unique identifier for the audio context.
        """
        await self._serialization_queue.put(context_id)
        self._audio_contexts[context_id] = asyncio.Queue()
        logger.trace(f"{self} created audio context {context_id}")

    async def append_to_audio_context(
        self, context_id: str, frame: Frame | _WordTimestampEntry | None
    ):
        """Append a frame or word-timestamp entry to an existing audio context queue.

        Passing ``None`` signals end-of-context (used by remove_audio_context to mark
        the queue for deletion). If the context no longer exists but the context_id
        matches the active turn, the context is transparently recreated before appending.

        Args:
            context_id: The context to append to.
            frame: The frame, word-timestamp entry, or ``None`` (end-of-context sentinel)
                to append.
        """
        if not context_id:
            logger.debug(f"{self} unable to append audio to context: no context ID provided")
            return
        if self.audio_context_available(context_id):
            logger.trace(f"{self} appending audio {frame} to audio context {context_id}")
            await self._audio_contexts[context_id].put(frame)
        # In case the frame is None, we should not recreate the context.
        elif context_id == self._turn_context_id and frame:
            # Sometimes the HTTP service can take more than 3 seconds without sending any audio
            # So we are now recreating the context id while we are in the same turn
            logger.debug(f"{self} recreating audio context {context_id}")
            await self.create_audio_context(context_id)
            logger.trace(f"{self} appending audio {frame} to audio context {context_id}")
            await self._audio_contexts[context_id].put(frame)
        else:
            logger.debug(f"{self} unable to append audio to context {context_id}")

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
            await self.append_to_audio_context(context_id, None)
        else:
            logger.warning(f"{self} unable to remove context {context_id}")

    def has_active_audio_context(self) -> bool:
        """Check if there is an active audio context.

        Returns:
            True if an active audio context exists, False otherwise.
        """
        return self._playing_context_id is not None and self.audio_context_available(
            self._playing_context_id
        )

    def get_audio_contexts(self) -> List[str]:
        """Get a list of all available audio contexts."""
        return list(self._audio_contexts.keys())

    def get_active_audio_context_id(self) -> Optional[str]:
        """Get the active audio context ID.

        Returns:
            The active context ID, or None if no context is active.
        """
        return self._playing_context_id

    async def remove_active_audio_context(self):
        """Remove the active audio context."""
        if self._playing_context_id:
            await self.remove_audio_context(self._playing_context_id)
            self.reset_active_audio_context()

    def reset_active_audio_context(self):
        """Reset the active audio context."""
        self._playing_context_id = None

    def audio_context_available(self, context_id: str) -> bool:
        """Check whether the given audio context is registered.

        Args:
            context_id: The context ID to check.

        Returns:
            True if the context exists and is available.
        """
        return context_id in self._audio_contexts

    def _refresh_audio_context(self, context_id: str):
        """Signal that the audio context is still in use, resetting the timeout."""
        if self.audio_context_available(context_id):
            self._audio_contexts[context_id].put_nowait(TTSService._CONTEXT_KEEPALIVE)

    def _create_audio_context_task(self):
        if not self._audio_context_task:
            # Single FIFO queue that serializes everything the TTS service emits downstream.
            # Items can be:
            #   str   – an audio context ID: process the per-context audio queue in full before
            #           moving on (see _handle_audio_context).
            #   Frame – a non-system downstream frame (e.g. AggregatedTextFrame, FooFrame) that
            #           must be emitted in-order relative to surrounding audio contexts.
            #   None  – shutdown sentinel (sent by stop()).
            self._serialization_queue: asyncio.Queue = asyncio.Queue()
            self._audio_contexts: Dict[str, asyncio.Queue] = {}
            self._audio_context_task = self.create_task(self._audio_context_task_handler())

    async def _stop_audio_context_task(self):
        if self._audio_context_task:
            await self.cancel_task(self._audio_context_task)
            self._audio_context_task = None

    async def _audio_context_task_handler(self):
        """Drain the serialization queue, preserving downstream frame order.

        The queue carries three kinds of items (see _create_audio_context_task):

        * str  – audio context ID: block until all audio for that context has been
                 pushed downstream, then call on_audio_context_completed().
        * Frame – a non-system downstream frame that must be emitted at this exact
                  position in the output stream (e.g. AggregatedTextFrame preceding
                  its audio, or an arbitrary frame that arrived between two speak frames).
        * None – shutdown sentinel; exit the loop once reached.
        """
        running = True
        while running:
            context_value = await self._serialization_queue.get()
            if isinstance(context_value, Frame):
                await self.push_frame(context_value)
            elif isinstance(context_value, str):
                context_id = context_value
                self._playing_context_id = context_id

                # Process the audio context until the context doesn't have more
                # audio available (i.e. we find None).
                await self._handle_audio_context(context_id)

                # We just finished processing the context, so we can safely remove it.
                del self._audio_contexts[context_id]
                await self.on_audio_context_completed(context_id=context_id)
                self.reset_active_audio_context()
            else:
                running = False

            self._serialization_queue.task_done()

    async def _maybe_reset_word_timestamps(self):
        """Reset word-timestamp state and emit LLMFullResponseEndFrame if needed.

        Called at the end of an audio context (either on clean completion timeout or
        when the context queue is drained). Resets the PTS baseline so the next turn
        starts fresh. If an LLM response is still marked as in-progress and text frames
        are not being pushed (which would have already emitted the frame), an
        LLMFullResponseEndFrame is pushed with the PTS of the last word frame.
        """
        await self.reset_word_timestamps()
        # If self._push_text_frames is True, we have already pushed the original LLMFullResponseEndFrame
        if self._llm_response_started and not self._push_text_frames:
            self._llm_response_started = False
            frame = LLMFullResponseEndFrame()
            frame.pts = self._word_last_pts
            await self.push_frame(frame)

    async def _handle_audio_context(self, context_id: str):
        """Process items from an audio context queue until it is exhausted."""
        queue = self._audio_contexts[context_id]
        running = True
        timestamps_started = False
        should_push_stop_frame = False
        while running:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=self._stop_frame_timeout_s)
                if frame is TTSService._CONTEXT_KEEPALIVE:
                    # Context is still in use, reset the timeout.
                    continue
                elif frame is None:
                    running = False
                elif isinstance(frame, _WordTimestampEntry):
                    # Route word timestamps through _add_word_timestamps so they are
                    # processed in playback order alongside audio frames.
                    await self._add_word_timestamps(
                        [(frame.word, frame.timestamp)], frame.context_id
                    )
                    continue
                elif isinstance(frame, TTSAudioRawFrame):
                    # Set the word-timestamp baseline once, on the first audio chunk.
                    if not timestamps_started:
                        await self.stop_ttfb_metrics()
                        await self.start_word_timestamps()
                        timestamps_started = True

                if frame:
                    if isinstance(frame, TTSStartedFrame):
                        should_push_stop_frame = self._push_stop_frames
                    elif isinstance(frame, TTSStoppedFrame):
                        should_push_stop_frame = False
                        # Setting the last word timestamp as the TTSStoppedFrame PTS
                        if not frame.pts:
                            frame.pts = self._word_last_pts

                    if isinstance(frame, ErrorFrame):
                        await self.push_error_frame(frame)
                    else:
                        await self.push_frame(frame)
            except asyncio.TimeoutError:
                # We didn't get audio, so let's consider this context finished.
                logger.trace(f"{self} time out on audio context {context_id}")
                if should_push_stop_frame and self._push_stop_frames:
                    await self.push_frame(TTSStoppedFrame(context_id=context_id))
                    should_push_stop_frame = False
                break

        if should_push_stop_frame and self._push_stop_frames:
            await self.push_frame(TTSStoppedFrame(context_id=context_id))
        await self._maybe_reset_word_timestamps()

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


class WordTTSService(TTSService):
    """Deprecated. Use TTSService directly instead.

    .. deprecated:: 0.0.105
        Word timestamp functionality is now always active in TTSService.
    """

    def __init__(self, **kwargs):
        """Initialize the Word TTS service.

        Args:
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(**kwargs)


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

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream with TTS-specific handling.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        # This prevents a race condition in cases where run_tts has been invoked but the
        # BotStartedSpeakingFrame has not yet been received, which could allow stale audio to leak through.
        if isinstance(frame, TTSStartedFrame):
            self._bot_speaking = True

        await super().push_frame(frame, direction)

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
    """Deprecated. Use WebsocketTTSService directly instead.

    .. deprecated:: 0.0.105
        Word timestamp functionality is now always active in TTSService.
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the Websocket Word TTS service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(reconnect_on_error=reconnect_on_error, **kwargs)


class InterruptibleWordTTSService(InterruptibleTTSService):
    """Deprecated. Use InterruptibleTTSService directly instead.

    .. deprecated:: 0.0.105
        Word timestamp functionality is now always active in TTSService.
    """

    def __init__(self, **kwargs):
        """Initialize the Interruptible Word TTS service.

        Args:
            **kwargs: Additional arguments passed to the parent InterruptibleTTSService.
        """
        super().__init__(**kwargs)


class AudioContextTTSService(WebsocketTTSService):
    """Deprecated. Inherit from WebsocketTTSService directly instead.

    Audio context management (previously the main purpose of this class) is now
    built into TTSService. This class is kept only for backwards compatibility.

    .. deprecated:: 0.0.105
        Subclass :class:`WebsocketTTSService` directly and pass
        ``reuse_context_id_within_turn`` as
        keyword arguments to its ``__init__``.
    """

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
        import warnings

        warnings.warn(
            "AudioContextTTSService is deprecated. Inherit from WebsocketTTSService directly "
            "and pass reuse_context_id_within_turn as kwargs.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            reuse_context_id_within_turn=reuse_context_id_within_turn,
            reconnect_on_error=reconnect_on_error,
            **kwargs,
        )


class AudioContextWordTTSService(AudioContextTTSService):
    """Deprecated. Use WebsocketTTSService directly instead.

    .. deprecated:: 0.0.105
        Subclass :class:`WebsocketTTSService` directly.
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the Audio Context Word TTS service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        import warnings

        warnings.warn(
            "AudioContextWordTTSService is deprecated. Inherit from WebsocketTTSService directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(reconnect_on_error=reconnect_on_error, **kwargs)
