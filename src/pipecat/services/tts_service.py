#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base classes for Text-to-speech services."""

import asyncio
import uuid
from abc import abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import (
    Any,
)

from loguru import logger
from websockets.protocol import State

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.audio.utils import create_stream_resampler
from pipecat.expressive.parser import EXPRESSION_TAGS, Segment
from pipecat.expressive.tts_expression import EXPRESSION_TAG_PATTERN, TTSExpression
from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EagerTranscriptionFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMAssistantPushAggregationFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    SystemFrame,
    TextFrame,
    TextType,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSMetadataFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.services.settings import TTSSettings
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer
from pipecat.utils.context.audio_context_manager import AudioContextFrame, AudioContextManager
from pipecat.utils.context.word_timestamp_tracker import WordTimestamp, WordTimestampTracker
from pipecat.utils.deprecation import deprecated, warn_deprecated
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.string import resolve_sentence_tokenizer_language
from pipecat.utils.text.base_text_aggregator import Aggregation
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.text.pattern_pair_aggregator import PatternMatch
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.text.transforms.pronunciations import (
    PronunciationTransform,
    pronunciation_transform,
)
from pipecat.utils.text.transforms.text_transform import TextTransform
from pipecat.utils.text.word_timestamp_utils import merge_punct_tokens
from pipecat.utils.types import is_given


class TextAggregationMode(StrEnum):
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
class TTSContext:
    """Context information for a TTS request.

    Parameters:
        append_to_context: Whether this TTS output should be appended to the
            conversation context after it is spoken.
        push_assistant_aggregation: Whether to push an
            ``LLMAssistantPushAggregationFrame`` after the TTS has finished
            speaking, forcing the assistant aggregator to commit its current
            text buffer to the conversation context.
        interruptible: Whether an interruption may discard this synthesis.
    """

    append_to_context: bool = True
    push_assistant_aggregation: bool = False
    interruptible: bool = True


@dataclass
class _AudioContextPlayback:
    """Internal: what has happened so far while an audio context plays."""

    received_audio: bool = False
    stop_frame_due: bool = False


# The text types an LLM writes its replies in, where expressive conversation
# reads expression tags. Other text, such as a pattern's match, is spoken as is.
_EXPRESSIVE_TEXT_TYPES = (TextType.SENTENCE, TextType.TOKEN)


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

    # Renders expressive conversation's tags for this service (alpha). The base
    # adapter performs nothing.
    expression_adapter_class: type[BaseExpressionAdapter] = BaseExpressionAdapter

    def __init__(
        self,
        *,
        text_aggregation_mode: TextAggregationMode | None = None,
        aggregate_sentences: bool | None = None,
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
        pause_watchdog_timeout_s: float | None = None,
        # number of consecutive TTS contexts that may complete with no audio before the
        # service is reported unable to do its job; 0 disables the check
        max_consecutive_zero_audio_contexts: int = 3,
        # if True, append a trailing space to text before sending to TTS
        # (helps prevent some TTS services from vocalizing trailing punctuation)
        append_trailing_space: bool = False,
        # TTS output sample rate
        sample_rate: int | None = None,
        # Types of text that should not be spoken.
        skip_text_types: list[str] | None = None,
        skip_aggregator_types: list[str] | None = None,
        # A list of TextTransforms to transform text just before sending it to TTS.
        # Each TextTransform takes the aggregated text and its type, and returns the transformed
        # text. To register, provide a list of tuples of (text_type | '*', transform_function).
        text_transforms: list[tuple[TextType | str, TextTransform]] | None = None,
        # Text filter executed after text has been aggregated. Deprecated: use
        # text_transforms instead.
        text_filters: Sequence[BaseTextFilter] | None = None,
        # Audio transport destination of the generated frames.
        transport_destination: str | None = None,
        settings: TTSSettings | None = None,
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
                    Will be removed in 2.0.0.

            push_text_frames: Whether to push TextFrames and LLMFullResponseEndFrames.
            push_stop_frames: Whether to automatically push TTSStoppedFrames.
            push_start_frame: Whether to automatically create audio contexts and push TTSStartedFrames.
                When True, the base class handles ``create_audio_context`` and yields ``TTSStartedFrame``
                before each synthesis call, so ``run_tts`` implementations do not need to.
            stop_frame_timeout_s: Idle time before pushing TTSStoppedFrame when push_stop_frames is True.
            push_silence_after_stop: Whether to push silence audio after TTSStoppedFrame.
            silence_time_s: Duration of silence to push when push_silence_after_stop is True.
            pause_frame_processing: Whether to hold incoming frames after a turn until its
                audio contexts end. Needed by services whose audio doesn't carry its
                request's context ID.
            pause_watchdog_timeout_s: Unused.

                .. deprecated:: 1.8.0
                    No replacement. Frame processing is now paused only while an
                    audio context is open, and every context ends, so no timer is
                    needed to break the pause. Will be removed in 2.0.0.

            max_consecutive_zero_audio_contexts: How many consecutive TTS contexts may
                complete without producing any audio before the service is reported unable
                to do its job. Catches a provider that accepts requests and stays silent —
                an unknown voice ID, say — which no error ever surfaces. Every silent
                context reports an error the service can carry on from, whatever the limit
                is; on reaching the limit the service reports a permanent error instead,
                stops being given work, and the pipeline worker applies its
                :class:`~pipecat.pipeline.worker.ProcessorUnusablePolicy`. Set to 0 to
                report silent contexts without ever writing the service off.
            append_trailing_space: Whether to append a trailing space to text before sending to TTS.
                This helps prevent some TTS services from vocalizing trailing punctuation (e.g., "dot").
                Only applied in sentence aggregation mode; when streaming tokens, the incoming
                text's own whitespace is preserved.
            sample_rate: Output sample rate for generated audio.
            skip_text_types: List of text types that should not be spoken.
            skip_aggregator_types: List of text types that should not be spoken.

                .. deprecated:: 1.13.0
                    Use ``skip_text_types`` instead.
                    Will be removed in 2.0.0.

            text_transforms: A list of TextTransforms to transform text before just before
                sending it to TTS. Each TextTransform takes the aggregated text and its type,
                and returns the transformed text. To register, provide a list of tuples of
                (text_type | '*', transform_function).

            text_filters: Text filters run before the transforms.

                .. deprecated:: 1.14.0
                    Use ``text_transforms`` instead, with
                    :data:`~pipecat.utils.text.transforms.strip_markdown` in place of
                    :class:`~pipecat.utils.text.markdown_text_filter.MarkdownTextFilter`.
                    Will be removed in 2.0.0.

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

        aggregation_language = resolve_sentence_tokenizer_language(
            self._settings.language if is_given(self._settings.language) else None
        )

        # Runtime updates are converted by _update_settings(), but init-time
        # settings bypass that path.
        if isinstance(self._settings.language, str):
            self._settings.language = self._service_language(self._settings.language)

        # Resolve text_aggregation_mode from the new param or deprecated aggregate_sentences
        if aggregate_sentences is not None:
            warn_deprecated(
                "`aggregate_sentences` is deprecated since 0.0.104 and will be removed in "
                "2.0.0. Use `text_aggregation_mode=TextAggregationMode.SENTENCE` or "
                "`text_aggregation_mode=TextAggregationMode.TOKEN` instead.",
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
        if pause_watchdog_timeout_s is not None:
            warn_deprecated(
                "`pause_watchdog_timeout_s` is deprecated since 1.8.0 and will be "
                "removed in 2.0.0. No replacement. Frame processing is paused only "
                "while an audio context is open, so the pause cannot outlive what it "
                "waits for.",
                stacklevel=2,
            )
        self._max_consecutive_zero_audio_contexts: int = max_consecutive_zero_audio_contexts
        self._consecutive_zero_audio_contexts: int = 0
        self._append_trailing_space: bool = append_trailing_space
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._text_aggregator = SimpleTextAggregator(
            aggregation_type=self._text_aggregation_mode,
            language=aggregation_language,
        )
        # Cuts LLM text marked skip_tts into sentences, so a reply that isn't
        # spoken still reaches clients as bot output. It's separate from the
        # aggregator above because that one yields single tokens in token mode,
        # and only spoken tokens are regrouped into sentences afterwards (by the
        # sequencer).
        self._skipped_text_aggregator = SimpleTextAggregator(
            aggregation_type=TextAggregationMode.SENTENCE,
            language=aggregation_language,
        )

        if skip_aggregator_types is not None:
            warn_deprecated(
                "`skip_aggregator_types` is deprecated since 1.13.0 and will be removed in "
                "2.0.0. Use `skip_text_types` instead.",
                stacklevel=2,
            )
            if skip_text_types is None:
                skip_text_types = skip_aggregator_types
        self._skip_text_types: list[str] = skip_text_types or []
        self._text_transforms: list[tuple[TextType | str, TextTransform]] = text_transforms or []
        # Whether pronunciation transforms are being skipped, so the warning is
        # logged once each time they start being skipped, not for every sentence.
        self._skipping_pronunciations = False
        if text_filters:
            warn_deprecated(
                "`text_filters` is deprecated since 1.14.0 and will be removed in 2.0.0. "
                "Use `text_transforms` instead.",
                stacklevel=2,
            )
        self._text_filters: Sequence[BaseTextFilter] = text_filters or []
        self._transport_destination: str | None = transport_destination

        # Ordered sequence of every AggregatedTextFrame slot that passes through
        # _handle_aggregated_text (both spoken and skipped). Skipped frames are held here
        # until all preceding spoken slots are complete, then flushed downstream so
        # their append_to_context=True arrives at the assistant aggregator in the
        # correct order relative to the TTSTextFrames from spoken sentences.
        #
        # In streaming (TOKEN) mode the sequencer regroups tokens back into
        # sentences, which relies on the whole turn sharing one context ID; this is
        # only sound with reuse_context_id_within_turn=True (see the sequencer's
        # `streaming` docstring).
        self._aggregated_frame_sequencer = AggregatedFrameSequencer(
            name=str(self), streaming=self._is_streaming_tokens
        )

        self._resampler = create_stream_resampler()

        self._tts_contexts: dict[str, TTSContext] = {}
        self._streamed_text: str = ""
        self._sent_non_whitespace_in_context: bool = False
        self._text_aggregation_metrics_started: bool = False

        # Turns provider word times into the PTS of word frames, and gives
        # TTSStoppedFrame and LLMFullResponseEndFrame the PTS of the last word.
        self._word_timestamp_tracker = WordTimestampTracker()
        self._reuse_context_id_within_turn: bool = reuse_context_id_within_turn

        # Shared by the sentences of the current turn, so they form one audio
        # context. A TTSSpeakFrame swaps in its own while it is spoken. Unlike the
        # playing context (see get_active_audio_context_id), it is cleared once
        # the turn's synthesis completes, before its audio has played.
        self._turn_context_id: str | None = None

        # Audio contexts and the downstream frames between them, emitted in order.
        # Created once so it survives interruptions, which drop interruptible
        # contexts and frames while keeping uninterruptible ones (e.g.
        # FunctionCallResultFrame) that must not be lost mid-flight.
        self._audio_context_manager = AudioContextManager()
        self._audio_context_task: asyncio.Task | None = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_connection_error")
        self._register_event_handler("on_tts_request")

        # Whether the TTS process is currently yielding audio frames synchronously.
        self._is_yielding_frames_synchronously = False

        self._expression_adapter = self.expression_adapter_class()
        # Performs the expression tags an LLM writes, when the pipeline has
        # expressive conversation on.
        self._expression: TTSExpression | None = None

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

    @property
    def text_aggregation_language(self) -> str:
        """Current sentence tokenizer language, derived from TTS settings."""
        return self._text_aggregator.language

    @property
    def supports_processing_metrics(self) -> bool:
        """Whether this service has a meaningful processing-time metric.

        Processing time is measured around :meth:`run_tts`, so it only means
        something when synthesis finishes before ``run_tts`` returns. Services
        that hand the text off and receive audio elsewhere — anything holding a
        persistent connection with its own receive task — return False, since
        the measurement would cover the send and nothing else. TTFB and TTFA
        carry the latency for those.
        """
        return True

    def get_expression_adapter(self) -> BaseExpressionAdapter:
        """The adapter that renders this service's expression tags.

        Part of expressive conversation (alpha). Its capabilities are what the
        service can perform: what it broadcasts at start, and what it reads
        and renders. A service whose expression depends on its settings, such
        as its model, overrides this to choose an adapter for them.

        Returns:
            An instance of :attr:`expression_adapter_class`.
        """
        return self._expression_adapter

    @property
    def expression_modes(self) -> list[str]:
        """The modes the request being synthesized is spoken in.

        Part of expressive conversation (alpha). Set before each request, for
        a service that takes modes with the request, such as a style
        instruction, rather than as markup in its text.

        Returns:
            The modes, empty for the voice's baseline.
        """
        return self._expression.modes if self._expression else []

    def service_metadata_frame(self) -> TTSMetadataFrame | None:
        """Build the TTS metadata frame broadcast at start.

        Overrides :meth:`AIService.service_metadata_frame`. With expressive
        conversation on, a :class:`~pipecat.frames.frames.TTSMetadataFrame`
        carries its expression adapter's capabilities; otherwise nothing is
        broadcast.

        Returns:
            The metadata frame, or None when expressive conversation is off.
        """
        if not self._expression:
            return None
        return TTSMetadataFrame(
            service_name=self.name, expression=self.get_expression_adapter().capabilities()
        )

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

    @deprecated(
        "`TTSService.set_model` is deprecated since 0.0.104 and will be removed in 2.0.0. "
        "Use `TTSUpdateSettingsFrame(model=...)` instead."
    )
    async def set_model(self, model: str):
        """Set the TTS model to use.

        .. deprecated:: 0.0.104
            Use ``TTSUpdateSettingsFrame(model=...)`` instead.
            Will be removed in 2.0.0.

        Args:
            model: The name of the TTS model.
        """
        logger.info(f"Switching TTS model to: [{model}]")
        settings_cls = type(self._settings)
        await self._update_settings(settings_cls(model=model))

    @deprecated(
        "`TTSService.set_voice` is deprecated since 0.0.104 and will be removed in 2.0.0. "
        "Use `TTSUpdateSettingsFrame(voice=...)` instead."
    )
    async def set_voice(self, voice: str):
        """Set the voice for speech synthesis.

        .. deprecated:: 0.0.104
            Use ``TTSUpdateSettingsFrame(voice=...)`` instead.
            Will be removed in 2.0.0.

        Args:
            voice: The voice identifier or name.
        """
        logger.info(f"Switching TTS voice to: [{voice}]")
        settings_cls = type(self._settings)
        await self._update_settings(settings_cls(voice=voice))

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._sample_rate = self._init_sample_rate or setup.audio_out_sample_rate
        if setup.enable_expressive_conversation:
            self._expression = TTSExpression(self.get_expression_adapter, self)
            self._aggregated_frame_sequencer.llm_markup = EXPRESSION_TAG_PATTERN
            # Each tag reaches the parser whole, however the text streams in.
            self._text_aggregator = SkipTagsAggregator(
                EXPRESSION_TAGS,
                aggregation_type=self._text_aggregation_mode,
                language=self._text_aggregator.language,
            )

    async def start(self, frame: StartFrame):
        """Start the TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._create_audio_context_task()

    async def stop(self, frame: EndFrame):
        """Stop the TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        if self._audio_context_task:
            # Playback ends once all pending contexts and frames have played.
            await self._audio_context_manager.end_playback()
            await self._audio_context_task
            self._audio_context_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        # Prompt stop of audio production. cleanup() repeats this idempotently.
        await self._stop_audio_context_task()

    async def cleanup(self):
        """Release TTS resources at teardown."""
        await super().cleanup()
        await self._stop_audio_context_task()

    async def set_usable(self, is_usable: bool):
        """Set whether this service can be given work.

        Args:
            is_usable: Whether the service can be given work.
        """
        # Whatever silence wrote the service off has been dealt with, so the
        # count of silent contexts starts over.
        if is_usable:
            self._consecutive_zero_audio_contexts = 0
        await super().set_usable(is_usable)

    #
    # Text transforms and pronunciations
    #

    def add_text_transformer(
        self,
        transform_function: TextTransform,
        text_type: TextType | str = "*",
        *,
        aggregation_type: TextType | str | None = None,
    ):
        """Transform text of a specific type.

        Args:
            transform_function: The function to apply for transformation. This function should take
                the text and its type as input and return the transformed text.
                Ex.: async def my_transform(text: str, text_type: str) -> str:
            text_type: The type of text to transform. This value defaults to "*" indicating
                the function should handle all text before sending to TTS.
            aggregation_type: The type of text to transform.

                .. deprecated:: 1.13.0
                    Use ``text_type`` instead.
                    Will be removed in 2.0.0.
        """
        if aggregation_type is not None:
            warn_deprecated(
                "`aggregation_type` is deprecated since 1.13.0 and will be removed in 2.0.0. "
                "Use `text_type` instead.",
                stacklevel=2,
            )
            text_type = aggregation_type
        self._text_transforms.append((text_type, transform_function))

    def remove_text_transformer(
        self,
        transform_function: TextTransform,
        text_type: TextType | str = "*",
        *,
        aggregation_type: TextType | str | None = None,
    ):
        """Remove a text transformer for a specific text type.

        Args:
            transform_function: The function to remove.
            text_type: The type of text to remove the transformer for.
            aggregation_type: The type of text to remove the transformer for.

                .. deprecated:: 1.13.0
                    Use ``text_type`` instead.
                    Will be removed in 2.0.0.
        """
        if aggregation_type is not None:
            warn_deprecated(
                "`aggregation_type` is deprecated since 1.13.0 and will be removed in 2.0.0. "
                "Use `text_type` instead.",
                stacklevel=2,
            )
            text_type = aggregation_type
        self._text_transforms = [
            (transform_type, func)
            for transform_type, func in self._text_transforms
            if not (transform_type == text_type and func == transform_function)
        ]

    @classmethod
    def format_pronunciation(cls, word: str, ipa: str) -> str | None:
        """Render a word's pronunciation in this service's markup.

        Services that support pronunciation hints override this. The base
        implementation supports none.

        Args:
            word: The word as it appears in the text.
            ipa: How to pronounce it, in IPA.

        Returns:
            The text to send in place of ``word``, or None when this service cannot
            use the pronunciation.
        """
        return None

    @property
    def supports_pronunciations(self) -> bool:
        """Whether this service reads pronunciation markup with its current settings.

        Pronunciation transforms are skipped while this is False, so their words
        are spoken as written. Services whose markup depends on the model or on a
        setting override this; it is checked for every text sent, so it follows
        settings updates.

        Returns:
            True in the base implementation.
        """
        return True

    @classmethod
    def pronunciation_transform_ipa(
        cls, pronunciations: Mapping[str, str]
    ) -> PronunciationTransform:
        """Create a text transform that makes this service say words as IPA describes.

        Each word is replaced with :meth:`format_pronunciation` output, so the
        same IPA works with any service that supports it. Words the service
        cannot use are reported once and spoken as written. While the service
        cannot read pronunciation markup at all (see
        :attr:`supports_pronunciations`), the transform is skipped.

        Register this transform last in ``text_transforms``. Transforms run in
        order, and one that runs after it (stripping markdown or symbols,
        replacing text) can rewrite the markup it inserts, such as Cartesia's
        ``<<…>>`` blocks or an SSML ``<phoneme>`` tag, and break the hint.

        Args:
            pronunciations: Word to IPA, e.g. ``{"Metformin": "mɛtˈfɔɹmɪn"}``.

        Returns:
            A transform to register with ``text_transforms``.

        Example::

            pronounce = CartesiaTTSService.pronunciation_transform_ipa(
                {"Metformin": "mɛtˈfɔɹmɪn"}
            )
            tts = CartesiaTTSService(
                text_transforms=[
                    ("*", strip_markdown),
                    ("*", expand_currency),
                    ("*", pronounce),  # last, so nothing rewrites its markup
                ],
            )
        """
        return pronunciation_transform(
            pronunciations, cls.format_pronunciation, service_name=cls.__name__
        )

    #
    # Subclass interface
    #

    @abstractmethod
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Run text-to-speech synthesis on the provided text.

        This method must be implemented by subclasses to provide actual TTS functionality.
        The base class logs the synthesized text before invoking this method, so
        implementations should not log it again.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        raise NotImplementedError
        yield  # pragma: no cover

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a language to the service-specific language format.

        Args:
            language: The language to convert.

        Returns:
            The service-specific language identifier. Return ``None`` to
            indicate an unsupported language. This optional return is an
            extension hook for future or third-party subclasses; as of
            2026-04-28, first-party services return a string.
        """
        return Language(language)

    def create_context_id(self) -> str:
        """Generate or reuse a context ID based on concurrent TTS support.

        Returns:
            A context ID string for the TTS request.
        """
        if self._reuse_context_id_within_turn and self._turn_context_id:
            self._refresh_audio_context(self._turn_context_id)
            context_id = self._turn_context_id
        else:
            context_id = str(uuid.uuid4())
        return context_id

    async def flush_audio(self, context_id: str | None = None):
        """Flush any buffered audio data.

        Args:
            context_id: The specific context to flush. If None, falls back to the
                currently active context (for non-concurrent services).
        """
        pass

    async def tts_process_generator(
        self, context_id: str, generator: AsyncGenerator[Frame | None, None]
    ):
        """Process frames from an async generator, routing them through the audio context.

        All non-None frames yielded by the generator are appended to the audio context
        identified by context_id. The audio context must be created by run_tts (via
        create_audio_context) before the first frame is yielded.

        WebSocket services yield None to signal that audio will arrive via a separate
        receive loop; those services manage context lifetime themselves (via remove_audio_context
        in the receive loop on "done"). HTTP services never yield None and do NOT call
        remove_audio_context in run_tts — on_turn_context_completed closes the context
        once the turn's remaining frames (e.g. TTSTextFrame) are queued.

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
        # HTTP services yield their audio inline, so their audio context is closed
        # here, once all the turn's frames (including TTSTextFrame) are queued.
        if (
            self._is_yielding_frames_synchronously
            and self._turn_context_id is not None
            and self.audio_context_available(self._turn_context_id)
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

    async def on_audio_context_interrupted(self, context_id: str):
        """Called when an audio context is cancelled due to an interruption.

        Override this in a subclass to perform provider-specific cleanup (e.g.
        sending a cancel/close message over the WebSocket) when the bot is
        interrupted mid-speech. The context is still registered when this hook
        runs. Uninterruptible contexts continue playing and do not call this hook.

        Args:
            context_id: The ID of the audio context that was interrupted.
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

    #
    # Word timestamps
    #

    async def start_word_timestamps(self):
        """Start tracking word timestamps from the current time."""
        for cached in self._word_timestamp_tracker.start(self.get_clock().get_time()):
            await self._add_word_timestamps(
                [(cached.word, cached.timestamp)],
                cached.context_id,
                includes_inter_frame_spaces=cached.includes_inter_frame_spaces,
            )

    async def reset_word_timestamps(self):
        """Reset word timestamp tracking."""
        self._word_timestamp_tracker.reset()

    async def add_word_timestamps(
        self,
        word_times: list[tuple[str, float]],
        context_id: str | None = None,
        includes_inter_frame_spaces: bool | None = None,
        pre_merge_tokens: bool = False,
    ):
        """Add word timestamps for processing.

        When an audio context exists for this context_id, timestamps are routed into the
        per-context audio queue alongside audio frames so they are processed in strict
        playback order by _handle_audio_context. Otherwise they are processed immediately
        via _add_word_timestamps.

        Args:
            word_times: List of (word, timestamp) tuples where timestamp is in seconds.
            context_id: Unique identifier for the TTS context.
            includes_inter_frame_spaces: When True, each emitted TTSTextFrame carries
                ``includes_inter_frame_spaces=True`` so downstream consumers do not inject
                extra spaces between consecutive frames (e.g. ElevenLabs for CJK languages
                where word tokens already self-contain their spacing). None leaves the
                frame's own default unchanged.
            pre_merge_tokens: When True, punctuation and space-only tokens are merged into
                the preceding word via ``merge_punct_tokens`` before queuing (e.g. Inworld
                TTS, which emits spaces and punctuation as separate tokens). After merging
                the resulting tokens are clean word strings, so ``includes_inter_frame_spaces``
                should be left False (its default).
        """
        if pre_merge_tokens:
            word_times = merge_punct_tokens(word_times)

        ifs = bool(includes_inter_frame_spaces)
        if context_id and self.audio_context_available(context_id):
            for word, timestamp in word_times:
                await self.append_to_audio_context(
                    context_id,
                    WordTimestamp(
                        word=word,
                        timestamp=timestamp,
                        context_id=context_id,
                        includes_inter_frame_spaces=ifs,
                    ),
                )
        elif context_id is not None:
            logger.trace(f"Dropping stale words for context {context_id}; word times: {word_times}")
        else:
            await self._add_word_timestamps(
                word_times=word_times, context_id=context_id, includes_inter_frame_spaces=ifs
            )

    #
    # Audio contexts
    #

    async def create_audio_context(self, context_id: str):
        """Create a new audio context for grouping related audio.

        Args:
            context_id: Unique identifier for the audio context.
        """
        context = self._tts_contexts.get(context_id)
        interruptible = context.interruptible if context else True
        await self._audio_context_manager.create(context_id, interruptible=interruptible)
        logger.trace(f"{self} created audio context {context_id}")

    async def append_to_audio_context(
        self, context_id: str | None, frame: Frame | WordTimestamp | None
    ):
        """Append a frame or word-timestamp entry to an existing audio context queue.

        Passing a ``frame`` of ``None`` signals end-of-context (used by remove_audio_context
        to mark the queue for deletion). If the context no longer exists but the context_id
        matches the active turn, the context is transparently recreated before appending.

        Args:
            context_id: The context to append to. ``None`` is accepted as a no-op
                (with a debug log) so callers can pass through values from
                ``get_active_audio_context_id()`` without an explicit guard.
            frame: The frame, word-timestamp entry, or ``None`` (end-of-context sentinel)
                to append.
        """
        if not context_id:
            logger.debug(f"{self} unable to append audio to context: no context ID provided")
            return
        # Sometimes the HTTP service can take more than 3 seconds without sending
        # any audio, so the context is recreated while we are in the same turn.
        # An end marker (None) never recreates it.
        if (
            not self.audio_context_available(context_id)
            and context_id == self._turn_context_id
            and frame
        ):
            logger.debug(f"{self} recreating audio context {context_id}")
            await self.create_audio_context(context_id)

        if self.audio_context_available(context_id):
            logger.trace(f"{self} appending audio {frame} to audio context {context_id}")
            await self._audio_context_manager.append(context_id, frame)
        else:
            logger.debug(f"{self} unable to append audio to context {context_id}")

    async def remove_audio_context(self, context_id: str | None):
        """Remove an existing audio context.

        Args:
            context_id: The context to remove. ``None`` is accepted as a
                no-op (logged) so callers can pass through values from
                ``get_active_audio_context_id()`` without an explicit guard.
        """
        if not context_id:
            logger.debug(f"{self} unable to remove audio context: no context ID provided")
            return
        if self.audio_context_available(context_id):
            # We just mark the audio context for deletion by appending
            # None. Once we reach None while handling audio we know we can
            # safely remove the context.
            logger.trace(f"{self} marking audio context {context_id} for deletion")
            await self.append_to_audio_context(context_id, None)
        else:
            logger.warning(f"{self} unable to remove context {context_id}")

    def audio_context_available(self, context_id: str) -> bool:
        """Check whether the given audio context is registered.

        Args:
            context_id: The context ID to check.

        Returns:
            True if the context exists and is available.
        """
        return self._audio_context_manager.available(context_id)

    def get_audio_contexts(self) -> list[str]:
        """Get a list of all available audio contexts."""
        return self._audio_context_manager.context_ids

    def has_active_audio_context(self) -> bool:
        """Check if there is an active audio context.

        Returns:
            True if an active audio context exists, False otherwise.
        """
        playing_context_id = self._audio_context_manager.playing_context_id
        return playing_context_id is not None and self.audio_context_available(playing_context_id)

    def get_active_audio_context_id(self) -> str | None:
        """Get the active audio context ID.

        Returns the playback cursor when set (during active playback), falling
        back to the current turn's synthesis context_id. The fallback covers
        the gap between contexts and the start of a turn before the playback
        task has dequeued the just-created context —
        important for services whose wire protocol does not echo context_id
        back on incoming audio.

        Returns:
            The active context ID, or None if neither cursor is set.
        """
        return self._audio_context_manager.playing_context_id or self._turn_context_id

    async def remove_active_audio_context(self):
        """Remove the active audio context."""
        playing_context_id = self._audio_context_manager.playing_context_id
        if playing_context_id:
            await self.remove_audio_context(playing_context_id)
            self.reset_active_audio_context()

    def reset_active_audio_context(self):
        """Reset the active audio context."""
        self._audio_context_manager.playing_context_id = None

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for text-to-speech conversion.

        Handles TextFrames for synthesis, interruption frames, settings updates,
        and various control frames.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        # A reply that isn't spoken is queued behind the audio still playing, so
        # it comes out after that audio, its start, sentences and end in order.
        if isinstance(frame, LLMFullResponseStartFrame) and frame.skip_tts:
            await self._audio_context_manager.put_frame(frame)
        elif isinstance(frame, LLMTextFrame) and frame.skip_tts:
            await self._handle_skipped_llm_text(frame)
        elif isinstance(frame, LLMFullResponseEndFrame) and frame.skip_tts:
            await self._handle_skipped_llm_response_end(frame)
        elif isinstance(frame, TextFrame) and frame.skip_tts:
            await self.push_frame(frame, direction)
        elif isinstance(frame, AggregatedTextFrame):
            await self._handle_aggregated_text(frame)
        elif (
            isinstance(frame, TextFrame)
            and not isinstance(frame, InterimTranscriptionFrame)
            and not isinstance(frame, TranscriptionFrame)
            and not isinstance(frame, EagerTranscriptionFrame)
        ):
            await self.start_text_aggregation_metrics()
            await self._handle_text(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruption(frame, direction)
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_llm_response_start(frame)
        elif isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self._handle_llm_response_end(frame, direction)
        elif isinstance(frame, TTSSpeakFrame):
            await self._handle_tts_speak(frame)
        elif isinstance(frame, TTSUpdateSettingsFrame):
            await self._handle_update_settings(frame, direction)
        else:
            if direction == FrameDirection.DOWNSTREAM and not isinstance(frame, SystemFrame):
                # Queue non-system downstream frames behind the audio contexts so they
                # are emitted in the same order they arrive relative to any audio contexts that
                # are already queued (e.g. a FooFrame sent right after a TTSSpeakFrame must
                # not overtake the TTSStartedFrame / TTSAudioRawFrame / TTSStoppedFrame
                # sequence from that speak frame).
                await self._audio_context_manager.put_frame(frame)
            else:
                await self.push_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream with TTS-specific handling.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        self._apply_context_interruptible(frame)
        if isinstance(frame, TTSStoppedFrame):
            await self._finish_tts_context(frame)
            await self._push_stop_silence(frame)
            if self._expression:
                self._expression.forget(frame.context_id)
        if isinstance(frame, (TTSStartedFrame, TTSStoppedFrame, TTSAudioRawFrame, TTSTextFrame)):
            frame.transport_destination = self._transport_destination
        if isinstance(frame, AggregatedTextFrame):
            self._stamp_segment_announcement(frame)
        if self._expression:
            events, keep = self._report_expression(frame)
            for event in events:
                await self.push_frame(event, direction)
            if not keep:
                return
        await super().push_frame(frame, direction)

    def _report_expression(self, frame: Frame) -> tuple[list[TTSTextFrame], bool]:
        """Give spoken text the modes it is spoken in, and report the events before it.

        The text the service reports speaking covers part of the text the LLM
        wrote, tags included, which tells its modes and the events performed
        since the text before it. Markup the service echoes as a word stands
        for the event it performs, or isn't spoken.

        Returns:
            A frame for each event or pause performed before the text, and
            whether the frame itself is pushed.
        """
        assert self._expression
        if isinstance(frame, AggregatedTextProgressFrame):
            frame.modes = self._expression.current_modes(frame.context_id)
            return [], True
        if not isinstance(frame, TTSTextFrame) or frame.text_type in (
            TextType.EXPRESSION_EVENT,
            TextType.EXPRESSION_PAUSE,
        ):
            return [], True
        if frame.raw_text is None and _is_markup(frame.text):
            echoed = self._expression.echoed(frame.context_id)
            if echoed is None:
                return [], False
            frame.text_type, frame.text = echoed
            frame.append_to_context = False
            return [], True
        modes, events = self._expression.spoken(frame.context_id, frame.raw_text)
        if modes is not None:
            frame.modes = modes
        reported = []
        for text_type, text in events:
            event = TTSTextFrame(text, text_type=text_type)
            event.pts = frame.pts
            event.context_id = frame.context_id
            event.will_be_spoken = True
            # The context records the event's tag with the text around it.
            event.append_to_context = False
            reported.append(event)
        return reported, True

    #
    # Settings
    #

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a TTS settings delta.

        Translates language to service-specific value before applying.

        Args:
            delta: A TTS settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        language = delta.language
        # Translate language *before* applying so the stored value is canonical.
        if is_given(delta.language) and isinstance(delta.language, str):
            delta.language = self._service_language(delta.language)

        changed = await super()._update_settings(delta)

        if is_given(language):
            self._text_aggregator.set_language(language)
            self._skipped_text_aggregator.set_language(language)

        return changed

    def _service_language(self, language: str) -> str:
        """Convert a language to the service's format, passing unknown codes as they are."""
        # Raw strings (e.g. "de-DE") are first converted to Language enums so
        # they go through the same resolution logic.
        if not isinstance(language, Language):
            try:
                language = Language(language)
            except ValueError:
                logger.debug(
                    f"Language string '{language}' is not a recognized "
                    f"Language code. It will be passed to the service as-is."
                )
        if isinstance(language, Language):
            converted = self.language_to_service_language(language)
            if converted is not None:
                language = converted
        return language

    #
    # Frame handling
    #

    async def _handle_llm_response_start(self, frame: LLMFullResponseStartFrame):
        """Start a turn whose sentences share one context ID."""
        self._turn_context_id = self.create_context_id()
        await self.on_turn_context_created(self._turn_context_id)
        # Queue behind the audio contexts so this frame is emitted only
        # after any earlier audio context already queued has fully drained,
        # rather than racing ahead of it (e.g. a new turn starting via
        # LLMMessagesAppendFrame while the previous turn's audio is still
        # playing out downstream).
        await self._audio_context_manager.put_frame(frame)

    async def _handle_llm_response_end(
        self, frame: LLMFullResponseEndFrame | EndFrame, direction: FrameDirection
    ):
        """Speak the turn's remaining text and complete the turn."""
        # Flush any remaining text (including text waiting for lookahead)
        remaining = await self._text_aggregator.flush()
        # Stop the aggregation metric (no-op if already stopped on first sentence).
        await self.stop_text_aggregation_metrics()
        if remaining:
            await self._handle_aggregated_text(self._aggregated_text_frame(remaining))
        if self._expression:
            self._expression.end_response()

        # Force-promote any sentence still pending in the sequencer (streaming
        # mode only; a no-op otherwise) — handles a response that ends with no
        # terminal punctuation.
        await self._push_sequencer_frames(
            await self._aggregated_frame_sequencer.finalize(self._turn_context_id),
            self._turn_context_id,
        )

        await self._hold_frames_while_audio_contexts_open()

        # Log accumulated streamed text and emit aggregated usage metric.
        if self._streamed_text:
            logger.debug(f"{self}: Generating TTS [{self._streamed_text}]")
            await super().start_tts_usage_metrics(self._streamed_text)
            self._streamed_text = ""

        # Reset aggregator state
        self._sent_non_whitespace_in_context = False
        # A filter's state belongs to one reply, so it's cleared as an
        # interruption clears it.
        for filter in self._text_filters:
            await filter.handle_interruption()
        if isinstance(frame, LLMFullResponseEndFrame):
            # Queue behind the audio contexts so the frame follows the turn's
            # audio and text.
            await self._audio_context_manager.put_frame(frame)
        else:
            await self.push_frame(frame, direction)

        await self.on_turn_context_completed()

    async def _handle_skipped_llm_text(self, frame: LLMTextFrame):
        """Queue each sentence of LLM text that isn't spoken."""
        async for aggregate in self._skipped_text_aggregator.aggregate(frame.text):
            await self._queue_skipped_llm_text(aggregate)

    async def _handle_skipped_llm_response_end(self, frame: LLMFullResponseEndFrame):
        """Queue the rest of a reply that isn't spoken, then the end of the reply."""
        remaining = await self._skipped_text_aggregator.flush()
        if remaining:
            await self._queue_skipped_llm_text(remaining)
        await self._audio_context_manager.put_frame(frame)

    async def _queue_skipped_llm_text(self, aggregation: Aggregation):
        """Queue a sentence of LLM text that isn't spoken, as the bot's output."""
        if not aggregation.text.strip():
            return
        frame = self._aggregated_text_frame(aggregation)
        frame.skip_tts = True
        frame.transport_destination = self._transport_destination
        await self._audio_context_manager.put_frame(frame)

    async def _handle_tts_speak(self, frame: TTSSpeakFrame):
        """Speak a TTSSpeakFrame as a turn of its own, even in the middle of an LLM turn."""
        saved_sent_non_whitespace = self._sent_non_whitespace_in_context
        self._sent_non_whitespace_in_context = False
        # TTSSpeakFrame is independent — temporarily clear the turn context
        # so create_context_id() generates a fresh UUID for this utterance.
        saved_turn_context_id = self._turn_context_id
        self._turn_context_id = None
        # Creating a new context_id for the TTS request.
        speak_context_id = self.create_context_id()
        self._turn_context_id = speak_context_id
        # Inside an LLM turn, the end of the turn commits the spoken text to the
        # context. Outside one, the speak frame commits it once spoken.
        self._tts_contexts[speak_context_id] = TTSContext(
            append_to_context=frame.append_to_context,
            push_assistant_aggregation=frame.append_to_context and not saved_turn_context_id,
            interruptible=frame.interruptible,
        )
        await self.on_turn_context_created(speak_context_id)
        text_frame = AggregatedTextFrame(frame.text, frame.text_type, raw_text=frame.text)
        text_frame.interruptible = frame.interruptible
        if self._expression and frame.text_type in _EXPRESSIVE_TEXT_TYPES:
            # Text of its own, apart from any response it is spoken in the middle of.
            self._expression.restate_modes()
            await self._speak_expression(text_frame, speak_context_id, whole=True)
            self._expression.restate_modes()
        else:
            await self._handle_aggregated_text(text_frame, speak_context_id)
        # Force-promote the utterance's pending sentence into a real slot (streaming
        # mode only; a no-op otherwise). A TTSSpeakFrame is a self-contained sentence
        # with no following token to confirm its boundary, so without this it would
        # stay buffered and never be spoken.
        await self._push_sequencer_frames(
            await self._aggregated_frame_sequencer.finalize(self._turn_context_id),
            self._turn_context_id,
        )
        await self.on_turn_context_completed()
        if not self.audio_context_available(speak_context_id):
            self._tts_contexts.pop(speak_context_id, None)
        self._turn_context_id = saved_turn_context_id
        self._sent_non_whitespace_in_context = saved_sent_non_whitespace
        # Inside an LLM turn, the end of the turn holds the next frames.
        if not saved_turn_context_id:
            await self._hold_frames_while_audio_contexts_open()

    async def _handle_update_settings(
        self, frame: TTSUpdateSettingsFrame, direction: FrameDirection
    ):
        """Apply a settings update meant for this service, or pass it on."""
        if frame.service is not None and frame.service is not self:
            await self.push_frame(frame, direction)
        elif frame.delta is not None:
            await self._update_settings(frame.delta)
        elif frame.settings:
            # Backward-compatible path: convert legacy dict to settings object.
            warn_deprecated(
                "`TTSUpdateSettingsFrame(settings={...})` is deprecated since 0.0.104 and "
                "will be removed in 2.0.0. Use "
                "`TTSUpdateSettingsFrame(delta=TTSSettings(...))` instead.",
                stacklevel=2,
            )
            delta = type(self._settings).from_mapping(frame.settings)
            await self._update_settings(delta)

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Reset the turn and drop interruptible audio contexts."""
        self._sent_non_whitespace_in_context = False
        self._streamed_text = ""
        self._text_aggregation_metrics_started = False

        await self._text_aggregator.handle_interruption()
        await self._skipped_text_aggregator.handle_interruption()
        if self._expression:
            self._expression.reset()
        for filter in self._text_filters:
            await filter.handle_interruption()

        await self._interrupt_audio_contexts()
        self._create_audio_context_task()
        # Uninterruptible speech still open holds frames, even when the reply
        # it was part of never ended to hold them.
        await self._hold_frames_while_audio_contexts_open()
        # Dropped contexts never reach _end_audio_context, so frames held for
        # them are released here.
        await self._release_frames_if_audio_contexts_ended()

    async def _interrupt_audio_contexts(self):
        """Discard interruptible contexts while preserving ongoing synthesis and playback."""
        preserved = {
            key for key, context in self._tts_contexts.items() if not context.interruptible
        }
        interrupt_playing = self._audio_context_manager.playing_context_id not in preserved
        if interrupt_playing:
            await self._stop_audio_context_task()
            await self.reset_word_timestamps()
            # Next words follow the uninterruptible words still to play.
            self._word_timestamp_tracker.last_pts = (
                self._word_timestamp_tracker.uninterruptible_last_pts
            )

        self._audio_context_manager.reset()
        self._aggregated_frame_sequencer.clear(keep_contexts=preserved)

        for context_id in self.get_audio_contexts():
            if context_id in preserved:
                continue
            # A failing provider hook must not prevent cleanup or task restart.
            try:
                await self.on_audio_context_interrupted(context_id=context_id)
            except Exception as e:
                logger.warning(f"{self} on_audio_context_interrupted failed for {context_id}: {e}")
            self._audio_context_manager.remove(context_id)
            self._tts_contexts.pop(context_id, None)

        if interrupt_playing:
            self.reset_active_audio_context()
        if self._turn_context_id not in preserved:
            self._turn_context_id = None

    async def _hold_frames_while_audio_contexts_open(self):
        """Hold incoming frames while an audio context is open.

        Services whose audio doesn't carry its request's context ID label
        incoming audio with the context still open, so the next request waits
        for every open context to end. Contexts play in order, so audio from
        different requests never overlaps.
        """
        if self._pause_frame_processing and self.get_audio_contexts():
            await self.pause_processing_frames()

    async def _release_frames_if_audio_contexts_ended(self):
        """Let held frames through once no audio context is open."""
        if self._pause_frame_processing and not self.get_audio_contexts():
            await self.resume_processing_frames()

    def _apply_context_interruptible(self, frame: Frame):
        """Make a frame of an uninterruptible context uninterruptible too."""
        context_id = getattr(frame, "context_id", None)
        context = self._tts_contexts.get(context_id) if context_id is not None else None
        if context:
            frame.interruptible = frame.interruptible and context.interruptible
        if isinstance(frame, TTSTextFrame) and frame.pts and not frame.interruptible:
            self._word_timestamp_tracker.uninterruptible_last_pts = frame.pts

    async def _finish_tts_context(self, frame: TTSStoppedFrame):
        """Forget a stopped context and push the assistant aggregation it asked for."""
        context = self._tts_contexts.pop(frame.context_id, None) if frame.context_id else None
        if not context:
            return
        logger.debug(f"{self} cleaning up TTS context {frame.context_id}")
        if context.push_assistant_aggregation:
            aggregation_frame = LLMAssistantPushAggregationFrame()
            aggregation_frame.interruptible = frame.interruptible
            # When word-level TTSTextFrames are routed through the transport's clock
            # queue (PTS-based), the aggregation frame would otherwise take the audio
            # (sync) queue path and could overtake the final word frames. Stamping it
            # with a PTS just past the last word forces it through the clock queue
            # too, so the assistant aggregator sees every word before flushing.
            if self._word_timestamp_tracker.last_pts:
                aggregation_frame.pts = self._word_timestamp_tracker.last_pts + 1
            await self.push_frame(aggregation_frame)

    async def _push_stop_silence(self, frame: TTSStoppedFrame):
        """Push the silence configured to follow a TTSStoppedFrame."""
        if not self._push_silence_after_stop:
            return
        silence_num_bytes = int(self._silence_time_s * self.sample_rate * 2)  # 16-bit
        silence_frame = TTSAudioRawFrame(
            audio=b"\x00" * silence_num_bytes,
            sample_rate=self.sample_rate,
            num_channels=1,
        )
        silence_frame.interruptible = frame.interruptible
        await self.push_frame(silence_frame)

    def _stamp_segment_announcement(self, frame: AggregatedTextFrame):
        """Send a segment's will_be_spoken frame through the transport's clock queue."""
        # The will_be_spoken AggregatedTextFrame is the "new segment" announcement whose
        # segment_id the per-word progress frames reference. Those progress frames carry a
        # PTS and travel the transport's clock queue; the announcement itself has no PTS and
        # would take the audio (sync) queue, so for a context whose audio is delayed behind
        # another context's, its clock-queued progress can be delivered before it. Stamp it
        # with the same baseline the first word will use so it rides the clock queue too,
        # sorted immediately before its first progress frame (ties broken by push order).
        # Only on the word-timestamp path (push_text_frames services have no progress frames).
        if frame.will_be_spoken and frame.pts is None and not self._push_text_frames:
            frame.pts = max(self._word_timestamp_tracker.last_pts, self.get_clock().get_time())

    #
    # Text to speech
    #

    @property
    def _is_streaming_tokens(self) -> bool:
        """Whether the service is streaming tokens directly without sentence aggregation."""
        return self._text_aggregation_mode == TextAggregationMode.TOKEN

    async def _handle_text(self, frame: TextFrame):
        """Aggregate LLM text and speak each aggregation."""
        async for aggregate in self._text_aggregator.aggregate(frame.text):
            if aggregate.type != TextType.TOKEN:
                # Stop the aggregation metric on the first sentence only.
                await self.stop_text_aggregation_metrics()
            await self._handle_aggregated_text(self._aggregated_text_frame(aggregate))

    def _aggregated_text_frame(self, aggregation: Aggregation) -> AggregatedTextFrame:
        """Build the frame for an aggregation, keeping a pattern's delimiters in its raw text."""
        raw_text = (
            aggregation.full_match if isinstance(aggregation, PatternMatch) else aggregation.text
        )
        return AggregatedTextFrame(aggregation.text, aggregation.type, raw_text=raw_text)

    async def _handle_aggregated_text(
        self, src_frame: AggregatedTextFrame, context_id: str | None = None
    ):
        """Speak aggregated text in ``context_id`` or a new context, or push it on if skipped."""
        context_id = context_id or self.create_context_id()
        if src_frame.text_type in self._skip_text_types:
            await self._push_skipped_text(src_frame, context_id)
        elif self._expression and src_frame.text_type in _EXPRESSIVE_TEXT_TYPES:
            await self._speak_expression(src_frame, context_id)
        elif (text := self._trim_text(src_frame.text)) is not None:
            await self._speak_text(src_frame, text, context_id)

    async def _speak_expression(
        self, src_frame: AggregatedTextFrame, context_id: str, *, whole: bool = False
    ):
        """Speak a unit of text with expression tags in it.

        The unit stays one frame: its words without the tags, for clients, and
        its text as the LLM wrote it, for the context. The request is built
        from its segments.

        Args:
            src_frame: The unit.
            context_id: The context to speak it in.
            whole: Whether the text stands on its own, apart from the response,
                such as a ``TTSSpeakFrame``'s.
        """
        if not self._expression:
            return
        text = src_frame.text
        if src_frame.text_type != TextType.TOKEN:
            text = text.strip()
        segments = self._expression.read_whole(text) if whole else self._expression.parse(text)
        if not segments:
            return
        frame = AggregatedTextFrame(
            self._expression.shown_text(segments, src_frame.text_type),
            src_frame.text_type,
            raw_text=text,
        )
        frame.interruptible = src_frame.interruptible
        await self._speak_text(frame, self._trim_text(frame.text) or "", context_id, segments)

    async def _push_skipped_text(self, frame: AggregatedTextFrame, context_id: str):
        """Push text that isn't spoken, in order with the spoken text around it."""
        # Marked so whatever is downstream knows the TTS left it out.
        frame.skip_tts = True
        # The sequencer returns the frame at once if no spoken slot precedes it,
        # or holds it until it can flush it in order.
        await self._push_sequencer_frames(
            await self._aggregated_frame_sequencer.register_skipped(
                frame, context_id, self._transport_destination
            ),
            context_id,
        )

    async def _speak_text(
        self,
        src_frame: AggregatedTextFrame,
        text: str,
        context_id: str,
        segments: list[Segment] | None = None,
    ):
        """Filter, transform and synthesize text, timing it as processing.

        With expressive conversation on, ``segments`` are the unit's words and
        tags, and the request is built from them: the words transformed, and
        the tags rendered in the service's markup.
        """
        # Accumulate text for a single debug log at flush time when streaming tokens.
        if self._is_streaming_tokens:
            self._streamed_text += text

        # Two things disqualify the measurement. A service that hands text off
        # for synthesis elsewhere would time only the handoff, and a token is
        # the wrong unit to report a processing time against — one metric per
        # token, for work that spans a sentence. TTFB and TTFA carry the timing
        # that matters in both cases.
        if self.supports_processing_metrics and not self._is_streaming_tokens:
            await self.start_processing_metrics()

        if segments is not None and self._expression:
            prepared_text = await self._expression.request(
                segments,
                src_frame.raw_text or "",
                context_id,
                src_frame.text_type,
                self._transform_text,
                self._trim_text,
            )
        else:
            prepared_text = await self._transform_text(text, src_frame.text_type)
        if prepared_text:
            # A TTSSpeakFrame sets up its context before its text gets here.
            context = self._tts_contexts.setdefault(
                context_id, TTSContext(interruptible=src_frame.interruptible)
            )
            await self._announce_text(src_frame, context_id)
            await self._call_event_handler("on_tts_request", context_id, prepared_text)
            await self._start_audio_context(context_id)
            await self._register_spoken_text(src_frame, prepared_text, context_id)
            await self._synthesize_text(prepared_text, context_id)
            await self._push_spoken_text(src_frame, text, context_id, context.append_to_context)

        await self.stop_processing_metrics()

    def _trim_text(self, text: str) -> str | None:
        """Trim leading whitespace from text to synthesize, or None if nothing is left to say."""
        # Whitespace gating depends on aggregation mode:
        # - Token streaming: drop all leading whitespace at the start of a context, as
        #   nothing substantive has been sent yet for it to attach to. Once a non-whitespace
        #   token has been sent, send whitespace as-is since it can influence prosody between
        #   non-whitespace tokens.
        #
        # - Sentence aggregation: strip leading newlines only and drop pure-whitespace frames.
        if self._is_streaming_tokens:
            if not self._sent_non_whitespace_in_context:
                text = text.lstrip()
            has_text = bool(text)
        else:
            text = text.lstrip("\n")
            has_text = bool(text.strip())
        return text if has_text else None

    async def _transform_text(self, text: str, text_type: str) -> str | None:
        """Turn text to speak into the text sent to the TTS.

        Filters and transforms only affect the text sent to the TTS, for
        TTS-specific purposes (e.g. inserting tags for spelling or emotion, or
        replacing an @ with "at"). The context keeps the text as it arrived.

        Returns:
            The text to synthesize, or None if nothing is left to say or a
            transform failed.
        """
        transformed_text = text
        try:
            for filter in self._text_filters:
                await filter.reset_interruption()
                transformed_text = await filter.filter(transformed_text)
            for transform_type, transform in self._text_transforms:
                if transform_type in (text_type, "*") and (
                    not isinstance(transform, PronunciationTransform)
                    or self._can_apply_pronunciations()
                ):
                    transformed_text = await transform(transformed_text, text_type)
        except Exception as e:
            # Pushing the error with category "APPLICATION" since the failure came
            # from the application's text transformer. Speaking the untransformed
            # text isn't safe — a transformer may exist to remove something — so this
            # turn produces no audio.
            await self.push_error(
                error_msg=f"Error transforming text for TTS [{transformed_text}]: {e}",
                exception=e,
                category=ErrorCategory.APPLICATION,
            )
            return None

        # Mirrors the whitespace gate in _trim_text, so output that collapses to
        # whitespace-only is handled consistently. When streaming tokens,
        # whitespace is only sent once something else has been.
        if self._is_streaming_tokens:
            has_text = bool(transformed_text.strip()) or (
                bool(transformed_text) and self._sent_non_whitespace_in_context
            )
            if has_text:
                self._sent_non_whitespace_in_context = True
        else:
            has_text = bool(transformed_text.strip())
        if not has_text:
            return None
        # Apply any final text preparation (e.g., trailing space)
        return self._prepare_text_for_tts(transformed_text)

    def _can_apply_pronunciations(self) -> bool:
        """Whether pronunciation transforms apply, warning once when they stop applying."""
        supported = self.supports_pronunciations
        if not supported and not self._skipping_pronunciations:
            logger.warning(
                f"{self} does not read pronunciation markup with its current settings "
                f"(model {self._settings.model}), so pronunciation hints are spoken as written"
            )
        self._skipping_pronunciations = not supported
        return supported

    def _prepare_text_for_tts(self, text: str) -> str:
        """Prepare text for TTS by applying any transformations required by the TTS service.

        Args:
            text: The text to prepare.

        Returns:
            The prepared text with transformations applied.
        """
        # A trailing space only makes sense on sentence-sized inputs. When
        # streaming tokens, the incoming text's own whitespace is authoritative:
        # appending a space to every token would split words across tokens.
        if self._append_trailing_space and not self._is_streaming_tokens and not text.endswith(" "):
            return text + " "
        return text

    async def _announce_text(self, src_frame: AggregatedTextFrame, context_id: str):
        """Mark text as about to be spoken and push it ahead of its audio."""
        # To support use cases that may want to know the text before it's spoken, we
        # push the AggregatedTextFrame version before sending it to TTS. However, we
        # do not want to add this text to the assistant context until it is spoken,
        # so we set append_to_context to False.
        src_frame.append_to_context = False
        src_frame.context_id = context_id
        src_frame.will_be_spoken = True

        # In streaming (TOKEN) mode this src_frame is a single token, not the unit the
        # listener cares about: the sequencer regroups tokens into a sentence and emits
        # that sentence as the will_be_spoken anchor (see _promote). Pushing src_frame
        # too would emit a second, redundant anchor (for a TTSSpeakFrame, an exact
        # duplicate). So only the non-streaming path pushes src_frame, queued behind
        # the audio contexts so it is emitted immediately before the TTSStartedFrame
        # of the audio context it describes, or into that context if it exists.
        if not self._is_streaming_tokens:
            if self.audio_context_available(context_id):
                await self.append_to_audio_context(context_id, src_frame)
            else:
                await self._audio_context_manager.put_frame(src_frame)

    async def _start_audio_context(self, context_id: str):
        """Open an audio context with a TTSStartedFrame, for services that leave it to the base."""
        if not self._push_start_frame or self.audio_context_available(context_id):
            return
        await self.create_audio_context(context_id)
        await self.start_ttfb_metrics()
        # Note: TTSStartedFrame's append_to_context is stamped in
        # _push_audio_context_item, where every TTSStartedFrame
        # (base-class- and subclass-emitted) funnels through.
        await self.append_to_audio_context(context_id, TTSStartedFrame(context_id=context_id))

    async def _register_spoken_text(
        self, src_frame: AggregatedTextFrame, text: str, context_id: str
    ):
        """Give the text a slot in the sequencer, which orders it with skipped text."""
        # The slot unblocks any skipped frames queued behind it once complete.
        # Word-timestamp services complete it via process_word; push_text_frames
        # services complete it after the TTSTextFrame is appended to the audio
        # context. When streaming tokens, the sequencer accumulates this call's
        # text into a pending sentence internally and only registers a real slot
        # once a sentence boundary is detected.
        await self._push_sequencer_frames(
            await self._aggregated_frame_sequencer.register_spoken(
                src_frame,
                context_id,
                text,
                append_to_context=self._tts_contexts[context_id].append_to_context,
                build_tracker=not self._push_text_frames,
                language=self.text_aggregation_language,
            ),
            context_id,
        )

    async def _push_sequencer_frames(self, frames: list[Frame], context_id: str | None):
        """Emit frames returned by the aggregated-frame sequencer in audio order.

        These frames — skipped text the spoken text before it has released, and in
        streaming (TOKEN) mode a promoted sentence anchor and any replayed
        word/progress frames — follow text queued in an audio context, whose audio
        and text frames are emitted by the dedicated audio-context task. Routing
        them through that context's queue keeps a single consumer emitting
        everything in order; pushing directly here would let them overtake it.
        When the context is gone, push directly.

        Args:
            frames: Frames returned by register_spoken/register_skipped/finalize.
            context_id: The context these frames belong to.
        """
        for f in frames:
            if context_id and self.audio_context_available(context_id):
                await self.append_to_audio_context(context_id, f)
            else:
                await self.push_frame(f)

    async def _synthesize_text(self, text: str, context_id: str):
        """Run the service's run_tts for text, unless the service is no longer usable."""
        # When streaming tokens, per-call logs are kept at trace level; the
        # accumulated turn text is logged at debug level at flush time.
        if self._is_streaming_tokens:
            logger.trace(f"{self}: Generating TTS [{text}]")
        else:
            logger.debug(f"{self}: Generating TTS [{text}]")

        # A service that can no longer work can't synthesize anything, and
        # services that connect on demand would attempt a handshake per request.
        # The surrounding bookkeeping still runs, so the turn completes with no
        # audio rather than stalling.
        if self.is_usable:
            await self.tts_process_generator(context_id, self.run_tts(text, context_id))
        else:
            # Name the text that goes unspoken: silence from the bot is
            # otherwise hard to trace back to the service that caused it.
            logger.warning(f"{self}: service is no longer usable, not speaking [{text}]")

    async def _push_spoken_text(
        self, src_frame: AggregatedTextFrame, text: str, context_id: str, append_to_context: bool
    ):
        """Queue the spoken text behind its audio, for services without word timestamps."""
        # In TTS services that support word timestamps, the TTSTextFrames are pushed
        # as words are spoken. Services without word timestamps (_push_text_frames)
        # get the original (non-transformed) text after the TTS generation has
        # completed. This way, if we are interrupted, the text is not added to the
        # assistant context and the context that IS added does not include
        # TTS-specific tags or transformations. The frame carries the text as the
        # LLM wrote it, which the context records, as it does for services with
        # word timestamps.
        #
        # In streaming (TOKEN) mode this is handled instead by the sequencer's
        # per-sentence promotion (see AggregatedFrameSequencer._promote): a call
        # here represents a single token, not the sentence-level unit this frame
        # should carry.
        if not self._push_text_frames or self._is_streaming_tokens:
            return
        frame = TTSTextFrame(
            text,
            text_type=src_frame.text_type,
            raw_text=src_frame.raw_text,
            segment_id=src_frame.id,
        )
        frame.will_be_spoken = True
        frame.context_id = context_id
        frame.append_to_context = append_to_context
        # Appending to the context, so it preserves the ordering.
        await self.append_to_audio_context(context_id, frame)
        # TTSTextFrame is queued; mark the spoken slot complete so any skipped
        # frames (e.g. code blocks) waiting behind it can be flushed in order.
        await self._push_sequencer_frames(
            self._aggregated_frame_sequencer.complete_spoken_slot(), context_id
        )

    async def _stream_audio_frames_from_iterator(
        self,
        iterator: AsyncIterator[bytes],
        *,
        strip_wav_header: bool = False,
        in_sample_rate: int | None = None,
        context_id: str | None = None,
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
            yield TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=context_id)

    #
    # Word frames
    #

    async def _add_word_timestamps(
        self,
        word_times: list[tuple[str, float]],
        context_id: str | None = None,
        includes_inter_frame_spaces: bool = False,
    ):
        """Process word timestamps directly, building and pushing TTSTextFrames inline.

        Used both from _handle_audio_context (via WordTimestamp items) and from services
        that do not use audio contexts. Each entry emits a TTSTextFrame with a PTS
        relative to when start_word_timestamps() was called. Entries that come
        before that are cached and timed once it is called (i.e. when the first
        audio chunk is received).
        """
        for word, timestamp in word_times:
            if not self._word_timestamp_tracker.started:
                self._word_timestamp_tracker.cache(
                    WordTimestamp(word, timestamp, context_id, includes_inter_frame_spaces)
                )
                continue
            # Build TTSTextFrame(s) for this word token, advancing the active
            # slot's tracker and flushing any skipped frames now unblocked.
            await self._push_word_frames(
                self._aggregated_frame_sequencer.process_word(
                    word,
                    self._word_timestamp_tracker.pts(timestamp),
                    context_id,
                    includes_inter_frame_spaces,
                )
            )

    async def _push_word_frames(self, frames: list[Frame]):
        """Push frames built by the sequencer, noting the PTS of the last word."""
        for f in frames:
            if isinstance(f, TTSTextFrame):
                # The sequencer stamps every word frame it builds.
                assert f.pts is not None
                self._word_timestamp_tracker.last_pts = f.pts
            await self.push_frame(f)

    async def _apply_force_complete(self, context_id: str):
        """Force-complete a context's spoken slots and push any unblocked skipped frames.

        Called at end-of-context to handle TTS providers that silently drop word-timestamp
        events. Emits a TTSTextFrame for any remaining unspoken text in this context, then
        flushes skipped frames that were blocked by those incomplete slots. Slots for
        other contexts still in flight are left untouched.

        Args:
            context_id: The audio context that has ended.
        """
        await self._push_word_frames(
            self._aggregated_frame_sequencer.force_complete(
                context_id, self._word_timestamp_tracker.last_pts
            )
        )

    #
    # Audio context playback
    #

    def _refresh_audio_context(self, context_id: str):
        """Signal that the audio context is still in use, resetting the timeout."""
        self._audio_context_manager.keepalive(context_id)

    def _create_audio_context_task(self):
        """Create the task that plays audio contexts."""
        if not self._audio_context_task:
            self._audio_context_task = self.create_task(self._audio_context_task_handler())

    async def _stop_audio_context_task(self):
        """Cancel the task that plays audio contexts."""
        if self._audio_context_task:
            await self.cancel_task(self._audio_context_task)
            self._audio_context_task = None

    async def _audio_context_task_handler(self):
        """Play audio contexts and the frames between them in order.

        Each audio context plays until all its audio has been pushed downstream,
        then on_audio_context_completed() is called. A frame between contexts
        (e.g. an AggregatedTextFrame preceding its audio, or a frame that
        arrived between two speak frames) is pushed at its position.
        """
        running = True
        while running:
            context_value = await self._audio_context_manager.next_for_playback()
            if isinstance(context_value, AudioContextFrame):
                await self._handle_audio_context(context_value.context_id)
            elif isinstance(context_value, Frame):
                self._stamp_llm_response_frame(context_value)
                await self.push_frame(context_value)
            else:
                running = False

    def _stamp_llm_response_frame(self, frame: Frame):
        """Send a reply's end frame after its words, or behind its start frame if it timed none."""
        if isinstance(frame, LLMFullResponseStartFrame):
            self._word_timestamp_tracker.start_reply()
        elif isinstance(frame, LLMFullResponseEndFrame) and not self._push_text_frames:
            frame.pts = self._word_timestamp_tracker.reply_last_pts

    async def _handle_audio_context(self, context_id: str):
        """Push an audio context's items in order until it ends or goes quiet."""
        playback = _AudioContextPlayback()
        item = await self._next_audio_context_item(context_id)
        while item is not None:
            await self._push_audio_context_item(context_id, item, playback)
            item = await self._next_audio_context_item(context_id)
        await self._end_audio_context(context_id, playback)

    async def _next_audio_context_item(self, context_id: str) -> Frame | WordTimestamp | None:
        """Wait for the context's next item, or None once it ends or goes quiet."""
        item = None
        try:
            item = await self._audio_context_manager.get(
                context_id, timeout=self._stop_frame_timeout_s
            )
        except TimeoutError:
            # We didn't get audio, so let's consider this context finished.
            logger.trace(f"{self} time out on audio context {context_id}")
        return item

    async def _push_audio_context_item(
        self, context_id: str, item: Frame | WordTimestamp, playback: _AudioContextPlayback
    ):
        """Push one item of the playing audio context."""
        if isinstance(item, WordTimestamp):
            # Route word timestamps through _add_word_timestamps so they are
            # processed in playback order alongside audio frames.
            await self._add_word_timestamps(
                [(item.word, item.timestamp)],
                item.context_id,
                includes_inter_frame_spaces=item.includes_inter_frame_spaces,
            )
        elif isinstance(item, ErrorFrame):
            await self.push_error_frame(item)
        else:
            if isinstance(item, TTSAudioRawFrame):
                # Set the word-timestamp baseline once, on the first audio chunk.
                if not playback.received_audio:
                    await self.stop_ttfb_metrics()
                    await self.start_word_timestamps()
                playback.received_audio = True
                await self.process_ttfa_metrics(item)
            elif isinstance(item, TTSStartedFrame):
                playback.stop_frame_due = self._push_stop_frames
                # Stamp appropriate append_to_context value onto every
                # TTSStartedFrame here — the single point both base-class- and
                # subclass-emitted started frames pass through.
                tts_context = self._tts_contexts.get(context_id)
                if tts_context is not None:
                    item.append_to_context = tts_context.append_to_context
            elif isinstance(item, TTSStoppedFrame):
                # Checking if we have any remaining spoken slots before pushing the TTSStoppedFrame
                await self._apply_force_complete(context_id)
                playback.stop_frame_due = False
                # Setting the last word timestamp as the TTSStoppedFrame PTS
                if not item.pts:
                    item.pts = self._word_timestamp_tracker.last_pts
            await self.push_frame(item)

    async def _end_audio_context(self, context_id: str, playback: _AudioContextPlayback):
        """Finish the playing audio context once it ends or goes quiet."""
        # Words the provider never timed come before the stop frame.
        await self._apply_force_complete(context_id)
        if playback.stop_frame_due:
            await self.push_frame(TTSStoppedFrame(context_id=context_id))
        await self.reset_word_timestamps()
        await self._record_context_audio_outcome(context_id, playback.received_audio)

        # If audio resumes after a timeout, the context is recreated without
        # the partial sample held here.
        self._audio_context_manager.remove(context_id)
        # A failing hook must not end the playback task, or no more audio is played.
        try:
            await self.on_audio_context_completed(context_id=context_id)
        except Exception as e:
            logger.warning(f"{self} on_audio_context_completed failed for {context_id}: {e}")
        self.reset_active_audio_context()
        await self._release_frames_if_audio_contexts_ended()

    async def _record_context_audio_outcome(self, context_id: str, received_audio: bool):
        """Track whether contexts are producing audio, and act when they stop.

        A provider can accept every request and return no audio at all — an
        unknown voice ID, say — without ever reporting an error. Every context
        that completes in silence is reported as an error, so application code
        hears about a turn that produced no speech as it happens. Enough of them
        in a row means the service isn't going to speak again, so it is reported
        unable to do its job: it stops being given work, a
        :class:`~pipecat.pipeline.service_switcher.ServiceSwitcher` can fail over
        to another provider, and the pipeline worker applies its
        :class:`~pipecat.pipeline.worker.ProcessorUnusablePolicy`.

        Only contexts that complete are counted; an interrupted context is
        cancelled before it gets here.

        Args:
            context_id: The audio context that just completed.
            received_audio: Whether any audio arrived for that context.
        """
        if received_audio:
            self._consecutive_zero_audio_contexts = 0
            return

        # An unusable service is deliberately not given work (see
        # _synthesize_text), so its silent contexts say nothing new.
        if not self.is_usable:
            return

        self._consecutive_zero_audio_contexts += 1
        logger.warning(
            f"{self} audio context {context_id} completed with no audio "
            f"({self._consecutive_zero_audio_contexts} in a row)"
        )

        if (
            self._max_consecutive_zero_audio_contexts
            and self._consecutive_zero_audio_contexts >= self._max_consecutive_zero_audio_contexts
        ):
            await self.push_error(
                error_msg=(
                    f"{self._consecutive_zero_audio_contexts} consecutive TTS contexts "
                    "completed with no audio"
                ),
                force_treat_as_permanent=True,
            )
        else:
            # A single silent context says nothing about whether the service
            # will speak again, so it is reported without costing the service
            # its usability.
            await self.push_error(f"TTS context {context_id} completed with no audio")


@deprecated(
    "`WordTTSService` is deprecated since 0.0.105 and will be removed in 2.0.0. Use "
    "`TTSService` instead."
)
class WordTTSService(TTSService):
    """Deprecated. Use TTSService directly instead.

    .. deprecated:: 0.0.105
        Word timestamp functionality is now always active in TTSService.
        Will be removed in 2.0.0.
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

    Includes an optional keepalive for providers that close an idle connection.
    A subclass enables it by overriding ``_send_keepalive()`` to send its
    provider's keepalive message, and calling ``_create_keepalive_task(interval)``
    / ``_cancel_keepalive_task()`` from its ``_connect()`` / ``_disconnect()``.

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
        self._keepalive_task: asyncio.Task | None = None

    @property
    def supports_processing_metrics(self) -> bool:
        """Whether this service has a meaningful processing-time metric.

        False: ``run_tts`` sends the text and returns, and audio arrives later
        on the receive task, so there is no synthesis inside the measured
        window. A subclass that instead waits for the server to signal the end
        of synthesis before returning can override this back to True.
        """
        return False

    async def stop(self, frame: EndFrame):
        """Stop the websocket TTS service on a graceful end.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the websocket TTS service immediately.

        Disconnecting here is the prompt teardown: the websocket receive loop
        runs independently of the audio-context task, so it keeps reading from
        the provider until the socket is closed. Stopping only the audio-context
        task (in :meth:`TTSService.cancel`) halts output but not reception.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def cleanup(self):
        """Release websocket TTS resources at teardown."""
        await super().cleanup()
        await self._disconnect()

    async def _close_connection_contexts(self):
        """End the audio contexts of a connection that is being replaced.

        Providers keep context state per connection and don't resume it on a
        new one, so audio still due on the old connection never arrives. Every
        open context is closed (audio already received still plays), and a turn
        in progress moves to a new context ID so its remaining sentences open a
        fresh provider context. Reopening a connection that was already closed,
        to send the text at hand, doesn't come through here, so the context it
        sends to stays open.
        """
        for context_id in self.get_audio_contexts():
            await self.remove_audio_context(context_id)
        if self._turn_context_id:
            self._turn_context_id = None
            self._turn_context_id = self.create_context_id()

    async def _disconnect(self):
        # Subclasses call super()._disconnect() first, so this runs before they
        # close their socket.
        await self._close_connection_contexts()
        await super()._disconnect()

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        await self._close_connection_contexts()
        return await super()._reconnect_websocket(attempt_number)

    async def _report_error(self, error: ErrorFrame, force_treat_as_permanent: bool = False):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error_frame(error, force_treat_as_permanent=force_treat_as_permanent)

    def _create_keepalive_task(self, interval: float):
        """Start the keepalive task if it isn't running.

        Args:
            interval: Seconds between keepalive messages.
        """
        if self._keepalive_task and not self._keepalive_task.done():
            return
        self._keepalive_task = self.create_task(
            self._keepalive_task_handler(interval), name="keepalive"
        )

    async def _cancel_keepalive_task(self):
        """Stop the keepalive task if running."""
        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

    async def _keepalive_task_handler(self, interval: float):
        """Send a keepalive every ``interval`` seconds while connected.

        A failed send is logged and the loop keeps going: the receive loop
        handles a lost connection, and the next keepalive goes out on the
        reconnected socket.
        """
        while True:
            await asyncio.sleep(interval)
            try:
                if self._is_keepalive_ready():
                    await self._send_keepalive()
            except Exception as e:
                logger.warning(f"{self} keepalive error: {e}")

    def _is_keepalive_ready(self) -> bool:
        """Check if the service is ready to send a keepalive.

        Returns:
            True if the websocket is open.
        """
        return self._websocket is not None and self._websocket.state is State.OPEN

    async def _send_keepalive(self):
        """Send a keepalive message over the websocket.

        Subclasses that enable keepalive must override this to send their
        provider's keepalive message.
        """
        raise NotImplementedError("Subclasses must override _send_keepalive")


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

        # Whether the bot is speaking, as BotStartedSpeakingFrame and
        # BotStoppedSpeakingFrame report it.
        self._bot_speaking: bool = False
        # True once run_tts has been invoked (TTSStartedFrame pushed) for the
        # current turn but before BotStartedSpeakingFrame confirms playback —
        # the narrow window where _bot_speaking (which only reflects confirmed
        # playback) can't yet tell a reconnect is needed. Kept separate from
        # _bot_speaking so this early, unconfirmed marker never makes a turn
        # that produces no audio look like one that played.
        self._tts_started: bool = False

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        # If the bot is not speaking we don't need to reconnect when the user
        # speaks. If the bot is speaking and the user interrupts we need to
        # reconnect.
        should_reconnect = self._bot_speaking or self._tts_started
        # Without context IDs, audio still coming for an interrupted context
        # would be labeled with an uninterruptible one, so the connection is
        # kept only when uninterruptible speech is all that's open.
        contexts = [
            self._tts_contexts[c] for c in self.get_audio_contexts() if c in self._tts_contexts
        ]
        keep_connection = bool(contexts) and all(not c.interruptible for c in contexts)
        self._bot_speaking = False
        self._tts_started = False
        await super()._handle_interruption(frame, direction)
        if should_reconnect and not keep_connection:
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
            self._tts_started = True

        await super().push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, tracking whether an interruption needs a reconnect.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # The turn ended normally; nothing left to reconnect for.
            self._bot_speaking = False
            self._tts_started = False
        elif isinstance(frame, LLMFullResponseStartFrame):
            # Safety net for a previous turn that never produced audio, so
            # BotStoppedSpeakingFrame never arrived to clear it above.
            self._tts_started = False


@deprecated(
    "`WebsocketWordTTSService` is deprecated since 0.0.105 and will be removed in 2.0.0. "
    "Use `WebsocketTTSService` instead."
)
class WebsocketWordTTSService(WebsocketTTSService):
    """Deprecated. Use WebsocketTTSService directly instead.

    .. deprecated:: 0.0.105
        Word timestamp functionality is now always active in TTSService.
        Will be removed in 2.0.0.
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the Websocket Word TTS service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(reconnect_on_error=reconnect_on_error, **kwargs)


@deprecated(
    "`InterruptibleWordTTSService` is deprecated since 0.0.105 and will be removed in 2.0.0. "
    "Use `InterruptibleTTSService` instead."
)
class InterruptibleWordTTSService(InterruptibleTTSService):
    """Deprecated. Use InterruptibleTTSService directly instead.

    .. deprecated:: 0.0.105
        Word timestamp functionality is now always active in TTSService.
        Will be removed in 2.0.0.
    """

    def __init__(self, **kwargs):
        """Initialize the Interruptible Word TTS service.

        Args:
            **kwargs: Additional arguments passed to the parent InterruptibleTTSService.
        """
        super().__init__(**kwargs)


@deprecated(
    "`AudioContextTTSService` is deprecated since 0.0.105 and will be removed in 2.0.0. "
    "Use `WebsocketTTSService` instead."
)
class AudioContextTTSService(WebsocketTTSService):
    """Deprecated. Inherit from WebsocketTTSService directly instead.

    Audio context management (previously the main purpose of this class) is now
    built into TTSService. This class is kept only for backwards compatibility.

    .. deprecated:: 0.0.105
        Subclass :class:`WebsocketTTSService` directly and pass
        ``reuse_context_id_within_turn`` as
        keyword arguments to its ``__init__``.
        Will be removed in 2.0.0.
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
        super().__init__(
            reuse_context_id_within_turn=reuse_context_id_within_turn,
            reconnect_on_error=reconnect_on_error,
            **kwargs,
        )


@deprecated(
    "`AudioContextWordTTSService` is deprecated since 0.0.105 and will be removed in 2.0.0. "
    "Use `WebsocketTTSService` instead."
)
class AudioContextWordTTSService(AudioContextTTSService):
    """Deprecated. Use WebsocketTTSService directly instead.

    .. deprecated:: 0.0.105
        Subclass :class:`WebsocketTTSService` directly.
        Will be removed in 2.0.0.
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the Audio Context Word TTS service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(reconnect_on_error=reconnect_on_error, **kwargs)


def _is_markup(text: str) -> bool:
    """Whether a word a service reports is markup it echoed, rather than speech."""
    text = text.strip()
    return (text.startswith("[") and text.endswith("]")) or (
        text.startswith("<") and text.endswith(">")
    )
