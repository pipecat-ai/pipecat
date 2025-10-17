#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base classes for Text-to-speech services."""

import asyncio
from abc import abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
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
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.utils.time import seconds_to_nanoseconds


class TTSService(AIService):
    """Base class for text-to-speech services.

    Provides common functionality for TTS services including text aggregation,
    filtering, audio generation, and frame management. Supports configurable
    sentence aggregation, silence insertion, and frame processing control.

    Event handlers:
        on_connected: Called when connected to the STT service.
        on_connected: Called when disconnected from the STT service.
        on_connection_error: Called when a connection to the STT service error occurs.

    Example::

        @tts.event_handler("on_connected")
        async def on_connected(tts: TTSService):
            logger.debug(f"TTS connected")

        @tts.event_handler("on_disconnected")
        async def on_disconnected(tts: TTSService):
            logger.debug(f"TTS disconnected")

        @tts.event_handler("on_connection_error")
        async def on_connection_error(stt: TTSService, error: str):
            logger.error(f"TTS connection error: {error}")
    """

    def __init__(
        self,
        *,
        aggregate_sentences: bool = True,
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
        # TTS output sample rate
        sample_rate: Optional[int] = None,
        # Text aggregator to aggregate incoming tokens and decide when to push to the TTS.
        text_aggregator: Optional[BaseTextAggregator] = None,
        # Text filter executed after text has been aggregated.
        text_filters: Optional[Sequence[BaseTextFilter]] = None,
        text_filter: Optional[BaseTextFilter] = None,
        # Audio transport destination of the generated frames.
        transport_destination: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the TTS service.

        Args:
            aggregate_sentences: Whether to aggregate text into sentences before synthesis.
            push_text_frames: Whether to push TextFrames and LLMFullResponseEndFrames.
            push_stop_frames: Whether to automatically push TTSStoppedFrames.
            stop_frame_timeout_s: Idle time before pushing TTSStoppedFrame when push_stop_frames is True.
            push_silence_after_stop: Whether to push silence audio after TTSStoppedFrame.
            silence_time_s: Duration of silence to push when push_silence_after_stop is True.
            pause_frame_processing: Whether to pause frame processing during audio generation.
            sample_rate: Output sample rate for generated audio.
            text_aggregator: Custom text aggregator for processing incoming text.
            text_filters: Sequence of text filters to apply after aggregation.
            text_filter: Single text filter (deprecated, use text_filters).

                .. deprecated:: 0.0.59
                    Use `text_filters` instead, which allows multiple filters.

            transport_destination: Destination for generated audio frames.
            **kwargs: Additional arguments passed to the parent AIService.
        """
        super().__init__(**kwargs)
        self._aggregate_sentences: bool = aggregate_sentences
        self._push_text_frames: bool = push_text_frames
        self._push_stop_frames: bool = push_stop_frames
        self._stop_frame_timeout_s: float = stop_frame_timeout_s
        self._push_silence_after_stop: bool = push_silence_after_stop
        self._silence_time_s: float = silence_time_s
        self._pause_frame_processing: bool = pause_frame_processing
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._voice_id: str = ""
        self._settings: Dict[str, Any] = {}
        self._text_aggregator: BaseTextAggregator = text_aggregator or SimpleTextAggregator()
        self._text_filters: Sequence[BaseTextFilter] = text_filters or []
        self._transport_destination: Optional[str] = transport_destination
        self._tracing_enabled: bool = False

        if text_filter:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'text_filter' is deprecated, use 'text_filters' instead.",
                    DeprecationWarning,
                )
            self._text_filters = [text_filter]

        self._stop_frame_task: Optional[asyncio.Task] = None
        self._stop_frame_queue: asyncio.Queue = asyncio.Queue()

        self._processing_text: bool = False

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_connection_error")

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

        Args:
            model: The name of the TTS model.
        """
        self.set_model_name(model)

    def set_voice(self, voice: str):
        """Set the voice for speech synthesis.

        Args:
            voice: The voice identifier or name.
        """
        self._voice_id = voice

    # Converts the text to audio.
    @abstractmethod
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Run text-to-speech synthesis on the provided text.

        This method must be implemented by subclasses to provide actual TTS functionality.

        Args:
            text: The text to synthesize into speech.

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

    async def update_setting(self, key: str, value: Any):
        """Update a service-specific setting.

        Args:
            key: The setting key to update.
            value: The new value for the setting.
        """
        pass

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
        self._tracing_enabled = frame.enable_tracing

    async def stop(self, frame: EndFrame):
        """Stop the TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        if self._stop_frame_task:
            await self.cancel_task(self._stop_frame_task)
            self._stop_frame_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        if self._stop_frame_task:
            await self.cancel_task(self._stop_frame_task)
            self._stop_frame_task = None

    async def _update_settings(self, settings: Mapping[str, Any]):
        for key, value in settings.items():
            if key in self._settings:
                logger.info(f"Updating TTS setting {key} to: [{value}]")
                self._settings[key] = value
                if key == "language":
                    self._settings[key] = self.language_to_service_language(value)
            elif key == "model":
                self.set_model_name(value)
            elif key == "voice" or key == "voice_id":
                self.set_voice(value)
            elif key == "text_filter":
                for filter in self._text_filters:
                    await filter.update_settings(value)
            else:
                logger.warning(f"Unknown setting for TTS service: {key}")

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
        elif (
            isinstance(frame, TextFrame)
            and not isinstance(frame, InterimTranscriptionFrame)
            and not isinstance(frame, TranscriptionFrame)
        ):
            await self._process_text_frame(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruption(frame, direction)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            # We pause processing incoming frames if the LLM response included
            # text (it might be that it's only a function calling response). We
            # pause to avoid audio overlapping.
            await self._maybe_pause_frame_processing()

            sentence = self._text_aggregator.text
            await self._text_aggregator.reset()
            self._processing_text = False
            await self._push_tts_frames(sentence)
            if isinstance(frame, LLMFullResponseEndFrame):
                if self._push_text_frames:
                    await self.push_frame(frame, direction)
            else:
                await self.push_frame(frame, direction)
        elif isinstance(frame, TTSSpeakFrame):
            # Store if we were processing text or not so we can set it back.
            processing_text = self._processing_text
            await self._push_tts_frames(frame.text)
            # We pause processing incoming frames because we are sending data to
            # the TTS. We pause to avoid audio overlapping.
            await self._maybe_pause_frame_processing()
            await self.flush_audio()
            self._processing_text = processing_text
        elif isinstance(frame, TTSUpdateSettingsFrame):
            await self._update_settings(frame.settings)
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
        self, iterator: AsyncIterator[bytes], *, strip_wav_header: bool
    ) -> AsyncGenerator[Frame, None]:
        buffer = bytearray()
        need_to_strip_wav_header = strip_wav_header
        async for chunk in iterator:
            if need_to_strip_wav_header and chunk.startswith(b"RIFF"):
                chunk = chunk[44:]
                need_to_strip_wav_header = False

            # Append to current buffer.
            buffer.extend(chunk)

            # Round to nearest even number.
            aligned_length = len(buffer) & ~1  # 111111111...11110
            if aligned_length > 0:
                aligned_chunk = buffer[:aligned_length]
                buffer = buffer[aligned_length:]  # keep any leftover byte

                if len(aligned_chunk) > 0:
                    frame = TTSAudioRawFrame(bytes(aligned_chunk), self.sample_rate, 1)
                    yield frame

        if len(buffer) > 0:
            # Make sure we don't need an extra padding byte.
            if len(buffer) % 2 == 1:
                buffer.extend(b"\x00")
            frame = TTSAudioRawFrame(bytes(buffer), self.sample_rate, 1)
            yield frame

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        self._processing_text = False
        await self._text_aggregator.handle_interruption()
        for filter in self._text_filters:
            await filter.handle_interruption()

    async def _maybe_pause_frame_processing(self):
        if self._processing_text and self._pause_frame_processing:
            await self.pause_processing_frames()

    async def _maybe_resume_frame_processing(self):
        if self._pause_frame_processing:
            await self.resume_processing_frames()

    async def _process_text_frame(self, frame: TextFrame):
        text: Optional[str] = None
        if not self._aggregate_sentences:
            text = frame.text
        else:
            text = await self._text_aggregator.aggregate(frame.text)

        if text:
            await self._push_tts_frames(text)

    async def _push_tts_frames(self, text: str):
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

        await self.start_processing_metrics()

        # Process all filter.
        for filter in self._text_filters:
            await filter.reset_interruption()
            text = await filter.filter(text)

        if text:
            await self.process_generator(self.run_tts(text))

        await self.stop_processing_metrics()

        if self._push_text_frames:
            # We send the original text after the audio. This way, if we are
            # interrupted, the text is not added to the assistant context.
            await self.push_frame(TTSTextFrame(text))

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


class WordTTSService(TTSService):
    """Base class for TTS services that support word timestamps.

    Word timestamps are useful to synchronize audio with text of the spoken
    words. This way only the spoken words are added to the conversation context.
    """

    def __init__(self, **kwargs):
        """Initialize the Word TTS service.

        Args:
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(**kwargs)
        self._initial_word_timestamp = -1
        self._words_task = None
        self._llm_response_started: bool = False

    def start_word_timestamps(self):
        """Start tracking word timestamps from the current time."""
        if self._initial_word_timestamp == -1:
            self._initial_word_timestamp = self.get_clock().get_time()

    def reset_word_timestamps(self):
        """Reset word timestamp tracking."""
        self._initial_word_timestamp = -1

    async def add_word_timestamps(self, word_times: List[Tuple[str, float]]):
        """Add word timestamps to the processing queue.

        Args:
            word_times: List of (word, timestamp) tuples where timestamp is in seconds.
        """
        for word, timestamp in word_times:
            await self._words_queue.put((word, seconds_to_nanoseconds(timestamp)))

    async def start(self, frame: StartFrame):
        """Start the word TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._create_words_task()

    async def stop(self, frame: EndFrame):
        """Stop the word TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._stop_words_task()

    async def cancel(self, frame: CancelFrame):
        """Cancel the word TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._stop_words_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with word timestamp awareness.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._llm_response_started = True
        elif isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        self._llm_response_started = False
        self.reset_word_timestamps()

    def _create_words_task(self):
        if not self._words_task:
            self._words_queue = asyncio.Queue()
            self._words_task = self.create_task(self._words_task_handler())

    async def _stop_words_task(self):
        if self._words_task:
            await self.cancel_task(self._words_task)
            self._words_task = None

    async def _words_task_handler(self):
        last_pts = 0
        while True:
            frame = None
            (word, timestamp) = await self._words_queue.get()
            if word == "Reset" and timestamp == 0:
                self.reset_word_timestamps()
                if self._llm_response_started:
                    self._llm_response_started = False
                    frame = LLMFullResponseEndFrame()
                    frame.pts = last_pts
            elif word == "TTSStoppedFrame" and timestamp == 0:
                frame = TTSStoppedFrame()
                frame.pts = last_pts
            else:
                frame = TTSTextFrame(word)
                frame.pts = self._initial_word_timestamp + timestamp
            if frame:
                last_pts = frame.pts
                await self.push_frame(frame)
            self._words_queue.task_done()


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
        await self.push_error(error)


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


class WebsocketWordTTSService(WordTTSService, WebsocketService):
    """Base class for websocket-based TTS services that support word timestamps.

    Combines word timestamp functionality with websocket connectivity.
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the Websocket Word TTS service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        WordTTSService.__init__(self, **kwargs)
        WebsocketService.__init__(self, reconnect_on_error=reconnect_on_error, **kwargs)

    async def _report_error(self, error: ErrorFrame):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error(error)


class InterruptibleWordTTSService(WebsocketWordTTSService):
    """Websocket-based TTS service with word timestamps that handles interruptions.

    For TTS services that support word timestamps but can't correlate generated
    audio with requested text. Handles interruptions by reconnecting when needed.
    """

    def __init__(self, **kwargs):
        """Initialize the Interruptible Word TTS service.

        Args:
            **kwargs: Additional arguments passed to the parent WebsocketWordTTSService.
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


class AudioContextWordTTSService(WebsocketWordTTSService):
    """Websocket-based TTS service with word timestamps and audio context management.

    This is a base class for websocket-based TTS services that support word
    timestamps and also allow correlating the generated audio with the requested
    text.

    Each request could be multiple sentences long which are grouped by
    context. For this to work, the TTS service needs to support handling
    multiple requests at once (i.e. multiple simultaneous contexts).

    The audio received from the TTS will be played in context order. That is, if
    we requested audio for a context "A" and then audio for context "B", the
    audio from context ID "A" will be played first.
    """

    def __init__(self, **kwargs):
        """Initialize the Audio Context Word TTS service.

        Args:
            **kwargs: Additional arguments passed to the parent WebsocketWordTTSService.
        """
        super().__init__(**kwargs)
        self._contexts: Dict[str, asyncio.Queue] = {}
        self._audio_context_task = None

    async def create_audio_context(self, context_id: str):
        """Create a new audio context for grouping related audio.

        Args:
            context_id: Unique identifier for the audio context.
        """
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

    def audio_context_available(self, context_id: str) -> bool:
        """Check whether the given audio context is registered.

        Args:
            context_id: The context ID to check.

        Returns:
            True if the context exists and is available.
        """
        return context_id in self._contexts

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

            if context_id:
                # Process the audio context until the context doesn't have more
                # audio available (i.e. we find None).
                await self._handle_audio_context(context_id)

                # We just finished processing the context, so we can safely remove it.
                del self._contexts[context_id]

                # Append some silence between sentences.
                silence = b"\x00" * self.sample_rate
                frame = TTSAudioRawFrame(
                    audio=silence, sample_rate=self.sample_rate, num_channels=1
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
                if frame:
                    await self.push_frame(frame)
                running = frame is not None
            except asyncio.TimeoutError:
                # We didn't get audio, so let's consider this context finished.
                logger.trace(f"{self} time out on audio context {context_id}")
                break
