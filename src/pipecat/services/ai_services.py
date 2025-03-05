#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import wave
from abc import abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

from pipecat.audio.utils import calculate_audio_volume, exp_smoothing
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    STTMuteFrame,
    STTUpdateSettingsFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
    UserImageRequestFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language
from pipecat.utils.string import EndOfSenteceSkipTags, match_endofsentence, parse_eos_skip_tags
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.time import seconds_to_nanoseconds


class AIService(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name: str = ""
        self._settings: Dict[str, Any] = {}
        self._session_properties: Dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    def set_model_name(self, model: str):
        self._model_name = model
        self.set_core_metrics_data(MetricsData(processor=self.name, model=self._model_name))

    async def start(self, frame: StartFrame):
        pass

    async def stop(self, frame: EndFrame):
        pass

    async def cancel(self, frame: CancelFrame):
        pass

    async def _update_settings(self, settings: Mapping[str, Any]):
        from pipecat.services.openai_realtime_beta.events import (
            SessionProperties,
        )

        for key, value in settings.items():
            logger.debug("Update request for:", key, value)

            if key in self._settings:
                logger.info(f"Updating LLM setting {key} to: [{value}]")
                self._settings[key] = value
            elif key in SessionProperties.model_fields:
                logger.debug("Attempting to update", key, value)

                try:
                    from pipecat.services.openai_realtime_beta.events import (
                        TurnDetection,
                    )

                    if isinstance(self._session_properties, SessionProperties):
                        current_properties = self._session_properties
                    else:
                        current_properties = SessionProperties(**self._session_properties)

                    if key == "turn_detection" and isinstance(value, dict):
                        turn_detection = TurnDetection(**value)
                        setattr(current_properties, key, turn_detection)
                    else:
                        setattr(current_properties, key, value)

                    validated_properties = SessionProperties.model_validate(
                        current_properties.model_dump()
                    )
                    logger.info(f"Updating LLM setting {key} to: [{value}]")
                    self._session_properties = validated_properties.model_dump()
                except Exception as e:
                    logger.warning(f"Unexpected error updating session property {key}: {e}")
            elif key == "model":
                logger.info(f"Updating LLM setting {key} to: [{value}]")
                self.set_model_name(value)
            else:
                logger.warning(f"Unknown setting for {self.name} service: {key}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
        elif isinstance(frame, EndFrame):
            await self.stop(frame)

    async def process_generator(self, generator: AsyncGenerator[Frame | None, None]):
        async for f in generator:
            if f:
                if isinstance(f, ErrorFrame):
                    await self.push_error(f)
                else:
                    await self.push_frame(f)


class LLMService(AIService):
    """This class is a no-op but serves as a base class for LLM services."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._callbacks = {}
        self._start_callbacks = {}

        self._register_event_handler("on_completion_timeout")

    # TODO-CB: callback function type
    def register_function(self, function_name: Optional[str], callback, start_callback=None):
        # Registering a function with the function_name set to None will run that callback
        # for all functions
        self._callbacks[function_name] = callback
        # QUESTION FOR CB: maybe this isn't needed anymore?
        if start_callback:
            self._start_callbacks[function_name] = start_callback

    def unregister_function(self, function_name: Optional[str]):
        del self._callbacks[function_name]
        if self._start_callbacks[function_name]:
            del self._start_callbacks[function_name]

    def has_function(self, function_name: str):
        if None in self._callbacks.keys():
            return True
        return function_name in self._callbacks.keys()

    async def call_function(
        self,
        *,
        context: OpenAILLMContext,
        tool_call_id: str,
        function_name: str,
        arguments: str,
        run_llm: bool = True,
    ) -> None:
        f = None
        if function_name in self._callbacks.keys():
            f = self._callbacks[function_name]
        elif None in self._callbacks.keys():
            f = self._callbacks[None]
        else:
            return None
        await self.call_start_function(context, function_name)
        await context.call_function(
            f,
            function_name=function_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            llm=self,
            run_llm=run_llm,
        )

    # QUESTION FOR CB: maybe this isn't needed anymore?
    async def call_start_function(self, context: OpenAILLMContext, function_name: str):
        if function_name in self._start_callbacks.keys():
            await self._start_callbacks[function_name](function_name, self, context)
        elif None in self._start_callbacks.keys():
            return await self._start_callbacks[None](function_name, self, context)

    async def request_image_frame(self, user_id: str, *, text_content: Optional[str] = None):
        await self.push_frame(
            UserImageRequestFrame(user_id=user_id, context=text_content), FrameDirection.UPSTREAM
        )


class TTSService(AIService):
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
        # Text filter before passing text to the TTS service
        text_filter: Optional[BaseTextFilter] = None,
        # List of end-of-sent
        eos_skip_tags: Sequence[EndOfSenteceSkipTags] = [],
        **kwargs,
    ):
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
        self._text_filter: Optional[BaseTextFilter] = text_filter
        self._eos_skip_tags = eos_skip_tags

        self._stop_frame_task: Optional[asyncio.Task] = None
        self._stop_frame_queue: asyncio.Queue = asyncio.Queue()

        self._processing_text: bool = False
        self._current_sentence: str = ""
        self._current_eos_skip_tag: Optional[EndOfSenteceSkipTags] = None
        self._current_eos_skip_tag_index: int = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def set_model(self, model: str):
        self.set_model_name(model)

    def set_voice(self, voice: str):
        self._voice_id = voice

    @abstractmethod
    async def flush_audio(self):
        pass

    # Converts the text to audio.
    @abstractmethod
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        pass

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return Language(language)

    async def update_setting(self, key: str, value: Any):
        pass

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._sample_rate = self._init_sample_rate or frame.audio_out_sample_rate
        if self._push_stop_frames and not self._stop_frame_task:
            self._stop_frame_task = self.create_task(self._stop_frame_handler())

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self._stop_frame_task:
            await self.cancel_task(self._stop_frame_task)
            self._stop_frame_task = None

    async def cancel(self, frame: CancelFrame):
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
            elif key == "voice":
                self.set_voice(value)
            elif key == "text_filter" and self._text_filter:
                self._text_filter.update_settings(value)
            else:
                logger.warning(f"Unknown setting for TTS service: {key}")

    async def say(self, text: str):
        await self.queue_frame(TTSSpeakFrame(text))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, TextFrame)
            and not isinstance(frame, InterimTranscriptionFrame)
            and not isinstance(frame, TranscriptionFrame)
        ):
            await self._process_text_frame(frame)
        elif isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption(frame, direction)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            # We pause processing incoming frames if the LLM response included
            # text (it might be that it's only a function calling response). We
            # pause to avoid audio overlapping.
            await self._maybe_pause_frame_processing()

            sentence = self._current_sentence
            self._reset()
            await self._push_tts_frames(sentence)
            if isinstance(frame, LLMFullResponseEndFrame):
                if self._push_text_frames:
                    await self.push_frame(frame, direction)
            else:
                await self.push_frame(frame, direction)
        elif isinstance(frame, TTSSpeakFrame):
            await self._push_tts_frames(frame.text)
            # We pause processing incoming frames because we are sending data to
            # the TTS. We pause to avoid audio overlapping.
            await self._maybe_pause_frame_processing()
            await self.flush_audio()
            self._processing_text = False
        elif isinstance(frame, TTSUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._maybe_resume_frame_processing()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if self._push_silence_after_stop and isinstance(frame, TTSStoppedFrame):
            silence_num_bytes = int(self._silence_time_s * self.sample_rate * 2)  # 16-bit
            await self.push_frame(
                TTSAudioRawFrame(
                    audio=b"\x00" * silence_num_bytes,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
            )

        await super().push_frame(frame, direction)

        if self._push_stop_frames and (
            isinstance(frame, StartInterruptionFrame)
            or isinstance(frame, TTSStartedFrame)
            or isinstance(frame, TTSAudioRawFrame)
            or isinstance(frame, TTSStoppedFrame)
        ):
            await self._stop_frame_queue.put(frame)

    def _reset(self):
        self._current_sentence = ""
        self._processing_text = False
        self._current_eos_skip_tag = None
        self._current_eos_skip_tag_index = 0

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        self._reset()
        if self._text_filter:
            self._text_filter.handle_interruption()

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
            self._current_sentence += frame.text
            (self._current_eos_skip_tag_index, self._current_eos_skip_tag) = parse_eos_skip_tags(
                self._current_sentence,
                self._eos_skip_tags,
                self._current_eos_skip_tag,
                self._current_eos_skip_tag_index,
            )
            if not self._current_eos_skip_tag:
                eos_end_marker = match_endofsentence(self._current_sentence)
                if eos_end_marker:
                    text = self._current_sentence[:eos_end_marker]
                    self._current_sentence = self._current_sentence[eos_end_marker:]

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
        if self._text_filter:
            self._text_filter.reset_interruption()
            text = self._text_filter.filter(text)
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
                    self._stop_frame_queue.get(), self._stop_frame_timeout_s
                )
                if isinstance(frame, TTSStartedFrame):
                    has_started = True
                elif isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
                    has_started = False
            except asyncio.TimeoutError:
                if has_started:
                    await self.push_frame(TTSStoppedFrame())
                    has_started = False


class WordTTSService(TTSService):
    """This is a base class for TTS services that support word timestamps. Word
    timestamps are useful to synchronize audio with text of the spoken
    words. This way only the spoken words are added to the conversation context.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initial_word_timestamp = -1
        self._words_queue = asyncio.Queue()
        self._words_task = None

    def start_word_timestamps(self):
        if self._initial_word_timestamp == -1:
            self._initial_word_timestamp = self.get_clock().get_time()

    def reset_word_timestamps(self):
        self._initial_word_timestamp = -1

    async def add_word_timestamps(self, word_times: List[Tuple[str, float]]):
        for word, timestamp in word_times:
            await self._words_queue.put((word, seconds_to_nanoseconds(timestamp)))

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._create_words_task()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop_words_task()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop_words_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        self.reset_word_timestamps()

    def _create_words_task(self):
        if not self._words_task:
            self._words_task = self.create_task(self._words_task_handler())

    async def _stop_words_task(self):
        if self._words_task:
            await self.cancel_task(self._words_task)
            self._words_task = None

    async def _words_task_handler(self):
        last_pts = 0
        while True:
            (word, timestamp) = await self._words_queue.get()
            if word == "Reset" and timestamp == 0:
                self.reset_word_timestamps()
                frame = None
            elif word == "LLMFullResponseEndFrame" and timestamp == 0:
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
    """This is a base class for websocket-based TTS services."""

    def __init__(self, **kwargs):
        TTSService.__init__(self, **kwargs)
        WebsocketService.__init__(self)


class InterruptibleTTSService(WebsocketTTSService):
    """This is a base class for websocket-based TTS services that don't support
    word timestamps and that don't offer a way to correlate the generated audio
    to the requested text.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Indicates if the bot is speaking. If the bot is not speaking we don't
        # need to reconnect when the user speaks. If the bot is speaking and the
        # user interrupts we need to reconnect.
        self._bot_speaking = False

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        if self._bot_speaking:
            await self._disconnect()
            await self._connect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False


class WebsocketWordTTSService(WordTTSService, WebsocketService):
    """This is a base class for websocket-based TTS services that support word
    timestamps.

    """

    def __init__(self, **kwargs):
        WordTTSService.__init__(self, **kwargs)
        WebsocketService.__init__(self)


class InterruptibleWordTTSService(WebsocketWordTTSService):
    """This is a base class for websocket-based TTS services that support word
    timestamps but don't offer a way to correlate the generated audio to the
    requested text.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Indicates if the bot is speaking. If the bot is not speaking we don't
        # need to reconnect when the user speaks. If the bot is speaking and the
        # user interrupts we need to reconnect.
        self._bot_speaking = False

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        if self._bot_speaking:
            await self._disconnect()
            await self._connect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False


class AudioContextWordTTSService(WebsocketWordTTSService):
    """This is a base class for websocket-based TTS services that support word
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
        super().__init__(**kwargs)
        self._contexts_queue = asyncio.Queue()
        self._contexts: Dict[str, asyncio.Queue] = {}
        self._audio_context_task = None

    async def create_audio_context(self, context_id: str):
        """Create a new audio context."""
        await self._contexts_queue.put(context_id)
        self._contexts[context_id] = asyncio.Queue()
        logger.trace(f"{self} created audio context {context_id}")

    async def append_to_audio_context(self, context_id: str, frame: TTSAudioRawFrame):
        """Append audio to an existing context."""
        if self.audio_context_available(context_id):
            logger.trace(f"{self} appending audio {frame} to audio context {context_id}")
            await self._contexts[context_id].put(frame)
        else:
            logger.warning(f"{self} unable to append audio to context {context_id}")

    async def remove_audio_context(self, context_id: str):
        """Remove an existing audio context."""
        if self.audio_context_available(context_id):
            # We just mark the audio context for deletion by appending
            # None. Once we reach None while handling audio we know we can
            # safely remove the context.
            logger.trace(f"{self} marking audio context {context_id} for deletion")
            await self._contexts[context_id].put(None)
        else:
            logger.warning(f"{self} unable to remove context {context_id}")

    def audio_context_available(self, context_id: str) -> bool:
        """Checks whether the given audio context is registered."""
        return context_id in self._contexts

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._create_audio_context_task()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self._audio_context_task:
            # Indicate no more audio contexts are available. this will end the
            # task cleanly after all contexts have been processed.
            await self._contexts_queue.put(None)
            await self.wait_for_task(self._audio_context_task)
            self._audio_context_task = None

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop_audio_context_task()

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
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


class STTService(AIService):
    """STTService is a base class for speech-to-text services."""

    def __init__(
        self,
        audio_passthrough=False,
        # STT input sample rate
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._audio_passthrough = audio_passthrough
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._settings: Dict[str, Any] = {}
        self._muted: bool = False

    @property
    def is_muted(self) -> bool:
        """Returns whether the STT service is currently muted."""
        return self._muted

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @abstractmethod
    async def set_model(self, model: str):
        self.set_model_name(model)

    @abstractmethod
    async def set_language(self, language: Language):
        pass

    @abstractmethod
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Returns transcript as a string"""
        pass

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._sample_rate = self._init_sample_rate or frame.audio_in_sample_rate

    async def _update_settings(self, settings: Mapping[str, Any]):
        logger.info(f"Updating STT settings: {self._settings}")
        for key, value in settings.items():
            if key in self._settings:
                logger.info(f"Updating STT setting {key} to: [{value}]")
                self._settings[key] = value
                if key == "language":
                    await self.set_language(value)
            elif key == "model":
                self.set_model_name(value)
            else:
                logger.warning(f"Unknown setting for STT service: {key}")

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        if self._muted:
            return

        await self.process_generator(self.run_stt(frame.audio))
        if self._audio_passthrough:
            await self.push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it."""
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            # In this service we accumulate audio internally and at the end we
            # push a TextFrame. We also push audio downstream in case someone
            # else needs it.
            await self.process_audio_frame(frame, direction)
        elif isinstance(frame, STTUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        elif isinstance(frame, STTMuteFrame):
            self._muted = frame.mute
            logger.debug(f"STT service {'muted' if frame.mute else 'unmuted'}")
        else:
            await self.push_frame(frame, direction)


class SegmentedSTTService(STTService):
    """SegmentedSTTService is an STTService that will detect speech and will run
    speech-to-text on speech segments only, instead of a continous stream.

    """

    def __init__(
        self,
        *,
        min_volume: float = 0.6,
        max_silence_secs: float = 0.3,
        max_buffer_secs: float = 1.5,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._min_volume = min_volume
        self._max_silence_secs = max_silence_secs
        self._max_buffer_secs = max_buffer_secs
        self._content = None
        self._wave = None
        self._silence_num_frames = 0
        # Volume exponential smoothing
        self._smoothing_factor = 0.2
        self._prev_volume = 0

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        # Try to filter out empty background noise
        volume = self._get_smoothed_volume(frame)
        if volume >= self._min_volume:
            # If volume is high enough, write new data to wave file
            self._wave.writeframes(frame.audio)
            self._silence_num_frames = 0
        else:
            self._silence_num_frames += frame.num_frames
        self._prev_volume = volume

        # If buffer is not empty and we have enough data or there's been a long
        # silence, transcribe the audio gathered so far.
        silence_secs = self._silence_num_frames / self.sample_rate
        buffer_secs = self._wave.getnframes() / self.sample_rate
        if self._content.tell() > 0 and (
            buffer_secs > self._max_buffer_secs or silence_secs > self._max_silence_secs
        ):
            self._silence_num_frames = 0
            self._wave.close()
            self._content.seek(0)
            await self.process_generator(self.run_stt(self._content.read()))
            (self._content, self._wave) = self._new_wave()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if not self._wave:
            (self._content, self._wave) = self._new_wave()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        self._wave.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        self._wave.close()

    def _new_wave(self):
        content = io.BytesIO()
        ww = wave.open(content, "wb")
        ww.setsampwidth(2)
        ww.setnchannels(1)
        ww.setframerate(self.sample_rate)
        return (content, ww)

    def _get_smoothed_volume(self, frame: AudioRawFrame) -> float:
        volume = calculate_audio_volume(frame.audio, frame.sample_rate)
        return exp_smoothing(volume, self._prev_volume, self._smoothing_factor)


class ImageGenService(AIService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Renders the image. Returns an Image object.
    @abstractmethod
    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            await self.start_processing_metrics()
            await self.process_generator(self.run_image_gen(frame.text))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)


class VisionService(AIService):
    """VisionService is a base class for vision services."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._describe_text = None

    @abstractmethod
    async def run_vision(self, frame: VisionImageRawFrame) -> AsyncGenerator[Frame, None]:
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VisionImageRawFrame):
            await self.start_processing_metrics()
            await self.process_generator(self.run_vision(frame))
            await self.stop_processing_metrics()
        else:
            await self.push_frame(frame, direction)
