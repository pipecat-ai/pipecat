#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure Cognitive Services Text-to-Speech service implementations."""

import asyncio
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.azure.common import language_to_azure_language
from pipecat.services.tts_service import TTSService, WordTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from azure.cognitiveservices.speech import (
        CancellationReason,
        ResultReason,
        ServicePropertyChannel,
        SpeechConfig,
        SpeechSynthesisOutputFormat,
        SpeechSynthesizer,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Azure, you need to `pip install pipecat-ai[azure]`.")
    raise Exception(f"Missing module: {e}")


def sample_rate_to_output_format(sample_rate: int) -> SpeechSynthesisOutputFormat:
    """Convert sample rate to Azure speech synthesis output format.

    Args:
        sample_rate: Sample rate in Hz.

    Returns:
        Corresponding Azure SpeechSynthesisOutputFormat enum value.
        Defaults to Raw24Khz16BitMonoPcm if sample rate not found.
    """
    sample_rate_map = {
        8000: SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm,
        16000: SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm,
        22050: SpeechSynthesisOutputFormat.Raw22050Hz16BitMonoPcm,
        24000: SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,
        44100: SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm,
        48000: SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,
    }
    return sample_rate_map.get(sample_rate, SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm)


class AzureBaseTTSService:
    """Base mixin class for Azure Cognitive Services text-to-speech implementations.

    Provides common functionality for Azure TTS services including SSML
    construction, voice configuration, and parameter management.
    This is a mixin class and should be used alongside TTSService or its subclasses.
    """

    # Define SSML escape mappings based on SSML reserved characters
    # See - https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-structure
    SSML_ESCAPE_CHARS = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&apos;",
    }

    class InputParams(BaseModel):
        """Input parameters for Azure TTS voice configuration.

        Parameters:
            emphasis: Emphasis level for speech ("strong", "moderate", "reduced").
            language: Language for synthesis. Defaults to English (US).
            pitch: Voice pitch adjustment (e.g., "+10%", "-5Hz", "high").
            rate: Speech rate adjustment (e.g., "1.0", "1.25", "slow", "fast").
            role: Voice role for expression (e.g., "YoungAdultFemale").
            style: Speaking style (e.g., "cheerful", "sad", "excited").
            style_degree: Intensity of the speaking style (0.01 to 2.0).
            volume: Volume level (e.g., "+20%", "loud", "x-soft").
        """

        emphasis: Optional[str] = None
        language: Optional[Language] = Language.EN_US
        pitch: Optional[str] = None
        rate: Optional[str] = None
        role: Optional[str] = None
        style: Optional[str] = None
        style_degree: Optional[str] = None
        volume: Optional[str] = None

    def _init_azure_base(
        self,
        *,
        api_key: str,
        region: str,
        voice: str = "en-US-SaraNeural",
        params: Optional[InputParams] = None,
    ):
        """Initialize Azure-specific configuration.

        This method should be called by subclasses after initializing their TTSService parent.

        Args:
            api_key: Azure Cognitive Services subscription key.
            region: Azure region identifier (e.g., "eastus", "westus2").
            voice: Voice name to use for synthesis. Defaults to "en-US-SaraNeural".
            params: Voice and synthesis parameters configuration.
        """
        params = params or AzureBaseTTSService.InputParams()

        self._settings = {
            "emphasis": params.emphasis,
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-US",
            "pitch": params.pitch,
            "rate": params.rate,
            "role": params.role,
            "style": params.style,
            "style_degree": params.style_degree,
            "volume": params.volume,
        }

        self._api_key = api_key
        self._region = region
        self._voice_id = voice
        self._speech_synthesizer = None

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Azure language format.

        Args:
            language: The language to convert.

        Returns:
            The Azure-specific language code, or None if not supported.
        """
        return language_to_azure_language(language)

    def _construct_ssml(self, text: str) -> str:
        language = self._settings["language"]

        # Escape special characters
        escaped_text = self._escape_text(text)

        ssml = (
            f"<speak version='1.0' xml:lang='{language}' "
            "xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{self._voice_id}'>"
            "<mstts:silence type='Sentenceboundary' value='20ms' />"
        )

        if self._settings["style"]:
            ssml += f"<mstts:express-as style='{self._settings['style']}'"
            if self._settings["style_degree"]:
                ssml += f" styledegree='{self._settings['style_degree']}'"
            if self._settings["role"]:
                ssml += f" role='{self._settings['role']}'"
            ssml += ">"

        prosody_attrs = []
        if self._settings["rate"]:
            prosody_attrs.append(f"rate='{self._settings['rate']}'")
        if self._settings["pitch"]:
            prosody_attrs.append(f"pitch='{self._settings['pitch']}'")
        if self._settings["volume"]:
            prosody_attrs.append(f"volume='{self._settings['volume']}'")

        # Only wrap in prosody tag if there are prosody attributes
        if prosody_attrs:
            ssml += f"<prosody {' '.join(prosody_attrs)}>"

        if self._settings["emphasis"]:
            ssml += f"<emphasis level='{self._settings['emphasis']}'>"

        ssml += escaped_text

        if self._settings["emphasis"]:
            ssml += "</emphasis>"

        if prosody_attrs:
            ssml += "</prosody>"

        if self._settings["style"]:
            ssml += "</mstts:express-as>"

        ssml += "</voice></speak>"

        return ssml

    def _escape_text(self, text: str) -> str:
        """Escapes XML/SSML reserved characters according to Microsoft documentation.

        This method escapes the following characters:
        - & becomes &amp;
        - < becomes &lt;
        - > becomes &gt;
        - " becomes &quot;
        - ' becomes &apos;

        Args:
            text: The text to escape.

        Returns:
            The escaped text.
        """
        escaped_text = text
        for char, escape_code in AzureBaseTTSService.SSML_ESCAPE_CHARS.items():
            escaped_text = escaped_text.replace(char, escape_code)
        return escaped_text


class AzureTTSService(WordTTSService, AzureBaseTTSService):
    """Azure Cognitive Services streaming TTS service with word timestamps.

    Provides real-time text-to-speech synthesis using Azure's WebSocket-based
    streaming API. Audio chunks and word boundaries are streamed as they become
    available for lower latency playback and accurate word-level synchronization.
    """

    def __init__(
        self,
        *,
        api_key: str,
        region: str,
        voice: str = "en-US-SaraNeural",
        sample_rate: Optional[int] = None,
        params: Optional[AzureBaseTTSService.InputParams] = None,
        aggregate_sentences: bool = True,
        **kwargs,
    ):
        """Initialize the Azure streaming TTS service.

        Args:
            api_key: Azure Cognitive Services subscription key.
            region: Azure region identifier (e.g., "eastus", "westus2").
            voice: Voice name to use for synthesis. Defaults to "en-US-SaraNeural".
            sample_rate: Audio sample rate in Hz. If None, uses service default.
            params: Voice and synthesis parameters configuration.
            aggregate_sentences: Whether to aggregate sentences before synthesis.
            **kwargs: Additional arguments passed to parent WordTTSService.
        """
        # Initialize WordTTSService first to set up word timestamp tracking
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,  # We'll push text frames based on word timestamps
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        # Initialize Azure-specific functionality from mixin
        self._init_azure_base(api_key=api_key, region=region, voice=voice, params=params)

        self._speech_config = None
        self._speech_synthesizer = None
        self._audio_queue = asyncio.Queue()
        self._word_boundary_queue = asyncio.Queue()
        self._word_processor_task = None
        self._started = False
        self._first_chunk = True
        self._cumulative_audio_offset: float = 0.0  # Cumulative audio duration in seconds
        self._current_sentence_base_offset: float = 0.0  # Base offset for current sentence
        self._current_sentence_duration: float = 0.0  # Duration from Azure callback
        self._current_sentence_max_word_offset: float = (
            0.0  # Max word boundary offset seen in current sentence (for 8kHz workaround)
        )
        self._last_word: Optional[str] = None  # Track last word for punctuation merging
        self._last_timestamp: Optional[float] = None  # Track last timestamp

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Azure TTS service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Azure TTS service and initialize speech synthesizer.

        Args:
            frame: Start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._speech_config:
            return

        # Now self.sample_rate is properly initialized
        self._speech_config = SpeechConfig(
            subscription=self._api_key,
            region=self._region,
        )
        self._speech_config.speech_synthesis_language = self._settings["language"]
        self._speech_config.set_speech_synthesis_output_format(
            sample_rate_to_output_format(self.sample_rate)
        )
        self._speech_config.set_service_property(
            "synthesizer.synthesis.connection.synthesisConnectionImpl",
            "websocket",
            ServicePropertyChannel.UriQueryParameter,
        )

        self._speech_synthesizer = SpeechSynthesizer(
            speech_config=self._speech_config, audio_config=None
        )

        # Set up event handlers
        self._speech_synthesizer.synthesizing.connect(self._handle_synthesizing)
        self._speech_synthesizer.synthesis_completed.connect(self._handle_completed)
        self._speech_synthesizer.synthesis_canceled.connect(self._handle_canceled)
        self._speech_synthesizer.synthesis_word_boundary.connect(self._handle_word_boundary)

        # Start word processor task
        if not self._word_processor_task:
            self._word_processor_task = self.create_task(self._word_processor_task_handler())

    async def stop(self, frame: EndFrame):
        """Stop the Azure TTS service.

        Args:
            frame: End frame signaling service stop.
        """
        await super().stop(frame)
        await self.cancel_task(self._word_processor_task)
        self._word_processor_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the Azure TTS service.

        Args:
            frame: Cancel frame signaling service cancellation.
        """
        await super().cancel(frame)
        await self.cancel_task(self._word_processor_task)
        self._word_processor_task = None

    def _is_cjk_language(self) -> bool:
        """Check if the configured language is CJK (Chinese, Japanese, Korean).

        Returns:
            True if the language is CJK, False otherwise.
        """
        language = self._settings.get("language", "").lower()
        # Check if language starts with CJK language codes
        return language.startswith(("zh", "ja", "ko", "cmn", "yue", "wuu"))

    def _is_punctuation_only(self, text: str) -> bool:
        """Check if text consists only of punctuation and whitespace.

        Args:
            text: Text to check.

        Returns:
            True if text is only punctuation/whitespace, False otherwise.
        """
        return text and all(not c.isalnum() for c in text)

    def _handle_word_boundary(self, evt):
        """Handle word boundary events from Azure SDK.

        Azure sends punctuation as separate word boundaries, and breaks CJK text
        into individual characters/particles. This method routes to language-specific
        handlers to properly merge and emit word boundaries.

        Args:
            evt: SpeechSynthesisWordBoundaryEventArgs from Azure Speech SDK
                containing word text and audio offset timing.
        """
        # evt.text contains the word
        # evt.audio_offset contains timing in ticks (100-nanosecond units)
        # Convert ticks to seconds: divide by 10,000,000
        word = evt.text
        sentence_relative_seconds = evt.audio_offset / 10_000_000.0

        # Use base offset captured at start of run_tts to avoid race conditions
        # with callbacks from overlapping TTS requests
        absolute_seconds = self._current_sentence_base_offset + sentence_relative_seconds

        # Track max word offset for accurate cumulative timing
        # (audio_duration from Azure doesn't always match word boundary offsets at 8kHz)
        if sentence_relative_seconds > self._current_sentence_max_word_offset:
            self._current_sentence_max_word_offset = sentence_relative_seconds

        if not word:
            return

        # Route to language-specific handler
        if self._is_cjk_language():
            self._handle_cjk_word_boundary(word, absolute_seconds)
        else:
            self._handle_non_cjk_word_boundary(word, absolute_seconds)

    def _emit_pending_word(self):
        """Emit the currently buffered word if one exists."""
        if self._last_word is not None:
            self._word_boundary_queue.put_nowait((self._last_word, self._last_timestamp))
            self._last_word = None
            self._last_timestamp = None

    def _handle_cjk_word_boundary(self, word: str, timestamp: float):
        """Handle word boundaries for CJK languages (Chinese, Japanese, Korean).

        CJK languages don't use spaces between words, so we merge characters together
        and only emit at natural break points (punctuation or whitespace boundaries).
        Without this logic, we don't get word output for CJK languages.

        Args:
            word: The word/character from Azure.
            timestamp: Timestamp in seconds.
        """
        # First word: just store it
        if self._last_word is None:
            self._last_word = word
            self._last_timestamp = timestamp
            return

        # Punctuation: merge and emit (natural break)
        if self._is_punctuation_only(word):
            self._last_word += word
            self._emit_pending_word()
            return

        # Whitespace: emit before boundary, start new segment
        if word.strip() != word:
            self._emit_pending_word()
            self._last_word = word
            self._last_timestamp = timestamp
            return

        # Default: continue merging CJK characters
        self._last_word += word

    def _handle_non_cjk_word_boundary(self, word: str, timestamp: float):
        """Handle word boundaries for non-CJK languages.

        Non-CJK languages use spaces between words, so we emit each word separately
        after merging any trailing punctuation.

        Args:
            word: The word from Azure.
            timestamp: Timestamp in seconds.
        """
        # Punctuation: merge with previous word (don't emit yet)
        if self._is_punctuation_only(word) and self._last_word is not None:
            self._last_word += word
            return

        # Regular word: emit previous, store current
        if self._last_word is not None:
            self._word_boundary_queue.put_nowait((self._last_word, self._last_timestamp))
        self._last_word = word
        self._last_timestamp = timestamp

    async def _word_processor_task_handler(self):
        """Process word timestamps from the queue and call add_word_timestamps."""
        while True:
            try:
                word, timestamp_seconds = await self._word_boundary_queue.get()
                await self.add_word_timestamps([(word, timestamp_seconds)])
                self._word_boundary_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

    def _handle_synthesizing(self, evt):
        """Handle audio chunks as they arrive.

        Args:
            evt: Synthesis event containing audio data.
        """
        if evt.result and evt.result.audio_data:
            self._audio_queue.put_nowait(evt.result.audio_data)

    def _handle_completed(self, evt):
        """Handle synthesis completion.

        Args:
            evt: Completion event from Azure Speech SDK.
        """
        # Flush any pending word before completing
        if self._last_word is not None:
            self._word_boundary_queue.put_nowait((self._last_word, self._last_timestamp))
            self._last_word = None
            self._last_timestamp = None

        # Store duration for cumulative offset calculation
        if evt.result and evt.result.audio_duration:
            self._current_sentence_duration = evt.result.audio_duration.total_seconds()

        self._audio_queue.put_nowait(None)  # Signal completion

    def _handle_canceled(self, evt):
        """Handle synthesis cancellation.

        Args:
            evt: Cancellation event.
        """
        reason = evt.result.cancellation_details.reason
        # User cancellation (from interruption) is expected, not an error
        if reason == CancellationReason.CancelledByUser:
            logger.debug(f"{self}: Speech synthesis canceled by user (interruption)")
        else:
            logger.warning(f"{self}: Speech synthesis canceled: {reason}")
        self._audio_queue.put_nowait(None)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._reset_state()
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("Reset", 0)])

    def _reset_state(self):
        """Reset TTS state between turns."""
        self._started = False
        self._first_chunk = True
        self._cumulative_audio_offset = 0.0
        self._current_sentence_base_offset = 0.0
        self._current_sentence_duration = 0.0
        self._current_sentence_max_word_offset = 0.0
        self._last_word = None
        self._last_timestamp = None

    async def flush_audio(self):
        """Flush any pending audio data."""
        logger.trace(f"{self}: flushing audio")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruption by stopping current synthesis.

        Args:
            frame: The interruption frame.
            direction: Frame processing direction.
        """
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()

        # Stop Azure synthesis to prevent more word boundaries from being added
        if self._speech_synthesizer:
            try:
                # stop_speaking_async() returns a ResultFuture
                # We need to call .get() in a thread to wait for completion
                result_future = self._speech_synthesizer.stop_speaking_async()
                await asyncio.to_thread(result_future.get)
            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

        # Reset state on interruption
        self._reset_state()
        # Clear the audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
            except asyncio.QueueEmpty:
                break
        # Clear the word boundary queue
        while not self._word_boundary_queue.empty():
            try:
                self._word_boundary_queue.get_nowait()
                self._word_boundary_queue.task_done()
            except asyncio.QueueEmpty:
                break

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Azure's streaming synthesis.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing synthesized speech data.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Clear the audio queue in case there's still audio in it, causing the next audio response
        # to be cut off by the 'None' element returned at the end of the previous audio synthesis.
        # Empty the audio queue before processing the new text
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()
            self._audio_queue.task_done()

        try:
            if self._speech_synthesizer is None:
                return

            try:
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True
                    self._first_chunk = True

                # Capture base offset BEFORE starting synthesis to avoid race conditions
                # Word boundary callbacks will use this value
                self._current_sentence_base_offset = self._cumulative_audio_offset
                self._current_sentence_duration = 0.0
                self._current_sentence_max_word_offset = 0.0

                ssml = self._construct_ssml(text)
                self._speech_synthesizer.speak_ssml_async(ssml)
                await self.start_tts_usage_metrics(text)

                # Stream audio chunks as they arrive

                while True:
                    chunk = await self._audio_queue.get()
                    if chunk is None:  # End of stream
                        break

                    if self._first_chunk:
                        await self.stop_ttfb_metrics()
                        await self.start_word_timestamps()
                        self._first_chunk = False

                    frame = TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )
                    yield frame

                # Update cumulative offset for next sentence
                # At 8kHz, Azure's audio_duration doesn't match word boundary offsets,
                # so we use max_word_offset as a workaround. At other sample rates,
                # audio_duration is accurate.
                # TODO: Remove after Azure fixes word boundary timing at 8kHz
                if self.sample_rate == 8000:
                    self._cumulative_audio_offset += self._current_sentence_max_word_offset
                else:
                    self._cumulative_audio_offset += self._current_sentence_duration

            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame()
                self._reset_state()
                return

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


class AzureHttpTTSService(TTSService, AzureBaseTTSService):
    """Azure Cognitive Services HTTP-based TTS service.

    Provides text-to-speech synthesis using Azure's HTTP API for simpler,
    non-streaming synthesis. Suitable for use cases where streaming is not
    required and simpler integration is preferred.
    """

    def __init__(
        self,
        *,
        api_key: str,
        region: str,
        voice: str = "en-US-SaraNeural",
        sample_rate: Optional[int] = None,
        params: Optional[AzureBaseTTSService.InputParams] = None,
        **kwargs,
    ):
        """Initialize the Azure HTTP TTS service.

        Args:
            api_key: Azure Cognitive Services subscription key.
            region: Azure region identifier (e.g., "eastus", "westus2").
            voice: Voice name to use for synthesis. Defaults to "en-US-SaraNeural".
            sample_rate: Audio sample rate in Hz. If None, uses service default.
            params: Voice and synthesis parameters configuration.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Initialize Azure-specific functionality from mixin
        self._init_azure_base(api_key=api_key, region=region, voice=voice, params=params)

        self._speech_config = None
        self._speech_synthesizer = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Azure TTS service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Azure HTTP TTS service and initialize speech synthesizer.

        Args:
            frame: Start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._speech_config:
            return

        self._speech_config = SpeechConfig(
            subscription=self._api_key,
            region=self._region,
        )
        self._speech_config.speech_synthesis_language = self._settings["language"]
        self._speech_config.set_speech_synthesis_output_format(
            sample_rate_to_output_format(self.sample_rate)
        )
        self._speech_synthesizer = SpeechSynthesizer(
            speech_config=self._speech_config, audio_config=None
        )

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Azure's HTTP synthesis API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the complete synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        await self.start_ttfb_metrics()

        ssml = self._construct_ssml(text)

        result = await asyncio.to_thread(self._speech_synthesizer.speak_ssml, ssml)

        if result.reason == ResultReason.SynthesizingAudioCompleted:
            await self.start_tts_usage_metrics(text)
            await self.stop_ttfb_metrics()
            yield TTSStartedFrame()
            # Azure always sends a 44-byte header. Strip it off.
            yield TTSAudioRawFrame(
                audio=result.audio_data[44:],
                sample_rate=self.sample_rate,
                num_channels=1,
            )
            yield TTSStoppedFrame()
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.warning(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == CancellationReason.Error:
                yield ErrorFrame(
                    error=f"Unknown error occurred: {cancellation_details.error_details}"
                )
