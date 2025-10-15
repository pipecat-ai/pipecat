#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ElevenLabs text-to-speech service implementations.

This module provides WebSocket and HTTP-based TTS services using ElevenLabs API
with support for streaming audio, word timestamps, and voice customization.
"""

import asyncio
import base64
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Tuple, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import (
    AudioContextWordTTSService,
    WordTTSService,
)
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

# See .env.example for ElevenLabs configuration needed
try:
    import websockets
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use ElevenLabs, you need to `pip install pipecat-ai[elevenlabs]`.")
    raise Exception(f"Missing module: {e}")

ElevenLabsOutputFormat = Literal["pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100"]

# Models that support language codes
# The following models are excluded as they don't support language codes:
# - eleven_flash_v2
# - eleven_turbo_v2
# - eleven_multilingual_v2
ELEVENLABS_MULTILINGUAL_MODELS = {
    "eleven_flash_v2_5",
    "eleven_turbo_v2_5",
}


def language_to_elevenlabs_language(language: Language) -> Optional[str]:
    """Convert a Language enum to ElevenLabs language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding ElevenLabs language code, or None if not supported.
    """
    BASE_LANGUAGES = {
        Language.AR: "ar",
        Language.BG: "bg",
        Language.CS: "cs",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.FI: "fi",
        Language.FIL: "fil",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.MS: "ms",
        Language.NL: "nl",
        Language.NO: "no",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SV: "sv",
        Language.TA: "ta",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.VI: "vi",
        Language.ZH: "zh",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. es-ES -> es)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


def output_format_from_sample_rate(sample_rate: int) -> str:
    """Get the appropriate output format string for a given sample rate.

    Args:
        sample_rate: The audio sample rate in Hz.

    Returns:
        The ElevenLabs output format string.
    """
    match sample_rate:
        case 8000:
            return "pcm_8000"
        case 16000:
            return "pcm_16000"
        case 22050:
            return "pcm_22050"
        case 24000:
            return "pcm_24000"
        case 44100:
            return "pcm_44100"
    logger.warning(
        f"ElevenLabsTTSService: No output format available for {sample_rate} sample rate"
    )
    return "pcm_24000"


def build_elevenlabs_voice_settings(
    settings: Dict[str, Any],
) -> Optional[Dict[str, Union[float, bool]]]:
    """Build voice settings dictionary for ElevenLabs based on provided settings.

    Args:
        settings: Dictionary containing voice settings parameters.

    Returns:
        Dictionary of voice settings or None if no valid settings are provided.
    """
    voice_setting_keys = ["stability", "similarity_boost", "style", "use_speaker_boost", "speed"]

    voice_settings = {}
    for key in voice_setting_keys:
        if key in settings and settings[key] is not None:
            voice_settings[key] = settings[key]

    return voice_settings or None


def calculate_word_times(
    alignment_info: Mapping[str, Any],
    cumulative_time: float,
    partial_word: str = "",
    partial_word_start_time: float = 0.0,
) -> tuple[List[Tuple[str, float]], str, float]:
    """Calculate word timestamps from character alignment information.

    Args:
        alignment_info: Character alignment data from ElevenLabs API.
        cumulative_time: Base time offset for this chunk.
        partial_word: Partial word carried over from previous chunk.
        partial_word_start_time: Start time of the partial word.

    Returns:
        Tuple of (word_times, new_partial_word, new_partial_word_start_time):
        - word_times: List of (word, timestamp) tuples for complete words
        - new_partial_word: Incomplete word at end of chunk (empty if chunk ends with space)
        - new_partial_word_start_time: Start time of the incomplete word
    """
    chars = alignment_info["chars"]
    char_start_times_ms = alignment_info["charStartTimesMs"]

    if len(chars) != len(char_start_times_ms):
        logger.error(
            f"calculate_word_times: length mismatch - chars={len(chars)}, times={len(char_start_times_ms)}"
        )
        return ([], partial_word, partial_word_start_time)

    # Build words and track their start positions
    words = []
    word_start_times = []
    current_word = partial_word  # Start with any partial word from previous chunk
    word_start_time = partial_word_start_time if partial_word else None

    for i, char in enumerate(chars):
        if char == " ":
            # End of current word
            if current_word:  # Only add non-empty words
                words.append(current_word)
                word_start_times.append(word_start_time)
                current_word = ""
                word_start_time = None
        else:
            # Building a word
            if word_start_time is None:  # First character of new word
                # Convert from milliseconds to seconds and add cumulative offset
                word_start_time = cumulative_time + (char_start_times_ms[i] / 1000.0)
            current_word += char

    # Build result for complete words
    word_times = list(zip(words, word_start_times))

    # Return any incomplete word at the end of this chunk
    new_partial_word = current_word if current_word else ""
    new_partial_word_start_time = word_start_time if word_start_time is not None else 0.0

    return (word_times, new_partial_word, new_partial_word_start_time)


class ElevenLabsTTSService(AudioContextWordTTSService):
    """ElevenLabs WebSocket-based TTS service with word timestamps.

    Provides real-time text-to-speech using ElevenLabs' WebSocket streaming API.
    Supports word-level timestamps, audio context management, and various voice
    customization options including stability, similarity boost, and speed controls.
    """

    class InputParams(BaseModel):
        """Input parameters for ElevenLabs TTS configuration.

        Parameters:
            language: Language to use for synthesis.
            stability: Voice stability control (0.0 to 1.0).
            similarity_boost: Similarity boost control (0.0 to 1.0).
            style: Style control for voice expression (0.0 to 1.0).
            use_speaker_boost: Whether to use speaker boost enhancement.
            speed: Voice speed control (0.7 to 1.2).
            auto_mode: Whether to enable automatic mode optimization.
            enable_ssml_parsing: Whether to parse SSML tags in text.
            enable_logging: Whether to enable ElevenLabs logging.
            apply_text_normalization: Text normalization mode ("auto", "on", "off").
        """

        language: Optional[Language] = None
        stability: Optional[float] = None
        similarity_boost: Optional[float] = None
        style: Optional[float] = None
        use_speaker_boost: Optional[bool] = None
        speed: Optional[float] = None
        auto_mode: Optional[bool] = True
        enable_ssml_parsing: Optional[bool] = None
        enable_logging: Optional[bool] = None
        apply_text_normalization: Optional[Literal["auto", "on", "off"]] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "eleven_turbo_v2_5",
        url: str = "wss://api.elevenlabs.io",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        aggregate_sentences: Optional[bool] = True,
        **kwargs,
    ):
        """Initialize the ElevenLabs TTS service.

        Args:
            api_key: ElevenLabs API key for authentication.
            voice_id: ID of the voice to use for synthesis.
            model: TTS model to use (e.g., "eleven_turbo_v2_5").
            url: WebSocket URL for ElevenLabs TTS API.
            sample_rate: Audio sample rate. If None, uses default.
            params: Additional input parameters for voice customization.
            aggregate_sentences: Whether to aggregate sentences within the TTSService.
            **kwargs: Additional arguments passed to the parent service.
        """
        # Aggregating sentences still gives cleaner-sounding results and fewer
        # artifacts than streaming one word at a time. On average, waiting for a
        # full sentence should only "cost" us 15ms or so with GPT-4o or a Llama
        # 3 model, and it's worth it for the better audio quality.
        #
        # We also don't want to automatically push LLM response text frames,
        # because the context aggregators will add them to the LLM context even
        # if we're interrupted. ElevenLabs gives us word-by-word timestamps. We
        # can use those to generate text frames ourselves aligned with the
        # playout timing of the audio!
        #
        # Finally, ElevenLabs doesn't provide information on when the bot stops
        # speaking for a while, so we want the parent class to send TTSStopFrame
        # after a short period not receiving any audio.
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or ElevenLabsTTSService.InputParams()

        self._api_key = api_key
        self._url = url
        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else None,
            "stability": params.stability,
            "similarity_boost": params.similarity_boost,
            "style": params.style,
            "use_speaker_boost": params.use_speaker_boost,
            "speed": params.speed,
            "auto_mode": str(params.auto_mode).lower(),
            "enable_ssml_parsing": params.enable_ssml_parsing,
            "enable_logging": params.enable_logging,
            "apply_text_normalization": params.apply_text_normalization,
        }
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._output_format = ""  # initialized in start()
        self._voice_settings = self._set_voice_settings()

        # Indicates if we have sent TTSStartedFrame. It will reset to False when
        # there's an interruption or TTSStoppedFrame.
        self._started = False
        self._cumulative_time = 0
        # Track partial words that span across alignment chunks
        self._partial_word = ""
        self._partial_word_start_time = 0.0

        # Context management for v1 multi API
        self._context_id = None
        self._receive_task = None
        self._keepalive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as ElevenLabs service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to ElevenLabs language format.

        Args:
            language: The language to convert.

        Returns:
            The ElevenLabs-specific language code, or None if not supported.
        """
        return language_to_elevenlabs_language(language)

    def _set_voice_settings(self):
        return build_elevenlabs_voice_settings(self._settings)

    async def set_model(self, model: str):
        """Set the TTS model and reconnect.

        Args:
            model: The model name to use for synthesis.
        """
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")
        await self._disconnect()
        await self._connect()

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings and reconnect if voice, model, or language changed."""
        # Track previous values for settings that require reconnection
        prev_voice = self._voice_id
        prev_model = self.model_name
        prev_language = self._settings.get("language")
        # Create snapshot of current voice settings to detect changes after update
        prev_voice_settings = self._voice_settings.copy() if self._voice_settings else None

        await super()._update_settings(settings)

        # Update voice settings for the next context creation
        self._voice_settings = self._set_voice_settings()

        # Check if URL-level settings changed (these require reconnection)
        url_changed = (
            prev_voice != self._voice_id
            or prev_model != self.model_name
            or prev_language != self._settings.get("language")
        )

        # Check if only voice settings changed (speed, stability, etc.)
        voice_settings_changed = prev_voice_settings != self._voice_settings

        if url_changed:
            # These settings are in the WebSocket URL, so we need to reconnect
            logger.debug(
                f"URL-level setting changed (voice/model/language), reconnecting WebSocket"
            )
            await self._disconnect()
            await self._connect()
        elif voice_settings_changed and self._context_id:
            # Voice settings can be updated by closing current context
            # so new one gets created with updated voice settings
            logger.debug(f"Voice settings changed, closing current context to apply changes")
            try:
                if self._websocket:
                    await self._websocket.send(
                        json.dumps({"context_id": self._context_id, "close_context": True})
                    )
            except Exception as e:
                logger.warning(f"Error closing context for voice settings update: {e}")
            self._context_id = None
            self._started = False

    async def start(self, frame: StartFrame):
        """Start the ElevenLabs TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._output_format = output_format_from_sample_rate(self.sample_rate)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the ElevenLabs TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the ElevenLabs TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        """Flush any pending audio and finalize the current context."""
        if not self._context_id or not self._websocket:
            return
        logger.trace(f"{self}: flushing audio")
        msg = {"context_id": self._context_id, "flush": True}
        await self._websocket.send(json.dumps(msg))

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._started = False
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("Reset", 0)])

    async def _connect(self):
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to ElevenLabs")

            voice_id = self._voice_id
            model = self.model_name
            output_format = self._output_format
            url = f"{self._url}/v1/text-to-speech/{voice_id}/multi-stream-input?model_id={model}&output_format={output_format}&auto_mode={self._settings['auto_mode']}"

            if self._settings["enable_ssml_parsing"]:
                url += f"&enable_ssml_parsing={self._settings['enable_ssml_parsing']}"

            if self._settings["enable_logging"]:
                url += f"&enable_logging={self._settings['enable_logging']}"

            if self._settings["apply_text_normalization"] is not None:
                url += f"&apply_text_normalization={self._settings['apply_text_normalization']}"

            # Language can only be used with the ELEVENLABS_MULTILINGUAL_MODELS
            language = self._settings["language"]
            if model in ELEVENLABS_MULTILINGUAL_MODELS and language is not None:
                url += f"&language_code={language}"
                logger.debug(f"Using language code: {language}")
            elif language is not None:
                logger.warning(
                    f"Language code [{language}] not applied. Language codes can only be used with multilingual models: {', '.join(sorted(ELEVENLABS_MULTILINGUAL_MODELS))}"
                )

            # Set max websocket message size to 16MB for large audio responses
            self._websocket = await websocket_connect(
                url, max_size=16 * 1024 * 1024, additional_headers={"xi-api-key": self._api_key}
            )

            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from ElevenLabs")
                # Close all contexts and the socket
                if self._context_id:
                    await self._websocket.send(json.dumps({"close_socket": True}))
                await self._websocket.close()
                logger.debug("Disconnected from ElevenLabs")
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._started = False
            self._context_id = None
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruption by closing the current context."""
        await super()._handle_interruption(frame, direction)

        # Close the current context when interrupted without closing the websocket
        if self._context_id and self._websocket:
            logger.trace(f"Closing context {self._context_id} due to interruption")
            try:
                # ElevenLabs requires that Pipecat manages the contexts and closes them
                # when they're not longer in use. Since an InterruptionFrame is pushed
                # every time the user speaks, we'll use this as a trigger to close the context
                # and reset the state.
                # Note: We do not need to call remove_audio_context here, as the context is
                # automatically reset when super ()._handle_interruption is called.
                await self._websocket.send(
                    json.dumps({"context_id": self._context_id, "close_context": True})
                )
            except Exception as e:
                logger.error(f"Error closing context on interruption: {e}")
            self._context_id = None
            self._started = False
            self._partial_word = ""
            self._partial_word_start_time = 0.0

    async def _receive_messages(self):
        """Handle incoming WebSocket messages from ElevenLabs."""
        async for message in self._get_websocket():
            msg = json.loads(message)

            received_ctx_id = msg.get("contextId")

            # Handle final messages first, regardless of context availability
            # At the moment, this message is received AFTER the close_context message is
            # sent, so it doesn't serve any functional purpose. For now, we'll just log it.
            if msg.get("isFinal") is True:
                logger.trace(f"Received final message for context {received_ctx_id}")
                continue

            # Check if this message belongs to the current context.
            if not self.audio_context_available(received_ctx_id):
                if self._context_id == received_ctx_id:
                    logger.debug(
                        f"Received a delayed message, recreating the context: {self._context_id}"
                    )
                    await self.create_audio_context(self._context_id)
                else:
                    # This can happen if a message is received _after_ we have closed a context
                    # due to user interruption but _before_ the `isFinal` message for the context
                    # is received.
                    logger.debug(f"Ignoring message from unavailable context: {received_ctx_id}")
                    continue

            if msg.get("audio"):
                await self.stop_ttfb_metrics()
                self.start_word_timestamps()

                audio = base64.b64decode(msg["audio"])
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                await self.append_to_audio_context(received_ctx_id, frame)

            if msg.get("alignment"):
                alignment = msg["alignment"]
                word_times, self._partial_word, self._partial_word_start_time = (
                    calculate_word_times(
                        alignment,
                        self._cumulative_time,
                        self._partial_word,
                        self._partial_word_start_time,
                    )
                )

                if word_times:
                    await self.add_word_timestamps(word_times)

                    # Calculate the actual end time of this audio chunk
                    char_start_times_ms = alignment.get("charStartTimesMs", [])
                    char_durations_ms = alignment.get("charDurationsMs", [])

                    if char_start_times_ms and char_durations_ms:
                        # End time = start time of last character + duration of last character
                        chunk_end_time_ms = char_start_times_ms[-1] + char_durations_ms[-1]
                        chunk_end_time_seconds = chunk_end_time_ms / 1000.0
                        self._cumulative_time += chunk_end_time_seconds
                    else:
                        # Fallback: use the last word's start time (current behavior)
                        self._cumulative_time = word_times[-1][1]
                        logger.warning(
                            "_receive_messages: using fallback timing method - consider investigating alignment data structure"
                        )

    async def _keepalive_task_handler(self):
        """Send periodic keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 10
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            try:
                if self._websocket and self._websocket.state is State.OPEN:
                    if self._context_id:
                        # Send keepalive with context ID to keep the connection alive
                        keepalive_message = {
                            "text": "",
                            "context_id": self._context_id,
                        }
                        logger.trace(f"Sending keepalive for context {self._context_id}")
                    else:
                        # It's possible to have a user interruption which clears the context
                        # without generating a new TTS response. In this case, we'll just send
                        # an empty message to keep the connection alive.
                        keepalive_message = {"text": ""}
                        logger.trace("Sending keepalive without context")
                    await self._websocket.send(json.dumps(keepalive_message))
            except websockets.ConnectionClosed as e:
                logger.warning(f"{self} keepalive error: {e}")
                break

    async def _send_text(self, text: str):
        """Send text to the WebSocket for synthesis."""
        if self._websocket and self._context_id:
            msg = {"text": text, "context_id": self._context_id}
            await self._websocket.send(json.dumps(msg))

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using ElevenLabs' streaming WebSocket API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True
                    self._cumulative_time = 0
                    self._partial_word = ""
                    self._partial_word_start_time = 0.0
                    # If a context ID does not exist, create a new one and
                    # register it. If an ID exists, that means the Pipeline is
                    # configured for allow_interruptions=False, so continue
                    # using the current ID. When interruptions are enabled
                    # (e.g. allow_interruptions=True), user speech results in
                    # an interruption, which resets the context ID.
                    if not self._context_id:
                        self._context_id = str(uuid.uuid4())
                    if not self.audio_context_available(self._context_id):
                        await self.create_audio_context(self._context_id)

                    # Initialize context with voice settings
                    msg = {"text": " ", "context_id": self._context_id}
                    if self._voice_settings:
                        msg["voice_settings"] = self._voice_settings
                    await self._websocket.send(json.dumps(msg))
                    logger.trace(f"Created new context {self._context_id} with voice settings")

                    await self._send_text(text)
                    await self.start_tts_usage_metrics(text)
                else:
                    await self._send_text(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                self._started = False
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")


class ElevenLabsHttpTTSService(WordTTSService):
    """ElevenLabs HTTP-based TTS service with word timestamps.

    Provides text-to-speech using ElevenLabs' HTTP streaming API for simpler,
    non-WebSocket integration. Suitable for use cases where streaming WebSocket
    connection is not required or desired.
    """

    class InputParams(BaseModel):
        """Input parameters for ElevenLabs HTTP TTS configuration.

        Parameters:
            language: Language to use for synthesis.
            optimize_streaming_latency: Latency optimization level (0-4).
            stability: Voice stability control (0.0 to 1.0).
            similarity_boost: Similarity boost control (0.0 to 1.0).
            style: Style control for voice expression (0.0 to 1.0).
            use_speaker_boost: Whether to use speaker boost enhancement.
            speed: Voice speed control (0.25 to 4.0).
            apply_text_normalization: Text normalization mode ("auto", "on", "off").
        """

        language: Optional[Language] = None
        optimize_streaming_latency: Optional[int] = None
        stability: Optional[float] = None
        similarity_boost: Optional[float] = None
        style: Optional[float] = None
        use_speaker_boost: Optional[bool] = None
        speed: Optional[float] = None
        apply_text_normalization: Optional[Literal["auto", "on", "off"]] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "eleven_turbo_v2_5",
        base_url: str = "https://api.elevenlabs.io",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        aggregate_sentences: Optional[bool] = True,
        **kwargs,
    ):
        """Initialize the ElevenLabs HTTP TTS service.

        Args:
            api_key: ElevenLabs API key for authentication.
            voice_id: ID of the voice to use for synthesis.
            aiohttp_session: aiohttp ClientSession for HTTP requests.
            model: TTS model to use (e.g., "eleven_turbo_v2_5").
            base_url: Base URL for ElevenLabs HTTP API.
            sample_rate: Audio sample rate. If None, uses default.
            params: Additional input parameters for voice customization.
            aggregate_sentences: Whether to aggregate sentences within the TTSService.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
            push_stop_frames=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or ElevenLabsHttpTTSService.InputParams()

        self._api_key = api_key
        self._base_url = base_url
        self._params = params
        self._session = aiohttp_session

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else None,
            "optimize_streaming_latency": params.optimize_streaming_latency,
            "stability": params.stability,
            "similarity_boost": params.similarity_boost,
            "style": params.style,
            "use_speaker_boost": params.use_speaker_boost,
            "speed": params.speed,
            "apply_text_normalization": params.apply_text_normalization,
        }
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._output_format = ""  # initialized in start()
        self._voice_settings = self._set_voice_settings()

        # Track cumulative time to properly sequence word timestamps across utterances
        self._cumulative_time = 0
        self._started = False

        # Store previous text for context within a turn
        self._previous_text = ""

        # Track partial words that span across alignment chunks
        self._partial_word = ""
        self._partial_word_start_time = 0.0

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert pipecat Language to ElevenLabs language code.

        Args:
            language: The language to convert.

        Returns:
            The ElevenLabs-specific language code, or None if not supported.
        """
        return language_to_elevenlabs_language(language)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as ElevenLabs HTTP service supports metrics generation.
        """
        return True

    def _set_voice_settings(self):
        return build_elevenlabs_voice_settings(self._settings)

    def _reset_state(self):
        """Reset internal state variables."""
        self._cumulative_time = 0
        self._started = False
        self._previous_text = ""
        self._partial_word = ""
        self._partial_word_start_time = 0.0
        logger.debug(f"{self}: Reset internal state")

    async def start(self, frame: StartFrame):
        """Start the ElevenLabs HTTP TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._output_format = output_format_from_sample_rate(self.sample_rate)
        self._reset_state()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, TTSStoppedFrame)):
            # Reset timing on interruption or stop
            self._reset_state()

            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("Reset", 0)])

        elif isinstance(frame, LLMFullResponseEndFrame):
            # End of turn - reset previous text
            self._previous_text = ""

    def calculate_word_times(self, alignment_info: Mapping[str, Any]) -> List[Tuple[str, float]]:
        """Calculate word timing from character alignment data.

        This method handles partial words that may span across multiple alignment chunks.

        Args:
            alignment_info: Character timing data from ElevenLabs.

        Returns:
            List of (word, timestamp) pairs for complete words in this chunk.

        Example input data::

            {
                "characters": [" ", "H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"],
                "character_start_times_seconds": [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "character_end_times_seconds": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }

        Would produce word times (with cumulative_time=0)::

            [("Hello", 0.1), ("world", 0.5)]
        """
        chars = alignment_info.get("characters", [])
        char_start_times = alignment_info.get("character_start_times_seconds", [])

        if not chars or not char_start_times or len(chars) != len(char_start_times):
            logger.warning(
                f"Invalid alignment data: chars={len(chars)}, times={len(char_start_times)}"
            )
            return []

        # Build the words and find their start times
        words = []
        word_start_times = []
        # Start with any partial word from previous chunk
        current_word = self._partial_word
        word_start_time = self._partial_word_start_time if self._partial_word else None

        for i, char in enumerate(chars):
            if char == " ":
                if current_word:  # Only add non-empty words
                    words.append(current_word)
                    word_start_times.append(word_start_time)
                    current_word = ""
                    word_start_time = None
            else:
                if word_start_time is None:  # First character of a new word
                    # Use time of the first character of the word, offset by cumulative time
                    word_start_time = self._cumulative_time + char_start_times[i]
                current_word += char

        # Store any incomplete word at the end of this chunk
        self._partial_word = current_word if current_word else ""
        self._partial_word_start_time = word_start_time if word_start_time is not None else 0.0

        # Create word-time pairs for complete words only
        word_times = list(zip(words, word_start_times))

        return word_times

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using ElevenLabs streaming API with timestamps.

        Makes a request to the ElevenLabs API to generate audio and timing data.
        Tracks the duration of each utterance to ensure correct sequencing.
        Includes previous text as context for better prosody continuity.

        Args:
            text: Text to convert to speech.

        Yields:
            Frame: Audio and control frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Use the with-timestamps endpoint
        url = f"{self._base_url}/v1/text-to-speech/{self._voice_id}/stream/with-timestamps"

        payload: Dict[str, Union[str, Dict[str, Union[float, bool]]]] = {
            "text": text,
            "model_id": self._model_name,
        }

        # Include previous text as context if available
        if self._previous_text:
            payload["previous_text"] = self._previous_text

        if self._voice_settings:
            payload["voice_settings"] = self._voice_settings

        if self._settings["apply_text_normalization"] is not None:
            payload["apply_text_normalization"] = self._settings["apply_text_normalization"]

        language = self._settings["language"]
        if self._model_name in ELEVENLABS_MULTILINGUAL_MODELS and language:
            payload["language_code"] = language
            logger.debug(f"Using language code: {language}")
        elif language:
            logger.warning(
                f"Language code [{language}] not applied. Language codes can only be used with multilingual models: {', '.join(sorted(ELEVENLABS_MULTILINGUAL_MODELS))}"
            )

        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        # Build query parameters
        params = {
            "output_format": self._output_format,
        }
        if self._settings["optimize_streaming_latency"] is not None:
            params["optimize_streaming_latency"] = self._settings["optimize_streaming_latency"]

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(
                url, json=payload, headers=headers, params=params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"{self} error: {error_text}")
                    yield ErrorFrame(error=f"ElevenLabs API error: {error_text}")
                    return

                await self.start_tts_usage_metrics(text)

                # Start TTS sequence if not already started
                if not self._started:
                    self.start_word_timestamps()
                    yield TTSStartedFrame()
                    self._started = True

                # Track the duration of this utterance based on the last character's end time
                utterance_duration = 0
                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue

                    try:
                        # Parse the JSON object
                        data = json.loads(line_str)

                        # Process audio if present
                        if data and "audio_base64" in data:
                            await self.stop_ttfb_metrics()
                            audio = base64.b64decode(data["audio_base64"])
                            yield TTSAudioRawFrame(audio, self.sample_rate, 1)

                        # Process alignment if present
                        if data and "alignment" in data:
                            alignment = data["alignment"]
                            if alignment:  # Ensure alignment is not None
                                # Get end time of the last character in this chunk
                                char_end_times = alignment.get("character_end_times_seconds", [])
                                if char_end_times:
                                    chunk_end_time = char_end_times[-1]
                                    # Update to the longest end time seen so far
                                    utterance_duration = max(utterance_duration, chunk_end_time)

                                # Calculate word timestamps
                                word_times = self.calculate_word_times(alignment)
                                if word_times:
                                    await self.add_word_timestamps(word_times)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON from stream: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing response: {e}", exc_info=True)
                        continue

                # After processing all chunks, emit any remaining partial word
                # since this is the end of the utterance
                if self._partial_word:
                    final_word_time = [(self._partial_word, self._partial_word_start_time)]
                    await self.add_word_timestamps(final_word_time)
                    self._partial_word = ""
                    self._partial_word_start_time = 0.0

                # After processing all chunks, add the total utterance duration
                # to the cumulative time to ensure next utterance starts after this one
                if utterance_duration > 0:
                    self._cumulative_time += utterance_duration

                # Append the current text to previous_text for context continuity
                # Only add a space if there's already text
                if self._previous_text:
                    self._previous_text += " " + text
                else:
                    self._previous_text = text

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            await self.stop_ttfb_metrics()
            # Let the parent class handle TTSStoppedFrame
