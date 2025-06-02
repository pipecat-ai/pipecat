#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
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
        settings: Dictionary containing voice settings parameters

    Returns:
        Dictionary of voice settings or None if no valid settings are provided
    """
    voice_setting_keys = ["stability", "similarity_boost", "style", "use_speaker_boost", "speed"]

    voice_settings = {}
    for key in voice_setting_keys:
        if key in settings and settings[key] is not None:
            voice_settings[key] = settings[key]

    return voice_settings or None


def calculate_word_times(
    alignment_info: Mapping[str, Any], cumulative_time: float
) -> List[Tuple[str, float]]:
    zipped_times = list(zip(alignment_info["chars"], alignment_info["charStartTimesMs"]))

    words = "".join(alignment_info["chars"]).split(" ")

    # Calculate start time for each word. We do this by finding a space character
    # and using the previous word time, also taking into account there might not
    # be a space at the end.
    times = []
    for i, (a, b) in enumerate(zipped_times):
        if a == " " or i == len(zipped_times) - 1:
            t = cumulative_time + (zipped_times[i - 1][1] / 1000.0)
            times.append(t)

    word_times = list(zip(words, times))

    return word_times


class ElevenLabsTTSService(AudioContextWordTTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = None
        stability: Optional[float] = None
        similarity_boost: Optional[float] = None
        style: Optional[float] = None
        use_speaker_boost: Optional[bool] = None
        speed: Optional[float] = None
        auto_mode: Optional[bool] = True
        enable_ssml_parsing: Optional[bool] = None
        enable_logging: Optional[bool] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "eleven_flash_v2_5",
        url: str = "wss://api.elevenlabs.io",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
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
            aggregate_sentences=True,
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
        }
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._output_format = ""  # initialized in start()
        self._voice_settings = self._set_voice_settings()

        # Indicates if we have sent TTSStartedFrame. It will reset to False when
        # there's an interruption or TTSStoppedFrame.
        self._started = False
        self._cumulative_time = 0

        # Context management for v1 multi API
        self._context_id = None
        self._receive_task = None
        self._keepalive_task = None

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_elevenlabs_language(language)

    def _set_voice_settings(self):
        return build_elevenlabs_voice_settings(self._settings)

    async def set_model(self, model: str):
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")
        await self._disconnect()
        await self._connect()

    async def _update_settings(self, settings: Mapping[str, Any]):
        prev_voice = self._voice_id
        await super()._update_settings(settings)
        if not prev_voice == self._voice_id:
            logger.info(f"Switching TTS voice to: [{self._voice_id}]")
            await self._disconnect()
            await self._connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._output_format = output_format_from_sample_rate(self.sample_rate)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        if self._websocket and self._context_id:
            msg = {"context_id": self._context_id, "flush": True}
            await self._websocket.send(json.dumps(msg))

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
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
            if self._websocket and self._websocket.open:
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
            self._websocket = await websockets.connect(
                url, max_size=16 * 1024 * 1024, extra_headers={"xi-api-key": self._api_key}
            )

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
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._started = False
            self._context_id = None
            self._websocket = None

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)

        # Close the current context when interrupted without closing the websocket
        if self._context_id and self._websocket:
            logger.trace(f"Closing context {self._context_id} due to interruption")
            try:
                await self._websocket.send(
                    json.dumps({"context_id": self._context_id, "close_context": True})
                )
            except Exception as e:
                logger.error(f"Error closing context on interruption: {e}")
            self._context_id = None
            self._started = False

    async def _receive_messages(self):
        async for message in self._get_websocket():
            msg = json.loads(message)
            # Check if this message belongs to the current context
            received_ctx_id = msg.get("contextId")
            if not self.audio_context_available(received_ctx_id):
                logger.trace(f"Ignoring message from unavailable context: {received_ctx_id}")
                continue

            if msg.get("audio"):
                await self.stop_ttfb_metrics()
                self.start_word_timestamps()

                audio = base64.b64decode(msg["audio"])
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                await self.append_to_audio_context(received_ctx_id, frame)
            if msg.get("alignment"):
                word_times = calculate_word_times(msg["alignment"], self._cumulative_time)
                await self.add_word_timestamps(word_times)
                self._cumulative_time = word_times[-1][1]
            if msg.get("isFinal"):
                logger.trace(f"Received final message for context {received_ctx_id}")
                await self.remove_audio_context(received_ctx_id)
                # Reset context tracking if this was our active context
                if self._context_id == received_ctx_id:
                    self._context_id = None
                    self._started = False

    async def _keepalive_task_handler(self):
        while True:
            await asyncio.sleep(10)
            try:
                # Send an empty message to keep the connection alive
                if self._websocket and self._websocket.open:
                    await self._websocket.send(json.dumps({}))
            except websockets.ConnectionClosed as e:
                logger.warning(f"{self} keepalive error: {e}")
                break

    async def _send_text(self, text: str):
        if self._websocket:
            if not self._context_id:
                # First message for a new context - need a space to initialize
                msg = {"text": " ", "context_id": str(uuid.uuid4())}

                # Add voice settings only in first message for a context
                if self._voice_settings:
                    msg["voice_settings"] = self._voice_settings

                await self._websocket.send(json.dumps(msg))
                self._context_id = msg["context_id"]
                logger.trace(f"Created new context {self._context_id}")

                # Now send the actual text content
                msg = {"text": text, "context_id": self._context_id}
                await self._websocket.send(json.dumps(msg))
            else:
                # Continuing with an existing context
                msg = {"text": text, "context_id": self._context_id}
                await self._websocket.send(json.dumps(msg))

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.closed:
                await self._connect()

            try:
                # Close previous context if there was one
                if self._context_id and not self._started:
                    await self._websocket.send(
                        json.dumps({"context_id": self._context_id, "close_context": True})
                    )
                    await self.remove_audio_context(self._context_id)
                    self._context_id = None

                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True
                    self._cumulative_time = 0
                    # Create new context ID and register it
                    self._context_id = str(uuid.uuid4())
                    await self.create_audio_context(self._context_id)

                await self._send_text(text)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                self._started = False
                if self._context_id:
                    await self.remove_audio_context(self._context_id)
                    self._context_id = None
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")


class ElevenLabsHttpTTSService(WordTTSService):
    """ElevenLabs Text-to-Speech service using HTTP streaming with word timestamps.

    Args:
        api_key: ElevenLabs API key
        voice_id: ID of the voice to use
        aiohttp_session: aiohttp ClientSession
        model: Model ID (default: "eleven_flash_v2_5" for low latency)
        base_url: API base URL
        sample_rate: Output sample rate
        params: Additional parameters for voice configuration
    """

    class InputParams(BaseModel):
        language: Optional[Language] = None
        optimize_streaming_latency: Optional[int] = None
        stability: Optional[float] = None
        similarity_boost: Optional[float] = None
        style: Optional[float] = None
        use_speaker_boost: Optional[bool] = None
        speed: Optional[float] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "eleven_flash_v2_5",
        base_url: str = "https://api.elevenlabs.io",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(
            aggregate_sentences=True,
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

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert pipecat Language to ElevenLabs language code."""
        return language_to_elevenlabs_language(language)

    def can_generate_metrics(self) -> bool:
        """Indicate that this service can generate usage metrics."""
        return True

    def _set_voice_settings(self):
        return build_elevenlabs_voice_settings(self._settings)

    def _reset_state(self):
        """Reset internal state variables."""
        self._cumulative_time = 0
        self._started = False
        self._previous_text = ""
        logger.debug(f"{self}: Reset internal state")

    async def start(self, frame: StartFrame):
        """Initialize the service upon receiving a StartFrame."""
        await super().start(frame)
        self._output_format = output_format_from_sample_rate(self.sample_rate)
        self._reset_state()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (StartInterruptionFrame, TTSStoppedFrame)):
            # Reset timing on interruption or stop
            self._reset_state()

            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("Reset", 0)])

        elif isinstance(frame, LLMFullResponseEndFrame):
            # End of turn - reset previous text
            self._previous_text = ""

    def calculate_word_times(self, alignment_info: Mapping[str, Any]) -> List[Tuple[str, float]]:
        """Calculate word timing from character alignment data.

        Example input data:
        {
            "characters": [" ", "H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"],
            "character_start_times_seconds": [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "character_end_times_seconds": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }

        Would produce word times (with cumulative_time=0):
        [("Hello", 0.1), ("world", 0.5)]

        Args:
            alignment_info: Character timing data from ElevenLabs

        Returns:
            List of (word, timestamp) pairs
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
        current_word = ""
        first_char_idx = -1

        for i, char in enumerate(chars):
            if char == " ":
                if current_word:  # Only add non-empty words
                    words.append(current_word)
                    # Use time of the first character of the word, offset by cumulative time
                    word_start_times.append(
                        self._cumulative_time + char_start_times[first_char_idx]
                    )
                    current_word = ""
                    first_char_idx = -1
            else:
                if not current_word:  # This is the first character of a new word
                    first_char_idx = i
                current_word += char

        # Don't forget the last word if there's no trailing space
        if current_word and first_char_idx >= 0:
            words.append(current_word)
            word_start_times.append(self._cumulative_time + char_start_times[first_char_idx])

        # Create word-time pairs
        word_times = list(zip(words, word_start_times))

        return word_times

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using ElevenLabs streaming API with timestamps.

        Makes a request to the ElevenLabs API to generate audio and timing data.
        Tracks the duration of each utterance to ensure correct sequencing.
        Includes previous text as context for better prosody continuity.

        Args:
            text: Text to convert to speech

        Yields:
            Audio and control frames
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
