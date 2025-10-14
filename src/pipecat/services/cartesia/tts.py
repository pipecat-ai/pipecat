#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cartesia text-to-speech service implementations."""

import base64
import json
import uuid
import warnings
from typing import AsyncGenerator, List, Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

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
from pipecat.services.tts_service import AudioContextWordTTSService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts

# Suppress regex warnings from pydub (used by cartesia)
warnings.filterwarnings("ignore", message="invalid escape sequence", category=SyntaxWarning)


# See .env.example for Cartesia configuration needed
try:
    from cartesia import AsyncCartesia
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Cartesia, you need to `pip install pipecat-ai[cartesia]`.")
    raise Exception(f"Missing module: {e}")


def language_to_cartesia_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Cartesia language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Cartesia language code, or None if not supported.
    """
    BASE_LANGUAGES = {
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.NL: "nl",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RU: "ru",
        Language.SV: "sv",
        Language.TR: "tr",
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


class CartesiaTTSService(AudioContextWordTTSService):
    """Cartesia TTS service with WebSocket streaming and word timestamps.

    Provides text-to-speech using Cartesia's streaming WebSocket API.
    Supports word-level timestamps, audio context management, and various voice
    customization options including speed and emotion controls.
    """

    class InputParams(BaseModel):
        """Input parameters for Cartesia TTS configuration.

        Parameters:
            language: Language to use for synthesis.
            speed: Voice speed control.
            emotion: List of emotion controls.

                .. deprecated:: 0.0.68
                        The `emotion` parameter is deprecated and will be removed in a future version.
        """

        language: Optional[Language] = Language.EN
        speed: Optional[Literal["slow", "normal", "fast"]] = None
        emotion: Optional[List[str]] = []

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        cartesia_version: str = "2025-04-16",
        url: str = "wss://api.cartesia.ai/tts/websocket",
        model: str = "sonic-2",
        sample_rate: Optional[int] = None,
        encoding: str = "pcm_s16le",
        container: str = "raw",
        params: Optional[InputParams] = None,
        text_aggregator: Optional[BaseTextAggregator] = None,
        aggregate_sentences: Optional[bool] = True,
        **kwargs,
    ):
        """Initialize the Cartesia TTS service.

        Args:
            api_key: Cartesia API key for authentication.
            voice_id: ID of the voice to use for synthesis.
            cartesia_version: API version string for Cartesia service.
            url: WebSocket URL for Cartesia TTS API.
            model: TTS model to use (e.g., "sonic-2").
            sample_rate: Audio sample rate. If None, uses default.
            encoding: Audio encoding format.
            container: Audio container format.
            params: Additional input parameters for voice customization.
            text_aggregator: Custom text aggregator for processing input text.
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
        # if we're interrupted. Cartesia gives us word-by-word timestamps. We
        # can use those to generate text frames ourselves aligned with the
        # playout timing of the audio!
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            text_aggregator=text_aggregator or SkipTagsAggregator([("<spell>", "</spell>")]),
            **kwargs,
        )

        params = params or CartesiaTTSService.InputParams()

        self._api_key = api_key
        self._cartesia_version = cartesia_version
        self._url = url
        self._settings = {
            "output_format": {
                "container": container,
                "encoding": encoding,
                "sample_rate": 0,
            },
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en",
            "speed": params.speed,
            "emotion": params.emotion,
        }
        self.set_model_name(model)
        self.set_voice(voice_id)

        self._context_id = None
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Cartesia service supports metrics generation.
        """
        return True

    async def set_model(self, model: str):
        """Set the TTS model.

        Args:
            model: The model name to use for synthesis.
        """
        self._model_id = model
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Cartesia language format.

        Args:
            language: The language to convert.

        Returns:
            The Cartesia-specific language code, or None if not supported.
        """
        return language_to_cartesia_language(language)

    def _is_cjk_language(self, language: str) -> bool:
        """Check if the given language is CJK (Chinese, Japanese, Korean).

        Args:
            language: The language code to check.

        Returns:
            True if the language is Chinese, Japanese, or Korean.
        """
        cjk_languages = {"zh", "ja", "ko"}
        base_lang = language.split("-")[0].lower()
        return base_lang in cjk_languages

    def _process_word_timestamps_for_language(
        self, words: List[str], starts: List[float]
    ) -> List[tuple[str, float]]:
        """Process word timestamps based on the current language.

        For CJK languages, Cartesia groups related characters in the same timestamp message.
        For example, in Japanese a single message might be `['こ', 'ん', 'に', 'ち', 'は', '。']`.
        We combine these into single words so the downstream aggregator can add natural
        spacing between meaningful units rather than individual characters.

        For non-CJK languages, words are already properly separated and are used as-is.

        Args:
            words: List of words/characters from Cartesia.
            starts: List of start timestamps for each word/character.

        Returns:
            List of (word, start_time) tuples processed for the language.
        """
        current_language = self._settings.get("language", "en")

        # Check if this is a CJK language
        if self._is_cjk_language(current_language):
            # For CJK languages, combine all characters in this message into one word
            # using the first character's start time
            if words and starts:
                combined_word = "".join(words)
                first_start = starts[0]
                return [(combined_word, first_start)]
            else:
                return []
        else:
            # For non-CJK languages, use as-is
            return list(zip(words, starts))

    def _build_msg(
        self, text: str = "", continue_transcript: bool = True, add_timestamps: bool = True
    ):
        voice_config = {}
        voice_config["mode"] = "id"
        voice_config["id"] = self._voice_id

        if self._settings["emotion"]:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The 'emotion' parameter in __experimental_controls is deprecated and will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            voice_config["__experimental_controls"] = {}
            if self._settings["emotion"]:
                voice_config["__experimental_controls"]["emotion"] = self._settings["emotion"]

        msg = {
            "transcript": text,
            "continue": continue_transcript,
            "context_id": self._context_id,
            "model_id": self.model_name,
            "voice": voice_config,
            "output_format": self._settings["output_format"],
            "language": self._settings["language"],
            "add_timestamps": add_timestamps,
            "use_original_timestamps": False if self.model_name == "sonic" else True,
        }

        if self._settings["speed"]:
            msg["speed"] = self._settings["speed"]

        return json.dumps(msg)

    async def start(self, frame: StartFrame):
        """Start the Cartesia TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["output_format"]["sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Cartesia TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Stop the Cartesia TTS service.

        Args:
            frame: The end frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Cartesia TTS")
            self._websocket = await websocket_connect(
                f"{self._url}?api_key={self._api_key}&cartesia_version={self._cartesia_version}"
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
                logger.debug("Disconnecting from Cartesia")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._context_id = None
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        if self._context_id:
            cancel_msg = json.dumps({"context_id": self._context_id, "cancel": True})
            await self._get_websocket().send(cancel_msg)
            self._context_id = None

    async def flush_audio(self):
        """Flush any pending audio and finalize the current context."""
        if not self._context_id or not self._websocket:
            return
        logger.trace(f"{self}: flushing audio")
        msg = self._build_msg(text="", continue_transcript=False)
        await self._websocket.send(msg)
        self._context_id = None

    async def _process_messages(self):
        async for message in self._get_websocket():
            msg = json.loads(message)
            if not msg or not self.audio_context_available(msg["context_id"]):
                continue
            if msg["type"] == "done":
                await self.stop_ttfb_metrics()
                await self.add_word_timestamps([("TTSStoppedFrame", 0), ("Reset", 0)])
                await self.remove_audio_context(msg["context_id"])
            elif msg["type"] == "timestamps":
                # Process the timestamps based on language before adding them
                processed_timestamps = self._process_word_timestamps_for_language(
                    msg["word_timestamps"]["words"], msg["word_timestamps"]["start"]
                )
                await self.add_word_timestamps(processed_timestamps)
            elif msg["type"] == "chunk":
                await self.stop_ttfb_metrics()
                self.start_word_timestamps()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["data"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                await self.append_to_audio_context(msg["context_id"], frame)
            elif msg["type"] == "error":
                logger.error(f"{self} error: {msg}")
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(ErrorFrame(f"{self} error: {msg['error']}"))
                self._context_id = None
            else:
                logger.error(f"{self} error, unknown message type: {msg}")

    async def _receive_messages(self):
        while True:
            await self._process_messages()
            # Cartesia times out after 5 minutes of innactivity (no keepalive
            # mechanism is available). So, we try to reconnect.
            logger.debug(f"{self} Cartesia connection was disconnected (timeout?), reconnecting")
            await self._connect_websocket()

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Cartesia's streaming API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if not self._context_id:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._context_id = str(uuid.uuid4())
                await self.create_audio_context(self._context_id)

            msg = self._build_msg(text=text)

            try:
                await self._get_websocket().send(msg)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")


class CartesiaHttpTTSService(TTSService):
    """Cartesia HTTP-based TTS service.

    Provides text-to-speech using Cartesia's HTTP API for simpler, non-streaming
    synthesis. Suitable for use cases where streaming is not required and simpler
    integration is preferred.
    """

    class InputParams(BaseModel):
        """Input parameters for Cartesia HTTP TTS configuration.

        Parameters:
            language: Language to use for synthesis.
            speed: Voice speed control.
            emotion: List of emotion controls.

                .. deprecated:: 0.0.68
                        The `emotion` parameter is deprecated and will be removed in a future version.
        """

        language: Optional[Language] = Language.EN
        speed: Optional[Literal["slow", "normal", "fast"]] = None
        emotion: Optional[List[str]] = Field(default_factory=list)

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "sonic-2",
        base_url: str = "https://api.cartesia.ai",
        cartesia_version: str = "2024-11-13",
        sample_rate: Optional[int] = None,
        encoding: str = "pcm_s16le",
        container: str = "raw",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Cartesia HTTP TTS service.

        Args:
            api_key: Cartesia API key for authentication.
            voice_id: ID of the voice to use for synthesis.
            model: TTS model to use (e.g., "sonic-2").
            base_url: Base URL for Cartesia HTTP API.
            cartesia_version: API version string for Cartesia service.
            sample_rate: Audio sample rate. If None, uses default.
            encoding: Audio encoding format.
            container: Audio container format.
            params: Additional input parameters for voice customization.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or CartesiaHttpTTSService.InputParams()

        self._api_key = api_key
        self._base_url = base_url
        self._cartesia_version = cartesia_version
        self._settings = {
            "output_format": {
                "container": container,
                "encoding": encoding,
                "sample_rate": 0,
            },
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en",
            "speed": params.speed,
            "emotion": params.emotion,
        }
        self.set_voice(voice_id)
        self.set_model_name(model)

        self._client = AsyncCartesia(
            api_key=api_key,
            base_url=base_url,
        )

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Cartesia HTTP service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Cartesia language format.

        Args:
            language: The language to convert.

        Returns:
            The Cartesia-specific language code, or None if not supported.
        """
        return language_to_cartesia_language(language)

    async def start(self, frame: StartFrame):
        """Start the Cartesia HTTP TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["output_format"]["sample_rate"] = self.sample_rate

    async def stop(self, frame: EndFrame):
        """Stop the Cartesia HTTP TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._client.close()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Cartesia HTTP TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._client.close()

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Cartesia's HTTP API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            voice_config = {"mode": "id", "id": self._voice_id}

            if self._settings["emotion"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(
                        "The 'emotion' parameter in voice.__experimental_controls is deprecated and will be removed in a future version.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                voice_config["__experimental_controls"] = {"emotion": self._settings["emotion"]}

            await self.start_ttfb_metrics()

            payload = {
                "model_id": self._model_name,
                "transcript": text,
                "voice": voice_config,
                "output_format": self._settings["output_format"],
                "language": self._settings["language"],
            }

            if self._settings["speed"]:
                payload["speed"] = self._settings["speed"]

            yield TTSStartedFrame()

            session = await self._client._get_session()

            headers = {
                "Cartesia-Version": self._cartesia_version,
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
            }

            url = f"{self._base_url}/tts/bytes"

            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Cartesia API error: {error_text}")
                    await self.push_error(ErrorFrame(f"Cartesia API error: {error_text}"))
                    raise Exception(f"Cartesia API returned status {response.status}: {error_text}")

                audio_data = await response.read()

            await self.start_tts_usage_metrics(text)

            frame = TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
            )

            yield frame

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(f"Error generating TTS: {e}"))
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
