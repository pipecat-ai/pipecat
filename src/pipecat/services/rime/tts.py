#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime text-to-speech service implementations.

This module provides both WebSocket and HTTP-based text-to-speech services
using Rime's API for streaming and batch audio synthesis.
"""

import base64
import json
import uuid
from typing import Any, AsyncGenerator, Mapping, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import AudioContextWordTTSService, TTSService
from pipecat.transcriptions import language
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Rime, you need to `pip install pipecat-ai[rime]`.")
    raise Exception(f"Missing module: {e}")


def language_to_rime_language(language: Language) -> str:
    """Convert pipecat Language to Rime language code.

    Args:
        language: The pipecat Language enum value.

    Returns:
        Three-letter language code used by Rime (e.g., 'eng' for English).
    """
    LANGUAGE_MAP = {
        Language.DE: "ger",
        Language.FR: "fra",
        Language.EN: "eng",
        Language.ES: "spa",
    }
    return LANGUAGE_MAP.get(language, "eng")


class RimeTTSService(AudioContextWordTTSService):
    """Text-to-Speech service using Rime's websocket API.

    Uses Rime's websocket JSON API to convert text to speech with word-level timing
    information. Supports interruptions and maintains context across multiple messages
    within a turn.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Rime TTS service.

        Parameters:
            language: Language for synthesis. Defaults to English.
            speed_alpha: Speech speed multiplier. Defaults to 1.0.
            reduce_latency: Whether to reduce latency at potential quality cost.
            pause_between_brackets: Whether to add pauses between bracketed content.
            phonemize_between_brackets: Whether to phonemize bracketed content.
        """

        language: Optional[Language] = Language.EN
        speed_alpha: Optional[float] = 1.0
        reduce_latency: Optional[bool] = False
        pause_between_brackets: Optional[bool] = False
        phonemize_between_brackets: Optional[bool] = False

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        url: str = "wss://users.rime.ai/ws2",
        model: str = "mistv2",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        text_aggregator: Optional[BaseTextAggregator] = None,
        aggregate_sentences: Optional[bool] = True,
        **kwargs,
    ):
        """Initialize Rime TTS service.

        Args:
            api_key: Rime API key for authentication.
            voice_id: ID of the voice to use.
            url: Rime websocket API endpoint.
            model: Model ID to use for synthesis.
            sample_rate: Audio sample rate in Hz.
            params: Additional configuration parameters.
            text_aggregator: Custom text aggregator for processing input text.
            aggregate_sentences: Whether to aggregate sentences within the TTSService.
            **kwargs: Additional arguments passed to parent class.
        """
        # Initialize with parent class settings for proper frame handling
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            text_aggregator=text_aggregator or SkipTagsAggregator([("spell(", ")")]),
            **kwargs,
        )

        params = params or RimeTTSService.InputParams()

        # Store service configuration
        self._api_key = api_key
        self._url = url
        self._voice_id = voice_id
        self._model = model
        self._settings = {
            "speaker": voice_id,
            "modelId": model,
            "audioFormat": "pcm",
            "samplingRate": 0,
            "lang": self.language_to_service_language(params.language)
            if params.language
            else "eng",
            "speedAlpha": params.speed_alpha,
            "reduceLatency": params.reduce_latency,
            "pauseBetweenBrackets": json.dumps(params.pause_between_brackets),
            "phonemizeBetweenBrackets": json.dumps(params.phonemize_between_brackets),
        }

        # State tracking
        self._context_id = None  # Tracks current turn
        self._receive_task = None
        self._cumulative_time = 0  # Accumulates time across messages

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Rime service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert pipecat language to Rime language code.

        Args:
            language: The language to convert.

        Returns:
            The Rime-specific language code, or None if not supported.
        """
        return language_to_rime_language(language)

    async def set_model(self, model: str):
        """Update the TTS model.

        Args:
            model: The model name to use for synthesis.
        """
        self._model = model
        await super().set_model(model)

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings and reconnect if voice changed."""
        prev_voice = self._voice_id
        await super()._update_settings(settings)
        if not prev_voice == self._voice_id:
            self._settings["speaker"] = self._voice_id
            logger.info(f"Switching TTS voice to: [{self._voice_id}]")
            await self._disconnect()
            await self._connect()

    def _build_msg(self, text: str = "") -> dict:
        """Build JSON message for Rime API."""
        return {"text": text, "contextId": self._context_id}

    def _build_clear_msg(self) -> dict:
        """Build clear operation message."""
        return {"operation": "clear"}

    def _build_eos_msg(self) -> dict:
        """Build end-of-stream operation message."""
        return {"operation": "eos"}

    async def start(self, frame: StartFrame):
        """Start the service and establish websocket connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["samplingRate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close connection.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel current operation and clean up.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Establish websocket connection and start receive task."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close websocket connection and clean up tasks."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to Rime websocket API with configured settings."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            params = "&".join(f"{k}={v}" for k, v in self._settings.items())
            url = f"{self._url}?{params}"
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(url, additional_headers=headers)
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close websocket connection and reset state."""
        try:
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.send(json.dumps(self._build_eos_msg()))
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._context_id = None
            self._websocket = None

    def _get_websocket(self):
        """Get active websocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        """Handle interruption by clearing current context."""
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        if self._context_id:
            await self._get_websocket().send(json.dumps(self._build_clear_msg()))
            self._context_id = None

    def _calculate_word_times(self, words: list, starts: list, ends: list) -> list:
        """Calculate word timing pairs with proper spacing and punctuation.

        Args:
            words: List of words from Rime.
            starts: List of start times for each word.
            ends: List of end times for each word.

        Returns:
            List of (word, timestamp) pairs with proper timing.
        """
        word_pairs = []
        for i, (word, start_time, _) in enumerate(zip(words, starts, ends)):
            if not word.strip():
                continue

            # Adjust timing by adding cumulative time
            adjusted_start = start_time + self._cumulative_time

            # Handle punctuation by appending to previous word
            is_punctuation = bool(word.strip(",.!?") == "")
            if is_punctuation and word_pairs:
                prev_word, prev_time = word_pairs[-1]
                word_pairs[-1] = (prev_word + word, prev_time)
            else:
                word_pairs.append((word, adjusted_start))

        return word_pairs

    async def flush_audio(self):
        """Flush any pending audio synthesis."""
        if not self._context_id or not self._websocket:
            return

        logger.trace(f"{self}: flushing audio")
        await self._get_websocket().send(json.dumps({"text": " "}))
        self._context_id = None

    async def _receive_messages(self):
        """Process incoming websocket messages."""
        async for message in self._get_websocket():
            msg = json.loads(message)

            if not msg or not self.audio_context_available(msg["contextId"]):
                continue

            if msg["type"] == "chunk":
                # Process audio chunk
                await self.stop_ttfb_metrics()
                self.start_word_timestamps()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["data"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                await self.append_to_audio_context(msg["contextId"], frame)

            elif msg["type"] == "timestamps":
                # Process word timing information
                timestamps = msg.get("word_timestamps", {})
                words = timestamps.get("words", [])
                starts = timestamps.get("start", [])
                ends = timestamps.get("end", [])

                if words and starts:
                    # Calculate word timing pairs
                    word_pairs = self._calculate_word_times(words, starts, ends)
                    if word_pairs:
                        await self.add_word_timestamps(word_pairs)
                        self._cumulative_time = ends[-1] + self._cumulative_time
                        logger.debug(f"Updated cumulative time to: {self._cumulative_time}")

            elif msg["type"] == "error":
                logger.error(f"{self} error: {msg}")
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(ErrorFrame(f"{self} error: {msg['message']}"))
                self._context_id = None

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push frame and handle end-of-turn conditions.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("Reset", 0)])

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Rime's streaming API.

        Args:
            text: The text to convert to speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self._context_id:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._cumulative_time = 0
                    self._context_id = str(uuid.uuid4())
                    await self.create_audio_context(self._context_id)

                msg = self._build_msg(text=text)
                await self._get_websocket().send(json.dumps(msg))
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


class RimeHttpTTSService(TTSService):
    """Rime HTTP-based text-to-speech service.

    Provides text-to-speech synthesis using Rime's HTTP API for batch processing.
    Suitable for use cases where streaming is not required.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Rime HTTP TTS service.

        Parameters:
            language: Language for synthesis. Defaults to English.
            pause_between_brackets: Whether to add pauses between bracketed content.
            phonemize_between_brackets: Whether to phonemize bracketed content.
            inline_speed_alpha: Inline speed control markup.
            speed_alpha: Speech speed multiplier. Defaults to 1.0.
            reduce_latency: Whether to reduce latency at potential quality cost.
        """

        language: Optional[Language] = Language.EN
        pause_between_brackets: Optional[bool] = False
        phonemize_between_brackets: Optional[bool] = False
        inline_speed_alpha: Optional[str] = None
        speed_alpha: Optional[float] = 1.0
        reduce_latency: Optional[bool] = False

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "mistv2",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize Rime HTTP TTS service.

        Args:
            api_key: Rime API key for authentication.
            voice_id: ID of the voice to use.
            aiohttp_session: Shared aiohttp session for HTTP requests.
            model: Model ID to use for synthesis.
            sample_rate: Audio sample rate in Hz.
            params: Additional configuration parameters.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or RimeHttpTTSService.InputParams()

        self._api_key = api_key
        self._session = aiohttp_session
        self._base_url = "https://users.rime.ai/v1/rime-tts"
        self._settings = {
            "lang": self.language_to_service_language(params.language)
            if params.language
            else "eng",
            "speedAlpha": params.speed_alpha,
            "reduceLatency": params.reduce_latency,
            "pauseBetweenBrackets": params.pause_between_brackets,
            "phonemizeBetweenBrackets": params.phonemize_between_brackets,
        }
        self.set_voice(voice_id)
        self.set_model_name(model)

        if params.inline_speed_alpha:
            self._settings["inlineSpeedAlpha"] = params.inline_speed_alpha

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Rime HTTP service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert pipecat language to Rime language code.

        Args:
            language: The language to convert.

        Returns:
            The Rime-specific language code, or None if not supported.
        """
        return language_to_rime_language(language)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Rime's HTTP API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        headers = {
            "Accept": "audio/pcm",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = self._settings.copy()
        payload["text"] = text
        payload["speaker"] = self._voice_id
        payload["modelId"] = self._model_name
        payload["samplingRate"] = self.sample_rate

        # Arcana does not support PCM audio
        if payload["modelId"] == "arcana":
            headers["Accept"] = "audio/wav"
            need_to_strip_wav_header = True
        else:
            need_to_strip_wav_header = False

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_message = f"Rime TTS error: HTTP {response.status}"
                    logger.error(error_message)
                    yield ErrorFrame(error=error_message)
                    return

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()

                CHUNK_SIZE = self.chunk_size

                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    if need_to_strip_wav_header and chunk.startswith(b"RIFF"):
                        chunk = chunk[44:]
                        need_to_strip_wav_header = False

                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                        yield frame
        except Exception as e:
            logger.exception(f"Error generating TTS: {e}")
            yield ErrorFrame(error=f"Rime TTS error: {str(e)}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
