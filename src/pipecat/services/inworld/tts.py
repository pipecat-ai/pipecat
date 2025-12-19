#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inworld AI Text-to-Speech Service Implementation.

Contains two TTS services:
- InworldHttpTTSService: HTTP-based TTS service.
- InworldTTSService: WebSocket-based TTS service.
"""

import base64
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
from loguru import logger
from pydantic import BaseModel

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Inworld WebSocket TTS, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")

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
from pipecat.services.tts_service import AudioContextWordTTSService, WordTTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class InworldHttpTTSService(WordTTSService):
    """Inworld AI HTTP-based TTS service.

    Supports both streaming and non-streaming modes via the `streaming` parameter.
    Outputs LINEAR16 audio at configurable sample rates with word/character timestamps.
    """

    class InputParams(BaseModel):
        """Input parameters for Inworld TTS configuration.

        Parameters:
            temperature: Temperature for speech synthesis.
            speaking_rate: Speaking rate for speech synthesis.
        """

        temperature: Optional[float] = None
        speaking_rate: Optional[float] = None

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str = "Ashley",
        model: str = "inworld-tts-1",
        streaming: bool = True,
        sample_rate: Optional[int] = None,
        encoding: str = "LINEAR16",
        params: InputParams = None,
        **kwargs,
    ):
        """Initialize the Inworld TTS service.

        Args:
            api_key: Inworld API key.
            aiohttp_session: aiohttp ClientSession for HTTP requests.
            voice_id: ID of the voice to use for synthesis.
            model: ID of the model to use for synthesis.
            streaming: Whether to use streaming mode.
            sample_rate: Audio sample rate in Hz.
            encoding: Audio encoding format.
            params: Input parameters for Inworld TTS configuration.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(
            push_text_frames=False,
            push_stop_frames=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or InworldHttpTTSService.InputParams()

        self._api_key = api_key
        self._session = aiohttp_session
        self._streaming = streaming
        self._timestamp_type = "WORD"

        if streaming:
            self._base_url = "https://api.inworld.ai/tts/v1/voice:stream"
        else:
            self._base_url = "https://api.inworld.ai/tts/v1/voice"

        self._settings = {
            "voiceId": voice_id,
            "modelId": model,
            "audioConfig": {
                "audioEncoding": encoding,
                "sampleRateHertz": 0,
            },
        }

        if params.temperature is not None:
            self._settings["temperature"] = params.temperature
        if params.speaking_rate is not None:
            self._settings["audioConfig"]["speakingRate"] = params.speaking_rate

        self._started = False
        self._cumulative_time = 0.0

        self.set_voice(voice_id)
        self.set_model_name(model)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Inworld TTS service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Inworld TTS service.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        self._settings["audioConfig"]["sampleRateHertz"] = self.sample_rate

    async def stop(self, frame: EndFrame):
        """Stop the Inworld TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the Inworld TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, TTSStoppedFrame)):
            self._started = False
            self._cumulative_time = 0.0
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("Reset", 0)])

    def _calculate_word_times(
        self,
        timestamp_info: Dict[str, Any],
    ) -> Tuple[List[Tuple[str, float]], float]:
        """Calculate word timestamps from Inworld HTTP API word-level response.

        Note: Inworld HTTP provides timestamps that reset for each request.
        We track cumulative time across requests to maintain continuity.

        Args:
            timestamp_info: The timestamp information from Inworld API.

        Returns:
            Tuple of (word_times, chunk_end_time) where chunk_end_time is the
            end time of the last word in this chunk (not cumulative).
        """
        word_times: List[Tuple[str, float]] = []
        chunk_end_time = 0.0

        alignment = timestamp_info.get("wordAlignment", {})
        words = alignment.get("words", [])
        start_times = alignment.get("wordStartTimeSeconds", [])
        end_times = alignment.get("wordEndTimeSeconds", [])

        if words and start_times and len(words) == len(start_times):
            for i, word in enumerate(words):
                word_start = self._cumulative_time + start_times[i]
                word_times.append((word, word_start))

            # Track the end time of the last word in this chunk
            if end_times and len(end_times) > 0:
                chunk_end_time = end_times[-1]

        return (word_times, chunk_end_time)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio for the given text.

        Args:
            text: The text to generate TTS audio for.

        Returns:
            An asynchronous generator of frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}] (streaming={self._streaming})")

        payload = {
            "text": text,
            "voiceId": self._settings["voiceId"],
            "modelId": self._settings["modelId"],
            "audioConfig": self._settings["audioConfig"],
        }

        if "temperature" in self._settings:
            payload["temperature"] = self._settings["temperature"]

        # Use WORD timestamps for simplicity and correct spacing/capitalization
        payload["timestampType"] = self._timestamp_type

        headers = {
            "Authorization": f"Basic {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            await self.start_ttfb_metrics()

            if not self._started:
                await self.start_word_timestamps()
                yield TTSStartedFrame()
                self._started = True

            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Inworld API error: {error_text}")
                    yield ErrorFrame(error=f"Inworld API error: {error_text}")
                    return

                if self._streaming:
                    async for frame in self._process_streaming_response(response):
                        yield frame
                else:
                    async for frame in self._process_non_streaming_response(response):
                        yield frame

            await self.start_tts_usage_metrics(text)

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

        finally:
            await self.stop_all_metrics()

    async def _process_streaming_response(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[Frame, None]:
        """Process a streaming response from the Inworld API.

        Args:
            response: The response from the Inworld API.

        Returns:
            An asynchronous generator of frames.
        """
        buffer = ""
        # Track the duration of this utterance based on the last word's end time
        utterance_duration = 0.0

        async for chunk in response.content.iter_chunked(1024):
            buffer += chunk.decode("utf-8")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line_str = line.strip()

                if not line_str:
                    continue

                try:
                    chunk_data = json.loads(line_str)

                    if "result" in chunk_data and "audioContent" in chunk_data["result"]:
                        await self.stop_ttfb_metrics()
                        async for frame in self._process_audio_chunk(
                            base64.b64decode(chunk_data["result"]["audioContent"])
                        ):
                            yield frame

                    if "result" in chunk_data and "timestampInfo" in chunk_data["result"]:
                        timestamp_info = chunk_data["result"]["timestampInfo"]
                        word_times, chunk_end_time = self._calculate_word_times(timestamp_info)
                        if word_times:
                            await self.add_word_timestamps(word_times)
                        # Track the maximum end time across all chunks
                        utterance_duration = max(utterance_duration, chunk_end_time)

                except json.JSONDecodeError:
                    continue

        # After processing all chunks, add the total utterance duration
        # to the cumulative time to ensure next utterance starts after this one
        if utterance_duration > 0:
            self._cumulative_time += utterance_duration

    async def _process_non_streaming_response(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[Frame, None]:
        """Process a non-streaming response from the Inworld API.

        Args:
            response: The response from the Inworld API.

        Returns:
            An asynchronous generator of frames.
        """
        response_data = await response.json()

        if "audioContent" not in response_data:
            logger.error("No audioContent in Inworld API response")
            yield ErrorFrame(error="No audioContent in response")
            return

        utterance_duration = 0.0
        if "timestampInfo" in response_data:
            timestamp_info = response_data["timestampInfo"]
            word_times, chunk_end_time = self._calculate_word_times(timestamp_info)
            if word_times:
                await self.add_word_timestamps(word_times)
            utterance_duration = chunk_end_time

        audio_data = base64.b64decode(response_data["audioContent"])

        if len(audio_data) > 44 and audio_data.startswith(b"RIFF"):
            audio_data = audio_data[44:]

        chunk_size = self.chunk_size
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            if chunk:
                await self.stop_ttfb_metrics()
                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )

        # After processing all audio, add the utterance duration to cumulative time
        if utterance_duration > 0:
            self._cumulative_time += utterance_duration

    async def _process_audio_chunk(self, audio_chunk: bytes) -> AsyncGenerator[Frame, None]:
        """Process an audio chunk from the Inworld API.

        Args:
            audio_chunk: The audio chunk to process.

        Returns:
            An asynchronous generator of frames.
        """
        if not audio_chunk:
            return

        audio_data = audio_chunk

        if len(audio_chunk) > 44 and audio_chunk.startswith(b"RIFF"):
            audio_data = audio_chunk[44:]

        if audio_data:
            yield TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
            )


class InworldTTSService(AudioContextWordTTSService):
    """Inworld AI WebSocket-based TTS service.

    Uses bidirectional WebSocket for lower latency streaming. Supports multiple
    independent audio contexts per connection (max 5). Outputs LINEAR16 audio
    with word/character timestamps.
    """

    class InputParams(BaseModel):
        """Input parameters for Inworld WebSocket TTS configuration.

        Parameters:
            temperature: Temperature for speech synthesis.
            speaking_rate: Speaking rate for speech synthesis.
            apply_text_normalization: Whether to apply text normalization.
            max_buffer_delay_ms: Maximum buffer delay in milliseconds.
            buffer_char_threshold: Buffer character threshold.
        """

        temperature: Optional[float] = None
        speaking_rate: Optional[float] = None
        apply_text_normalization: Optional[str] = None
        max_buffer_delay_ms: Optional[int] = None
        buffer_char_threshold: Optional[int] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "Ashley",
        model: str = "inworld-tts-1",
        url: str = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional",
        sample_rate: Optional[int] = None,
        encoding: str = "LINEAR16",
        params: InputParams = None,
        **kwargs: Any,
    ):
        """Initialize the Inworld WebSocket TTS service.

        Args:
            api_key: Inworld API key.
            voice_id: ID of the voice to use for synthesis.
            model: ID of the model to use for synthesis.
            url: URL of the Inworld WebSocket API.
            sample_rate: Audio sample rate in Hz.
            encoding: Audio encoding format.
            params: Input parameters for Inworld WebSocket TTS configuration.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(
            push_text_frames=False,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or InworldTTSService.InputParams()

        self._api_key = api_key
        self._url = url
        self._settings: Dict[str, Any] = {
            "voiceId": voice_id,
            "modelId": model,
            "audioConfig": {
                "audioEncoding": encoding,
                "sampleRateHertz": 0,
            },
        }
        self._timestamp_type = "WORD"

        if params.temperature is not None:
            self._settings["temperature"] = params.temperature
        if params.speaking_rate is not None:
            self._settings["audioConfig"]["speakingRate"] = params.speaking_rate
        if params.apply_text_normalization is not None:
            self._settings["applyTextNormalization"] = params.apply_text_normalization

        self._buffer_settings = {
            "maxBufferDelayMs": params.max_buffer_delay_ms,
            "bufferCharThreshold": params.buffer_char_threshold,
        }

        self._receive_task = None
        self._context_id = None
        self._started = False

        self.set_voice(voice_id)
        self.set_model_name(model)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Inworld WebSocket TTS service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Inworld WebSocket TTS service.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        self._settings["audioConfig"]["sampleRateHertz"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Inworld WebSocket TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Inworld WebSocket TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        """Flush any pending audio from the Inworld WebSocket TTS service.

        Args:
            frame: The flush frame.
        """
        if self._context_id:
            ctx_to_close = self._context_id
            self._context_id = None
            await self._send_close_context(ctx_to_close)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, TTSStoppedFrame):
            self._started = False
            await self.add_word_timestamps([("Reset", 0)])

    def _calculate_word_times(self, timestamp_info: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Calculate word timestamps from Inworld WebSocket API response.

        Note: Inworld WebSocket provides cumulative timestamps across all chunks
        within a conversation turn, similar to Cartesia. No additional tracking needed.

        Args:
            timestamp_info: The timestamp information from Inworld API.

        Returns:
            A list of (word, timestamp) tuples.
        """
        word_times: List[Tuple[str, float]] = []

        alignment = timestamp_info.get("wordAlignment", {})
        words = alignment.get("words", [])
        start_times = alignment.get("wordStartTimeSeconds", [])

        if words and start_times and len(words) == len(start_times):
            for i, word in enumerate(words):
                word_times.append((word, start_times[i]))

        return word_times

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle an interruption from the Inworld WebSocket TTS service.

        Args:
            frame: The interruption frame.
            direction: The direction of the interruption.
        """
        await super()._handle_interruption(frame, direction)

        if self._context_id and self._websocket:
            logger.trace(f"Closing context {self._context_id} due to interruption")
            try:
                await self._send_close_context(self._context_id)
            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._context_id = None
            self._started = False

    def _get_websocket(self):
        """Get the websocket for the Inworld WebSocket TTS service.

        Returns:
            The websocket.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _connect(self):
        """Connect to the Inworld WebSocket TTS service.

        Returns:
            The websocket.
        """
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from the Inworld WebSocket TTS service.

        Returns:
            The websocket.
        """
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to the Inworld WebSocket TTS service.

        Returns:
            The websocket.
        """
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Inworld WebSocket TTS")
            headers = [("Authorization", f"Basic {self._api_key}")]
            self._websocket = await websocket_connect(self._url, additional_headers=headers)
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Disconnect from the Inworld WebSocket TTS service.

        Returns:
            The websocket.
        """
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Inworld WebSocket TTS")
                if self._context_id:
                    try:
                        await self._send_close_context(self._context_id)
                    except Exception:
                        pass
                await self._websocket.close()
                logger.debug("Disconnected from Inworld WebSocket TTS")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._started = False
            self._context_id = None
            self._cumulative_time = 0.0
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _process_messages(self):
        """Process incoming WebSocket messages from Inworld.

        Returns:
            The messages.
        """
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON message")
                continue

            result = msg.get("result", {})
            ctx_id = result.get("contextId") or result.get("context_id")

            # Check for errors
            status = result.get("status", {})
            if status.get("code", 0) != 0:
                error_msg = status.get("message", "Unknown error")
                await self.push_error(error_msg=f"Inworld API error: {error_msg}")
                continue

            if "error" in msg:
                await self.push_error(error_msg=str(msg["error"]))
                continue

            # Skip messages for unavailable contexts
            if ctx_id and not self.audio_context_available(ctx_id):
                continue

            # Process audio chunk
            audio_chunk = result.get("audioChunk", {})
            audio_b64 = audio_chunk.get("audioContent")

            if audio_b64:
                await self.stop_ttfb_metrics()
                await self.start_word_timestamps()
                audio = base64.b64decode(audio_b64)
                if len(audio) > 44 and audio.startswith(b"RIFF"):
                    audio = audio[44:]
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1)

                if ctx_id:
                    if not self.audio_context_available(ctx_id):
                        await self.create_audio_context(ctx_id)
                    await self.append_to_audio_context(ctx_id, frame)

                # timestampInfo is inside audioChunk
                timestamp_info = audio_chunk.get("timestampInfo")
                if timestamp_info:
                    word_times = self._calculate_word_times(timestamp_info)
                    if word_times:
                        await self.add_word_timestamps(word_times)

            # Handle context completion
            if "flushCompleted" in result or "contextClosed" in result:
                await self.stop_ttfb_metrics()
                await self.add_word_timestamps([("TTSStoppedFrame", 0), ("Reset", 0)])
                if ctx_id and self.audio_context_available(ctx_id):
                    await self.remove_audio_context(ctx_id)

    async def _receive_messages(self):
        """Receive messages from the Inworld WebSocket TTS service with auto-reconnect.

        Returns:
            The messages.
        """
        while True:
            await self._process_messages()
            # Inworld may disconnect after period of inactivity, so we try to reconnect
            logger.debug(f"{self} Inworld connection was disconnected, reconnecting")
            await self._connect_websocket()

    async def _send_context(self, context_id: str):
        """Send a context to the Inworld WebSocket TTS service.

        Args:
            context_id: The context ID.
        """
        create_config: Dict[str, Any] = {
            "voiceId": self._settings["voiceId"],
            "modelId": self._settings["modelId"],
            "audioConfig": self._settings["audioConfig"],
        }

        if "temperature" in self._settings:
            create_config["temperature"] = self._settings["temperature"]
        if "applyTextNormalization" in self._settings:
            create_config["applyTextNormalization"] = self._settings["applyTextNormalization"]
        if self._buffer_settings["maxBufferDelayMs"] is not None:
            create_config["maxBufferDelayMs"] = self._buffer_settings["maxBufferDelayMs"]
        if self._buffer_settings["bufferCharThreshold"] is not None:
            create_config["bufferCharThreshold"] = self._buffer_settings["bufferCharThreshold"]

        create_config["timestampType"] = self._timestamp_type

        msg = {"create": create_config, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_text(self, context_id: str, text: str):
        """Send text to the Inworld WebSocket TTS service.

        Args:
            context_id: The context ID.
            text: The text to send.
        """
        msg = {"send_text": {"text": text}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_flush(self, context_id: str):
        """Send a flush to the Inworld WebSocket TTS service.

        Args:
            context_id: The context ID.
        """
        msg = {"flush_context": {}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    async def _send_close_context(self, context_id: str):
        """Send a close context to the Inworld WebSocket TTS service.

        Args:
            context_id: The context ID.
        """
        msg = {"close_context": {}, "contextId": context_id}
        await self.send_with_retry(json.dumps(msg), self._report_error)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio for the given text using the Inworld WebSocket TTS service.

        Args:
            text: The text to generate TTS audio for.

        Returns:
            An asynchronous generator of frames.
        """
        logger.debug(f"{self}: Generating WebSocket TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True
                    if not self._context_id:
                        self._context_id = str(uuid.uuid4())
                    if not self.audio_context_available(self._context_id):
                        await self.create_audio_context(self._context_id)

                    await self._send_context(self._context_id)

                await self._send_text(self._context_id, text)
                await self.start_tts_usage_metrics(text)

            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame()
                self._started = False
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
