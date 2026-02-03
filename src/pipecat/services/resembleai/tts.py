#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Resemble AI text-to-speech service implementations."""

import base64
import json
from typing import AsyncGenerator, Optional

from loguru import logger

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
from pipecat.services.tts_service import AudioContextWordTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Resemble AI, you need to `pip install pipecat-ai[resembleai]`.")
    raise Exception(f"Missing module: {e}")


class ResembleAITTSService(AudioContextWordTTSService):
    """Resemble AI TTS service with WebSocket streaming and word timestamps.

    Provides text-to-speech using Resemble AI's streaming WebSocket API.
    Supports word-level timestamps and audio context management for handling
    multiple simultaneous synthesis requests with proper interruption support.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        url: str = "wss://websocket.cluster.resemble.ai/stream",
        precision: Optional[str] = "PCM_16",
        output_format: Optional[str] = "wav",
        sample_rate: Optional[int] = 22050,
        **kwargs,
    ):
        """Initialize the Resemble AI TTS service.

        Args:
            api_key: Resemble AI API key for authentication.
            voice_id: Voice UUID to use for synthesis.
            url: WebSocket URL for Resemble AI TTS API.
            precision: PCM bit depth (PCM_32, PCM_24, PCM_16, or MULAW).
            output_format: Audio format (wav or mp3).
            sample_rate: Audio sample rate (8000, 16000, 22050, 32000, or 44100). Defaults to 22050.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )

        self._api_key = api_key
        self._voice_id = voice_id
        self._url = url
        self._settings = {
            "precision": precision,
            "output_format": output_format,
            "sample_rate": sample_rate,
        }

        self._websocket = None
        self._request_id_counter = 0
        self._receive_task = None

        # Map request_id to context_id for tracking TTS requests
        self._request_id_to_context: dict[int, str] = {}

        # Per-request audio buffers to handle concurrent TTS requests
        # ResembleAI may send odd-length data even for PCM_16, so buffering helps us
        # create properly aligned frames while maintaining smooth audio output
        self._audio_buffers: dict[str, bytearray] = {}
        self._buffer_threshold_bytes = 2208

        # Jitter buffer: accumulate audio before starting playback to absorb network latency
        # ResembleAI sends audio in bursts with 300-450ms gaps between them
        # We need to buffer enough to cover these gaps before starting playback
        self._jitter_buffer_bytes = 44100  # ~1000ms at 22050Hz to handle 400ms+ network gaps
        self._playback_started: dict[str, bool] = {}  # Track if we've started playback per request

        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Resemble AI service supports metrics generation.
        """
        return True

    def _build_msg(self, text: str = "") -> str:
        """Build a JSON message for the Resemble AI WebSocket API.

        Args:
            text: The text or SSML to synthesize.

        Returns:
            JSON string containing the request payload.
        """
        msg = {
            "voice_uuid": self._voice_id,
            "data": text,
            "binary_response": False,  # Use JSON frames to get timestamps
            "request_id": self._request_id_counter,  # ResembleAI only accepts number
            "output_format": self._settings["output_format"],
            "sample_rate": self._settings["sample_rate"],
            "precision": self._settings["precision"],
            "no_audio_header": True,
        }

        self._request_id_counter += 1
        return json.dumps(msg)

    async def start(self, frame: StartFrame):
        """Start the Resemble AI TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Resemble AI TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Resemble AI TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Connect to the Resemble AI WebSocket."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from the Resemble AI WebSocket."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to Resemble AI."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Resemble AI TTS")
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(self._url, additional_headers=headers)
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection to Resemble AI."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Resemble AI")
                # ResembleAI doesn't send disconnect acknowledgement, set close_timeout to 0
                self._websocket.close_timeout = 0
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            self._audio_buffers.clear()
            self._playback_started.clear()
            self._request_id_to_context.clear()
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns:
            The active WebSocket connection.

        Raises:
            Exception: If websocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruption by stopping current synthesis.

        Args:
            frame: The interruption frame.
            direction: The direction of frame processing.
        """
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()

    async def flush_audio(self):
        """Flush any pending audio and finalize the current context."""
        logger.trace(f"{self}: flushing audio")
        # For Resemble AI, we just wait for the audio_end message
        # which is handled in _process_messages

    async def _process_messages(self):
        """Process incoming WebSocket messages from Resemble AI."""
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                await self.push_error(error_msg=f"Received invalid JSON: {message}")
                continue

            if not msg:
                continue

            msg_type = msg.get("type")
            request_id = msg.get("request_id")

            # Convert request_id to string for audio context tracking
            context_id = self._request_id_to_context.get(request_id, str(request_id))

            # Check if this message belongs to a valid audio context
            if not self.audio_context_available(context_id):
                continue

            if msg_type == "audio":
                await self.stop_ttfb_metrics()
                await self.start_word_timestamps()

                # Decode base64 audio content
                audio_content = msg.get("audio_content", "")
                if not audio_content:
                    continue

                audio_bytes = base64.b64decode(audio_content)
                if len(audio_bytes) == 0:
                    continue

                # Get or create buffer for this request
                if context_id not in self._audio_buffers:
                    self._audio_buffers[context_id] = bytearray()
                    self._playback_started[context_id] = False
                buffer = self._audio_buffers[context_id]

                # Add to buffer
                buffer.extend(audio_bytes)

                # Wait for jitter buffer to fill before starting playback
                # This absorbs network latency gaps (ResembleAI sends in bursts)
                if not self._playback_started.get(context_id, False):
                    if len(buffer) < self._jitter_buffer_bytes:
                        continue
                    self._playback_started[context_id] = True

                # Send complete (even-byte) chunks for PCM_16 alignment
                while len(buffer) >= self._buffer_threshold_bytes:
                    chunk_size = self._buffer_threshold_bytes
                    if chunk_size % 2 != 0:
                        chunk_size -= 1

                    chunk_to_send = bytes(buffer[:chunk_size])
                    self._audio_buffers[context_id] = buffer[chunk_size:]
                    buffer = self._audio_buffers[context_id]

                    if len(chunk_to_send) == 0:
                        continue

                    frame = TTSAudioRawFrame(
                        audio=chunk_to_send,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
                    await self.append_to_audio_context(context_id, frame)

                # Process timestamps if available
                timestamps = msg.get("audio_timestamps", {})
                if timestamps:
                    graph_chars = timestamps.get("graph_chars", [])
                    graph_times = timestamps.get("graph_times", [])

                    # Convert graph_times (start, end pairs) to word timestamps
                    word_times = []
                    for char, times in zip(graph_chars, graph_times):
                        if times and len(times) >= 2:
                            start_time = times[0]
                            word_times.append((char, start_time))

                    if word_times:
                        await self.add_word_timestamps(word_times, context_id)

            elif msg_type == "audio_end":
                await self.stop_ttfb_metrics()

                # Flush remaining buffer, ensuring even length for PCM_16
                buffer = self._audio_buffers.get(context_id, bytearray())
                if buffer:
                    remaining = bytes(buffer)
                    # PCM_16 requires even number of bytes
                    if len(remaining) % 2 != 0:
                        remaining = remaining[:-1]
                    if remaining:
                        frame = TTSAudioRawFrame(
                            audio=remaining,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )
                        await self.append_to_audio_context(context_id, frame)

                # Clean up buffer and playback tracking for this request
                if context_id in self._audio_buffers:
                    del self._audio_buffers[context_id]
                if context_id in self._playback_started:
                    del self._playback_started[context_id]
                    # Clean up request_id mapping
                    if request_id in self._request_id_to_context:
                        del self._request_id_to_context[request_id]

                await self.add_word_timestamps([("TTSStoppedFrame", 0), ("Reset", 0)], context_id)
                await self.remove_audio_context(context_id)

            elif msg_type == "error":
                error_name = msg.get("error_name", "Unknown")
                error_msg = msg.get("message", "Unknown error")
                status_code = msg.get("status_code", 0)
                await self.push_error(
                    error_msg=f"Error: {error_name} (status {status_code}): {error_msg}"
                )

                # Clean up buffer and playback tracking for this request
                if context_id in self._audio_buffers:
                    del self._audio_buffers[context_id]
                if context_id in self._playback_started:
                    del self._playback_started[context_id]

                await self.push_frame(TTSStoppedFrame(context_id=context_id))
                await self.stop_all_metrics()
                await self.push_error(ErrorFrame(error=f"{self} error: {error_name} - {error_msg}"))

                # Check if this is an unrecoverable error (connection-level failure)
                if status_code in [401, 403]:
                    # Close and reconnect for auth errors
                    await self._disconnect_websocket()
                    await self._connect_websocket()
            else:
                logger.warning(f"{self} unknown message type: {msg_type}")

    async def _receive_messages(self):
        """Main loop for receiving messages from Resemble AI."""
        while True:
            try:
                await self._process_messages()
            except Exception as e:
                await self.push_error(error_msg=f"Error in receive loop: {e}", exception=e)
                # Try to reconnect
                logger.debug(f"{self} Resemble AI connection lost, reconnecting")
                await self._connect_websocket()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Resemble AI's streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)

            # Map request_id to context_id for tracking
            self._request_id_to_context[self._request_id_counter] = context_id

            await self.create_audio_context(context_id)

            msg = self._build_msg(text=text)

            try:
                await self._get_websocket().send(msg)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
