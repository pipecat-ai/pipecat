#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Nemo Speech-to-Text service implementation.

This module provides a WebSocket-based STT service that integrates with
a custom Nemo-based ASR server for real-time speech recognition.
"""

import base64
import json
from collections import deque
from enum import IntEnum
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Nemo STT, you need to `pip install pipecat-ai[nemo]`.")
    raise Exception(f"Missing module: {e}")


class NemoLatencyMode(IntEnum):
    """Latency modes for Nemo ASR service.

    The latency mode affects the accuracy/latency tradeoff of the transcription.
    Lower latency means faster but potentially less accurate transcriptions.
    """

    LOW = 80
    MEDIUM = 480
    HIGH = 1040


class NemoInputParams(BaseModel):
    """Configuration parameters for Nemo ASR service.

    Parameters:
        latency_ms: Target latency in milliseconds. Supported values are
            80 (low latency), 480 (balanced, default), and 1040 (high accuracy).
    """

    latency_ms: int = Field(
        default=NemoLatencyMode.MEDIUM,
        description="Target latency in milliseconds. Supported: 80, 480, 1040",
    )


class NemoSTTService(WebsocketSTTService):
    """Speech-to-text service using a custom Nemo-based ASR server.

    Provides real-time speech transcription through WebSocket connection
    to a Nemo ASR service. Supports configurable latency modes and
    both interim and final transcriptions.

    The service communicates using a JSON-based protocol:
    - Client sends: start_stream, audio (base64 PCM16), end_stream
    - Server sends: status, transcription

    Event handlers available:

    - on_connected: Called when connected to the Nemo ASR server
    - on_disconnected: Called when disconnected from the server
    - on_connection_error: Called when a connection error occurs

    Example::

        @stt.event_handler("on_connected")
        async def on_connected(stt, frame):
            print("Connected to Nemo ASR")
    """

    def __init__(
        self,
        *,
        url: str = "ws://localhost:8000/ws/transcribe",
        sample_rate: Optional[int] = 16000,
        params: Optional[NemoInputParams] = None,
        ttfs_p99_latency: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the Nemo STT service.

        Args:
            url: WebSocket URL of the Nemo ASR server.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            params: Configuration parameters for the STT service.
            ttfs_p99_latency: P99 latency from speech end to final transcript
                in seconds. Override this based on your deployment's measured
                latency. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to parent WebsocketSTTService.
        """
        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=30,
            keepalive_interval=10,
            **kwargs,
        )

        self._params = params or NemoInputParams()
        self._url = url
        self._receive_task = None
        self._stream_started = False
        self._connecting = False
        self._waiting_for_final = False  # Track if we're waiting for final transcription

        # Audio buffering - match working client's ~320ms batches
        # The standalone Python client batches audio to reduce network overhead
        # and provide better context to the ASR model
        self._audio_buffer = bytearray()
        self._chunk_duration_ms = 160  # Target chunk size in ms
        self._batch_size = 2  # Send every 2 chunks = 320ms
        # For 16kHz mono: 16000 samples/s * 2 bytes/sample * 0.32s = 10,240 bytes
        self._target_buffer_bytes = (
            sample_rate * 2 * self._chunk_duration_ms * self._batch_size
        ) // 1000

        # Pre-speech buffer - captures audio BEFORE VAD fires
        # This prevents the beginning of utterances from being cut off
        # Similar to speech_pad_ms in working client's VAD
        self._pre_speech_pad_ms = 200  # 200ms of pre-speech audio
        # Calculate max bytes for pre-speech buffer
        self._pre_speech_max_bytes = (sample_rate * 2 * self._pre_speech_pad_ms) // 1000
        # Use deque with maxlen to maintain a rolling buffer
        self._pre_speech_buffer: deque[bytes] = deque()
        self._pre_speech_buffer_bytes = 0  # Track current size

        self.set_model_name("nemo-asr")

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, indicating metrics are supported.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the STT service and establish connection.

        Args:
            frame: Frame indicating service should start.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close connection.

        Args:
            frame: Frame indicating service should stop.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close connection.

        Args:
            frame: Frame indicating service should be cancelled.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _start_metrics(self):
        """Start performance metrics collection for transcription processing."""
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle speech events.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._start_metrics()
            # Start a new stream for this utterance
            await self._start_stream()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # End the stream to get final transcription
            await self._end_stream()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Nemo ASR for transcription.

        Audio is buffered to ~320ms chunks before sending to match the
        behavior of the working standalone Python client, which improves
        transcription quality by providing better context to the ASR model.

        Pre-speech audio is buffered even before VAD fires to capture the
        beginning of utterances that would otherwise be lost.

        Args:
            audio: Raw PCM16 audio bytes to transcribe.

        Yields:
            None - transcription results are handled via WebSocket responses.
        """
        if self._stream_started:
            # Active stream - buffer audio for batching and send
            if self._websocket and self._websocket.state is State.OPEN:
                self._audio_buffer.extend(audio)

                # Send when buffer reaches target size (~320ms)
                if len(self._audio_buffer) >= self._target_buffer_bytes:
                    await self._send_buffered_audio()
        else:
            # No active stream - maintain pre-speech rolling buffer
            # This captures audio BEFORE VAD fires so we don't lose
            # the beginning of utterances
            self._pre_speech_buffer.append(audio)
            self._pre_speech_buffer_bytes += len(audio)

            # Trim buffer if it exceeds max size (rolling buffer behavior)
            while self._pre_speech_buffer_bytes > self._pre_speech_max_bytes:
                if self._pre_speech_buffer:
                    removed = self._pre_speech_buffer.popleft()
                    self._pre_speech_buffer_bytes -= len(removed)
                else:
                    break

        yield None

    async def _send_buffered_audio(self):
        """Send buffered audio to the ASR server.

        Sends accumulated audio as a single chunk and clears the buffer.
        """
        if not self._audio_buffer:
            return

        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        try:
            audio_base64 = base64.b64encode(bytes(self._audio_buffer)).decode("utf-8")
            message = {
                "type": "audio",
                "data": audio_base64,
                "sample_rate": self.sample_rate,
                "is_final": False,
            }
            await self._websocket.send(json.dumps(message))
            self._audio_buffer.clear()
        except Exception as e:
            logger.warning(f"Failed to send audio: {e}")

    async def _connect(self):
        """Connect to Nemo ASR and start receive task."""
        await self._connect_websocket()

        await super()._connect()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect and cleanup tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection (without starting stream)."""
        if self._connecting:
            return

        self._connecting = True
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"Connecting to Nemo ASR at {self._url}")
            self._websocket = await websocket_connect(self._url)

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unable to connect to Nemo ASR: {e}", exception=e)
        finally:
            self._connecting = False

    async def _start_stream(self):
        """Start a new transcription stream for an utterance."""
        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        if self._stream_started:
            return  # Already started

        # Transfer pre-speech buffer to audio buffer
        # This includes audio captured BEFORE VAD fired, preventing
        # the beginning of utterances from being cut off
        self._audio_buffer.clear()
        if self._pre_speech_buffer:
            for chunk in self._pre_speech_buffer:
                self._audio_buffer.extend(chunk)
            logger.debug(
                f"Including {self._pre_speech_buffer_bytes} bytes of pre-speech audio "
                f"({self._pre_speech_buffer_bytes * 1000 // (self.sample_rate * 2)}ms)"
            )
            self._pre_speech_buffer.clear()
            self._pre_speech_buffer_bytes = 0

        try:
            start_message = {
                "type": "start_stream",
                "config": {"latency_ms": self._params.latency_ms},
            }
            await self._websocket.send(json.dumps(start_message))
            # Set _stream_started=True IMMEDIATELY after sending, don't wait for server confirmation
            # This matches the working client which sends audio right after start_stream
            # The server can handle audio arriving before it sends stream_started confirmation
            self._stream_started = True
            logger.debug("Sent start_stream to Nemo ASR")

            # Send pre-speech audio immediately if we have enough
            if len(self._audio_buffer) >= self._target_buffer_bytes:
                await self._send_buffered_audio()
        except Exception as e:
            logger.warning(f"Failed to send start_stream: {e}")

    async def _end_stream(self):
        """End the current transcription stream to get final transcription."""
        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        if not self._stream_started:
            return  # No active stream

        # Flush any remaining buffered audio before ending stream
        # This ensures all audio is sent to the ASR server
        if self._audio_buffer:
            await self._send_buffered_audio()

        # Set _stream_started = False IMMEDIATELY to prevent audio being sent
        # in the gap between sending end_stream and receiving stream_ended status
        self._stream_started = False

        try:
            end_message = {"type": "end_stream"}
            await self._websocket.send(json.dumps(end_message))
            self._waiting_for_final = True
            logger.debug("Sent end_stream to Nemo ASR")
        except Exception as e:
            logger.warning(f"Failed to send end_stream: {e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Nemo ASR")
                # End any active stream before closing
                if self._stream_started:
                    try:
                        end_message = {"type": "end_stream"}
                        await self._websocket.send(json.dumps(end_message))
                    except Exception:
                        pass  # Ignore errors when closing
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            self._stream_started = False
            self._waiting_for_final = False
            self._pre_speech_buffer.clear()
            self._pre_speech_buffer_bytes = 0
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns:
            The WebSocket connection object.

        Raises:
            Exception: If WebSocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Continuously receive and process WebSocket messages."""
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _process_response(self, data: dict):
        """Process a response message from Nemo ASR.

        Args:
            data: Parsed JSON response from the server.
        """
        msg_type = data.get("type")

        if msg_type == "status":
            status = data.get("status")
            message = data.get("message", "")

            if status == "stream_started":
                self._stream_started = True
                self._waiting_for_final = False
                logger.debug("Nemo ASR stream started")
            elif status == "stream_ended":
                self._stream_started = False
                self._waiting_for_final = False
                logger.debug("Nemo ASR stream ended - ready for next utterance")
            elif status == "error":
                await self.push_error(error_msg=f"Nemo ASR error: {message}")
            elif status == "ready":
                logger.debug("Nemo ASR server ready")

        elif msg_type == "transcription":
            await self._on_transcription(data)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing.

        Args:
            transcript: The transcribed text.
            is_final: Whether this is a final transcription.
            language: Optional language code.
        """
        pass

    async def _on_transcription(self, data: dict):
        """Handle transcription messages from the server.

        Args:
            data: Transcription message data containing text and is_partial fields.
        """
        text = data.get("text", "").strip()
        is_partial = data.get("is_partial", True)

        if not text:
            return

        if is_partial:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    None,
                    result=data,
                )
            )
        else:
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    None,
                    result=data,
                )
            )
            await self._handle_transcription(text, True, None)
            await self.stop_processing_metrics()

    async def _send_keepalive(self, silence: bytes):
        """Send silent audio to keep the connection alive.

        Args:
            silence: Silent 16-bit mono PCM audio bytes.
        """
        if self._stream_started and self._websocket and self._websocket.state is State.OPEN:
            try:
                audio_base64 = base64.b64encode(silence).decode("utf-8")
                message = {
                    "type": "audio",
                    "data": audio_base64,
                    "sample_rate": self.sample_rate,
                    "is_final": False,
                }
                await self._websocket.send(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send keepalive: {e}")
