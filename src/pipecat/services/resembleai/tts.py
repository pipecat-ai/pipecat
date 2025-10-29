#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ResembleAI's text-to-speech service implementations.

This module provides both WebSocket based text-to-speech services
using ResembleAI's API for streaming and batch audio synthesis.
"""

import base64
import json
import uuid
from typing import AsyncGenerator, Optional

from loguru import logger

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
from pipecat.services.tts_service import AudioContextWordTTSService
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    raise Exception(f"Missing module: {e}")


class ResembleAITTSService(AudioContextWordTTSService):
    """ResembleAI TTS service with WebSocket streaming and word timestamps.

    Provides text-to-speech using ResembleAI's streaming WebSocket API.
    Supports word-level timestamps, audio context management, and various voice
    customization options including speed and emotion controls.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_uuid: str,
        url: str = "wss://websocket.cluster.resemble.ai/stream",
        sample_rate: Optional[int] = 22050,
        aggregate_sentences: Optional[bool] = True,
        **kwargs,
    ):
        """Initialize the ResembleAI TTS service.

        Args:
            api_key: ResembleAI API key for authentication.
            voice_uuid: ID of the voice to use for synthesis.
            url: WebSocket URL for ResembleAI TTS API.
            sample_rate: Audio sample rate. If None, uses default.
            aggregate_sentences: Whether to aggregate sentences within the TTSService.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._voice_uuid = voice_uuid

        self._websocket = None
        self._receive_task = None
        self._context_id = None

        # Indicates if we have sent TTSStartedFrame. It will reset to False when
        # there's an interruption or TTSStoppedFrame.
        self._started = False

        # Buffer audio to smooth playback and handle ResembleAI's irregular chunk sizes
        # ResembleAI may send odd-length data even for PCM_16, so buffering helps us
        # create properly aligned frames while maintaining smooth audio output
        self._audio_buffer = bytearray()
        self._buffer_threshold_bytes = 8640

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as ResembleAI service supports metrics generation.
        """
        return True

    def _build_msg(self, text: str = "", context_id: Optional[int] = None) -> str:
        """Build message for ResembleAI WebSocket API."""
        logger.debug(f"Text {text} and request_id: {context_id} ")

        msg = {
            "voice_uuid": self._voice_uuid,
            "data": text,
            "binary_response": False,
            "request_id": context_id,
            "output_format": "wav",
            "sample_rate": self.sample_rate,
            "precision": "PCM_16",
            "no_audio_header": True,
        }
        return json.dumps(msg)

    async def start(self, frame: StartFrame):
        """Start the ResembleAI TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the ResembleAI TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Stop the ResembleAI TTS service.

        Args:
            frame: The end frame.
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

    def _is_websocket_connected(self) -> bool:
        """Check if WebSocket is properly connected and ready."""
        return self._websocket is not None and self._websocket.state is State.OPEN

    def _is_websocket_disconnected(self) -> bool:
        """Check if WebSocket is disconnected or in bad state."""
        return self._websocket is None or self._websocket.state in (State.CLOSED, State.CLOSING)

    async def _connect_websocket(self):
        """Connect to ResembleAI's websocket API with configured settings."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to ResembleAI TTS")
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(self._url, additional_headers=headers)
            logger.debug("Connected to ResembleAI TTS")
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from ResembleAI TTS")
                # ResembleAI doesn't send any disconnect acknowledgement need to set close_timeout to 0
                self._websocket.close_timeout = 0
                await self._websocket.close()
                logger.debug("Disconnected from ResembleAI TTS")
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None
            self._context_id = None
            self._started = False
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        self._started = False

        # ResembleAI doesn't support interruptions, so disconnect/reconnect
        logger.debug("Interruption: handling by disconnect/reconnect")

        if self._is_websocket_connected():
            try:
                await self._disconnect()
                await self._connect()
            except Exception as e:
                logger.error(f"Error during interruption reconnection: {e}")
        else:
            logger.debug("Interruption: WebSocket already disconnected")

    async def flush_audio(self):
        """Flush any pending audio and finalize the current context."""
        if self._is_websocket_disconnected():
            return

        # ResembleAI does not support audio flushing
        logger.trace(f"{self}: flushing audio")

        if self._context_id:
            logger.trace(f"{self}: Cannot flush, clearing context {self._context_id}")
            self._context_id = None

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes."""
        await super().push_frame(frame, direction)

        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
            self._started = False
            logger.debug(f"TTS stopped/interrupted, resetting _started flag")

    async def _receive_messages(self):
        """Process incoming WebSocket messages from ResembleAI."""
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)

                # Check if request_id exists
                if "request_id" not in msg:
                    logger.warning(f"Message without request_id: {msg}")
                    continue

                request_id = str(msg["request_id"])

                if not self.audio_context_available(request_id):
                    logger.debug(f"Ignoring message from unavailable context: {request_id}")
                    continue

                if msg.get("type") == "audio":
                    await self.stop_ttfb_metrics()

                    audio_bytes = base64.b64decode(msg["audio_content"])

                    # Store current context
                    self._context_id = request_id

                    # Add to buffer (handle odd bytes from ResembleAI)
                    self._audio_buffer.extend(audio_bytes)

                    # Send complete (even-byte chunks)
                    while len(self._audio_buffer) >= self._buffer_threshold_bytes:
                        chunk_size = self._buffer_threshold_bytes
                        if chunk_size % 2 != 0:
                            chunk_size -= 1

                        chunk_to_send = bytes(self._audio_buffer[:chunk_size])
                        self._audio_buffer = self._audio_buffer[chunk_size:]

                        frame = TTSAudioRawFrame(
                            audio=chunk_to_send,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                        )
                        await self.append_to_audio_context(request_id, frame)

                elif msg.get("type") == "audio_end":
                    await self.stop_ttfb_metrics()

                    # Send remaining buffer, ensuring even length for PCM_16
                    if self._audio_buffer and self._context_id == request_id:
                        remaining = bytes(self._audio_buffer)

                        # PCM_16 requires even number of bytes
                        if len(remaining) % 2 != 0:
                            logger.debug(f"Trimming final odd byte from {len(remaining)} bytes")
                            remaining = remaining[:-1]

                        if remaining:  # Only send if we have data after trimming
                            frame = TTSAudioRawFrame(
                                audio=remaining,
                                sample_rate=self.sample_rate,
                                num_channels=1,
                            )
                            await self.append_to_audio_context(request_id, frame)

                    # Clear buffer
                    self._audio_buffer.clear()
                    self._context_id = None

                    await self.remove_audio_context(request_id)
                    logger.debug(f"Removing request id: {request_id}")

                elif msg.get("type") == "error":
                    logger.error(f"{self} error: {msg}")
                    await self.remove_audio_context(request_id)

                    # Only clear current context if it matches the errored one
                    if self._context_id == request_id:
                        self._audio_buffer.clear()
                        self._context_id = None
                        logger.debug(f"Cleared current context {request_id} due to error")

                        await self.push_frame(TTSStoppedFrame())
                        await self.stop_all_metrics()
                    else:
                        logger.debug(
                            f"Error was for context {request_id}, keeping current context {self._context_id}"
                        )

                    await self.push_error(
                        ErrorFrame(f"{self} error: {msg.get('message', 'Unknown error')}")
                    )

                else:
                    logger.debug(f"Unknown message type: {msg.get('type')}: {msg}")

            except Exception as e:
                logger.error(f"Error processing message: {e}, message: {message}")
                continue

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using ResembleAI's streaming API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if self._is_websocket_disconnected():
                await self._connect()

            if not self._started:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._started = True

            # Create a NEW context_id for EACH sentence
            self._context_id = str(uuid.uuid4().int)
            logger.debug(f"Creating new context with: {self._context_id}")

            await self.create_audio_context(self._context_id)

            msg = self._build_msg(text=text, context_id=self._context_id)

            try:
                await self._get_websocket().send(msg)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                self._started = False
                await self.remove_audio_context(self._context_id)
                await self._disconnect()
                await self._connect()
                return

            yield None

        except Exception as e:
            logger.error(f"{self} exception: {e}")
