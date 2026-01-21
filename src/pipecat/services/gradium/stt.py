#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gradium's speech-to-text service implementation.

This module provides integration with Gradium's real-time speech-to-text
WebSocket API for streaming audio transcription.
"""

import base64
import json
from typing import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Gradium, you need to `pip install "pipecat-ai[gradium]"`.')
    raise Exception(f"Missing module: {e}")

SAMPLE_RATE = 24000


class GradiumSTTService(WebsocketSTTService):
    """Gradium real-time speech-to-text service.

    Provides real-time speech transcription using Gradium's WebSocket API.
    Supports both interim and final transcriptions with configurable parameters
    for audio processing and connection management.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_endpoint_base_url: str = "wss://eu.api.gradium.ai/api/speech/asr",
        json_config: str | None = None,
        **kwargs,
    ):
        """Initialize the Gradium STT service.

        Args:
            api_key: Gradium API key for authentication.
            api_endpoint_base_url: WebSocket endpoint URL. Defaults to Gradium's streaming endpoint.
            json_config: Optional JSON configuration string for additional model settings.
            **kwargs: Additional arguments passed to parent STTService class.
        """
        super().__init__(sample_rate=SAMPLE_RATE, **kwargs)

        self._api_key = api_key
        self._api_endpoint_base_url = api_endpoint_base_url
        self._websocket = None
        self._json_config = json_config

        self._receive_task = None

        self._audio_buffer = bytearray()
        self._chunk_size_ms = 80
        self._chunk_size_bytes = 0

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the speech-to-text service.

        Args:
            frame: Start frame to begin processing.
        """
        await super().start(frame)
        self._chunk_size_bytes = int(self._chunk_size_ms * self.sample_rate * 2 / 1000)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the speech-to-text service.

        Args:
            frame: End frame to stop processing.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the speech-to-text service.

        Args:
            frame: Cancel frame to abort processing.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data for speech-to-text conversion.

        Args:
            audio: Raw audio bytes to process.

        Yields:
            None (processing handled via WebSocket messages).
        """
        self._audio_buffer.extend(audio)
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

        while len(self._audio_buffer) >= self._chunk_size_bytes:
            chunk = bytes(self._audio_buffer[: self._chunk_size_bytes])
            self._audio_buffer = self._audio_buffer[self._chunk_size_bytes :]
            chunk = base64.b64encode(chunk).decode("utf-8")
            msg = {"type": "audio", "audio": chunk}
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(json.dumps(msg))

        yield None

    @traced_stt
    async def _trace_transcription(self, transcript: str, is_final: bool, language: Language):
        """Record transcription event for tracing."""
        pass

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            ws_url = self._api_endpoint_base_url
            headers = {
                "x-api-key": self._api_key,
                "x-api-source": "pipecat",
            }
            self._websocket = await websocket_connect(
                ws_url,
                additional_headers=headers,
            )
            await self._call_event_handler("on_connected")
            setup_msg = {
                "type": "setup",
                "input_format": "pcm",
            }
            if self._json_config is not None:
                setup_msg["json_config"] = self._json_config
            await self._websocket.send(json.dumps(setup_msg))
            ready_msg = await self._websocket.recv()
            ready_msg = json.loads(ready_msg)
            if ready_msg["type"] == "error":
                raise Exception(f"received error {ready_msg['message']}")
            if ready_msg["type"] != "ready":
                raise Exception(f"unexpected first message type {ready_msg['type']}")

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            raise

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _disconnect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Gradium STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _process_messages(self):
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")

    async def _receive_messages(self):
        while True:
            await self._process_messages()
            logger.debug(f"{self} Gradium connection was disconnected (timeout?), reconnecting")
            await self._connect_websocket()

    async def _process_response(self, msg):
        type_ = msg.get("type", "")
        if type_ == "text":
            await self._handle_text(msg["text"])
        elif type_ == "end_of_stream":
            await self._handle_end_of_stream()
        elif type_ == "error":
            await self.push_error(error_msg=f"Error: {msg}")

    async def _handle_end_of_stream(self):
        """Handle termination message."""
        logger.debug("Received end_of_stream message from server")

    async def _handle_text(self, text: str):
        """Handle transcription results."""
        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
            )
        )
