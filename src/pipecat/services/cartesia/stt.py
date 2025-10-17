#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cartesia Speech-to-Text service implementation.

This module provides a WebSocket-based STT service that integrates with
the Cartesia Live transcription API for real-time speech recognition.
"""

import asyncio
import json
import urllib.parse
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Cartesia, you need to `pip install pipecat-ai[cartesia]`.")
    raise Exception(f"Missing module: {e}")


class CartesiaLiveOptions:
    """Configuration options for Cartesia Live STT service.

    Manages transcription parameters including model selection, language,
    audio encoding format, and sample rate settings.
    """

    def __init__(
        self,
        *,
        model: str = "ink-whisper",
        language: str = Language.EN.value,
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
        **kwargs,
    ):
        """Initialize CartesiaLiveOptions with default or provided parameters.

        Args:
            model: The transcription model to use. Defaults to "ink-whisper".
            language: Target language for transcription. Defaults to English.
            encoding: Audio encoding format. Defaults to "pcm_s16le".
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            **kwargs: Additional parameters for the transcription service.
        """
        self.model = model
        self.language = language
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.additional_params = kwargs

    def to_dict(self):
        """Convert options to dictionary format.

        Returns:
            Dictionary containing all configuration parameters.
        """
        params = {
            "model": self.model,
            "language": self.language if isinstance(self.language, str) else self.language.value,
            "encoding": self.encoding,
            "sample_rate": str(self.sample_rate),
        }

        return params

    def items(self):
        """Get configuration items as key-value pairs.

        Returns:
            Iterator of (key, value) tuples for all configuration parameters.
        """
        return self.to_dict().items()

    def get(self, key, default=None):
        """Get a configuration value by key.

        Args:
            key: The configuration parameter name to retrieve.
            default: Default value if key is not found.

        Returns:
            The configuration value or default if not found.
        """
        if hasattr(self, key):
            return getattr(self, key)
        return self.additional_params.get(key, default)

    @classmethod
    def from_json(cls, json_str: str) -> "CartesiaLiveOptions":
        """Create options from JSON string.

        Args:
            json_str: JSON string containing configuration parameters.

        Returns:
            New CartesiaLiveOptions instance with parsed parameters.
        """
        return cls(**json.loads(json_str))


class CartesiaSTTService(WebsocketSTTService):
    """Speech-to-text service using Cartesia Live API.

    Provides real-time speech transcription through WebSocket connection
    to Cartesia's Live transcription service. Supports both interim and
    final transcriptions with configurable models and languages.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "",
        sample_rate: int = 16000,
        live_options: Optional[CartesiaLiveOptions] = None,
        **kwargs,
    ):
        """Initialize CartesiaSTTService with API key and options.

        Args:
            api_key: Authentication key for Cartesia API.
            base_url: Custom API endpoint URL. If empty, uses default.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            live_options: Configuration options for transcription service.
            **kwargs: Additional arguments passed to parent STTService.
        """
        sample_rate = sample_rate or (live_options.sample_rate if live_options else None)
        super().__init__(sample_rate=sample_rate, **kwargs)

        default_options = CartesiaLiveOptions(
            model="ink-whisper",
            language=Language.EN.value,
            encoding="pcm_s16le",
            sample_rate=sample_rate,
        )

        merged_options = default_options
        if live_options:
            merged_options_dict = default_options.to_dict()
            merged_options_dict.update(live_options.to_dict())
            merged_options = CartesiaLiveOptions(
                **{
                    k: v
                    for k, v in merged_options_dict.items()
                    if not isinstance(v, str) or v != "None"
                }
            )

        self._settings = merged_options
        self.set_model_name(merged_options.model)
        self._api_key = api_key
        self._base_url = base_url or "api.cartesia.ai"
        self._receive_task = None

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

    async def start_metrics(self):
        """Start performance metrics collection for transcription processing."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle speech events.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            await self.start_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Send finalize command to flush the transcription session
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send("finalize")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data for speech-to-text transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            None - transcription results are handled via WebSocket responses.
        """
        # If the connection is closed, due to timeout, we need to reconnect when the user starts speaking again
        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        await self._websocket.send(audio)
        yield None

    async def _connect(self):
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = asyncio.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Cartesia STT")

            params = self._settings.to_dict()
            ws_url = f"wss://{self._base_url}/stt/websocket?{urllib.parse.urlencode(params)}"
            headers = {"Cartesia-Version": "2025-04-16", "X-API-Key": self._api_key}

            self._websocket = await websocket_connect(ws_url, additional_headers=headers)
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self}: unable to connect to Cartesia: {e}")

    async def _disconnect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Cartesia STT")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
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
            # Cartesia times out after 5 minutes of innactivity (no keepalive
            # mechanism is available). So, we try to reconnect.
            logger.debug(f"{self} Cartesia connection was disconnected (timeout?), reconnecting")
            await self._connect_websocket()

    async def _process_response(self, data):
        if "type" in data:
            if data["type"] == "transcript":
                await self._on_transcript(data)

            elif data["type"] == "error":
                logger.error(f"Cartesia error: {data.get('message', 'Unknown error')}")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _on_transcript(self, data):
        if "text" not in data:
            return

        transcript = data.get("text", "")
        is_final = data.get("is_final", False)
        language = None

        if "language" in data:
            try:
                language = Language(data["language"])
            except (ValueError, KeyError):
                pass

        if len(transcript) > 0:
            await self.stop_ttfb_metrics()
            if is_final:
                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                    )
                )
                await self._handle_transcription(transcript, is_final, language)
                await self.stop_processing_metrics()
            else:
                # For interim transcriptions, just push the frame without tracing
                await self.push_frame(
                    InterimTranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                    )
                )
