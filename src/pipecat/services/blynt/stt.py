#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Blynt speech-to-text service implementation."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

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
from pipecat.services.blynt.models import (
    BlyntSessionContext,
    BlyntSTTOptions,
    DeclaredValues,
    Fact,
    STTLanguages,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

__all__ = [
    "BlyntSessionContext",
    "BlyntSTTOptions",
    "BlyntSTTService",
    "DeclaredValues",
    "Fact",
    "STTLanguages",
]


class BlyntSTTService(WebsocketSTTService):
    """Speech-to-text service using Blynt API.

    Provides real-time speech transcription through WebSocket connection
    to Blynt's STT service. Supports both interim and final transcriptions
    with configurable models and languages.
    """

    # Blynt's realtime API requires 16 kHz mono PCM.
    SAMPLE_RATE = 16000

    def __init__(
        self,
        *,
        options: BlyntSTTOptions,
        **kwargs: Any,
    ) -> None:
        """Initialize BlyntSTTService with options.

        Args:
            options: Configuration options for the STT service.
            **kwargs: Additional arguments passed to parent STTService.
        """
        super().__init__(
            sample_rate=self.SAMPLE_RATE,
            settings=STTSettings(model=None, language=Language(options.language_code)),
            **kwargs,
        )

        logger.info(f"BlyntSTTService initialized with options: {options}")

        self._options = options
        self._receive_task: asyncio.Task[None] | None = None
        self._turn_active = False

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, indicating metrics are supported.
        """
        return True

    async def start(self, frame: StartFrame) -> None:
        """Start the STT service and establish connection.

        Args:
            frame: Frame indicating service should start.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame) -> None:
        """Stop the STT service and close connection.

        Args:
            frame: Frame indicating service should stop.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel the STT service and close connection.

        Args:
            frame: Frame indicating service should be cancelled.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def start_metrics(self) -> None:
        """Start performance metrics collection for transcription processing."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process incoming frames and handle speech events.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # The Blynt API runs in manual turn-taking mode, so the client delimits
        # turns from the VAD-driven speaking frames.
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_metrics()
            await self._send_event({"type": "start_turn"})
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._send_event({"type": "end_turn"})

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Process audio data for speech-to-text transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            None - transcription results are handled via WebSocket responses.
        """
        # If the connection is closed, we need to reconnect
        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        assert self._websocket is not None

        await self._websocket.send(audio)
        # Transcripts are pushed from the receive loop; `process_generator`
        # skips the None yield.
        yield None

    async def _connect(self) -> None:
        """Connect to the Blynt WebSocket service."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = asyncio.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self) -> None:
        """Disconnect from the Blynt WebSocket service."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self) -> None:
        """Establish WebSocket connection to Blynt API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Blynt STT")

            ws_url = self._options.get_ws_url()
            self._websocket = await websocket_connect(
                ws_url, additional_headers=self._options.get_headers()
            )

            # Send start_session event
            await self._send_start_session()

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to Blynt: {e}", exception=e)

    async def _disconnect_websocket(self) -> None:
        """Close WebSocket connection to Blynt API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Blynt STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _send_start_session(self) -> None:
        """Send ClientStartSessionEvent to initialize the session."""
        language_code = self._options.language_code
        event: dict[str, Any] = {
            "type": "start_session",
            "language": language_code,
            "turn_taking_mode": "manual",
        }
        if self._options.session_context is not None:
            payload = self._options.session_context.to_payload()
            if payload is not None:
                event["sessionContext"] = payload

        await self._send_event(event)
        logger.debug(f"Sent start_session with language={language_code}")

    async def _send_event(self, event: dict[str, Any]) -> None:
        """Send a JSON client event over the WebSocket if it is open."""
        if not self._websocket or self._websocket.state is not State.OPEN:
            logger.warning(f"Cannot send {event.get('type')!r}: WebSocket not open")
            return
        await self._websocket.send(json.dumps(event))

    def _get_websocket(self) -> Any:
        """Get the current WebSocket connection."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _process_messages(self) -> None:
        """Process incoming messages from the WebSocket."""
        async for message in self._get_websocket():
            try:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")

                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)

    async def _receive_messages(self) -> None:
        """Receive messages in a loop with reconnection support."""
        while True:
            await self._process_messages()
            # If connection closed, try to reconnect
            logger.debug(f"{self} Blynt connection was disconnected, reconnecting")
            await self._connect_websocket()

    async def _process_response(self, data: dict[str, Any]) -> None:
        """Process a response message from the Blynt API.

        Args:
            data: Parsed JSON data from the WebSocket message.
        """
        event_type = data.get("type")

        if event_type == "turn_partial":
            # Map to InterimTranscriptionFrame
            transcript = data.get("transcript") or ""
            if len(transcript) > 0:
                await self.stop_ttfb_metrics()  # type: ignore[no-untyped-call]
                await self.push_frame(
                    InterimTranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        self._get_language(),
                    )
                )
                logger.debug(f"Interim transcript: {transcript}")

        elif event_type == "turn_ended":
            # Map to TranscriptionFrame (final). `transcript` is null for a
            # false_interruption; only emit a frame when there is text.
            transcript = data.get("transcript") or ""
            kind = data.get("kind", "end_of_utterance")

            if len(transcript) > 0:
                await self.stop_ttfb_metrics()  # type: ignore[no-untyped-call]
                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        self._get_language(),
                    )
                )
                logger.info(f"Final transcript ({kind}): {transcript}")
                await self.stop_processing_metrics()  # type: ignore[no-untyped-call]

        elif event_type in ("session_started", "turn_started"):
            logger.debug(f"Received {event_type}")

        elif event_type == "error":
            error_msg = data.get("message", "Unknown error")
            await self.push_error(error_msg=error_msg)

        else:
            logger.warning(f"Unknown server event type: {event_type}")

    def _get_language(self) -> Language | None:
        """Get the language as a Language enum value.

        Returns:
            Language enum value if available, None otherwise.
        """
        try:
            # Use base language code (e.g. "fr" from "fr-FR")
            return Language(self._options.language_code.split("-")[0])
        except (ValueError, KeyError):
            return None
