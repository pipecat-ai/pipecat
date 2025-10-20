#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AssemblyAI speech-to-text service implementation.

This module provides integration with AssemblyAI's real-time speech-to-text
WebSocket API for streaming audio transcription.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict
from urllib.parse import urlencode

from loguru import logger

from pipecat import __version__ as pipecat_version
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
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

from .models import (
    AssemblyAIConnectionParams,
    BaseMessage,
    BeginMessage,
    TerminationMessage,
    TurnMessage,
)

try:
    import websockets
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use AssemblyAI, you need to `pip install "pipecat-ai[assemblyai]"`.')
    raise Exception(f"Missing module: {e}")


class AssemblyAISTTService(STTService):
    """AssemblyAI real-time speech-to-text service.

    Provides real-time speech transcription using AssemblyAI's WebSocket API.
    Supports both interim and final transcriptions with configurable parameters
    for audio processing and connection management.
    """

    def __init__(
        self,
        *,
        api_key: str,
        language: Language = Language.EN,  # AssemblyAI only supports English
        api_endpoint_base_url: str = "wss://streaming.assemblyai.com/v3/ws",
        connection_params: AssemblyAIConnectionParams = AssemblyAIConnectionParams(),
        vad_force_turn_endpoint: bool = True,
        **kwargs,
    ):
        """Initialize the AssemblyAI STT service.

        Args:
            api_key: AssemblyAI API key for authentication.
            language: Language code for transcription. Defaults to English (Language.EN).
            api_endpoint_base_url: WebSocket endpoint URL. Defaults to AssemblyAI's streaming endpoint.
            connection_params: Connection configuration parameters. Defaults to AssemblyAIConnectionParams().
            vad_force_turn_endpoint: Whether to force turn endpoint on VAD stop. Defaults to True.
            **kwargs: Additional arguments passed to parent STTService class.
        """
        self._api_key = api_key
        self._language = language
        self._api_endpoint_base_url = api_endpoint_base_url
        self._connection_params = connection_params
        self._vad_force_turn_endpoint = vad_force_turn_endpoint

        super().__init__(sample_rate=self._connection_params.sample_rate, **kwargs)

        self._websocket = None
        self._termination_event = asyncio.Event()
        self._received_termination = False
        self._connected = False

        self._receive_task = None

        self._audio_buffer = bytearray()
        self._chunk_size_ms = 50
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
        self._chunk_size_bytes = int(self._chunk_size_ms * self._sample_rate * 2 / 1000)
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

        while len(self._audio_buffer) >= self._chunk_size_bytes:
            chunk = bytes(self._audio_buffer[: self._chunk_size_bytes])
            self._audio_buffer = self._audio_buffer[self._chunk_size_bytes :]
            await self._websocket.send(chunk)

        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for VAD and metrics handling.

        Args:
            frame: Frame to process.
            direction: Direction of frame processing.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            await self.start_ttfb_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._vad_force_turn_endpoint:
                await self._websocket.send(json.dumps({"type": "ForceEndpoint"}))
            await self.start_processing_metrics()

    @traced_stt
    async def _trace_transcription(self, transcript: str, is_final: bool, language: Language):
        """Record transcription event for tracing."""
        pass

    def _build_ws_url(self) -> str:
        """Build WebSocket URL with query parameters using urllib.parse.urlencode."""
        params = {}
        for k, v in self._connection_params.model_dump().items():
            if v is not None:
                if k == "keyterms_prompt":
                    params[k] = json.dumps(v)
                elif isinstance(v, bool):
                    params[k] = str(v).lower()
                else:
                    params[k] = v

        if params:
            query_string = urlencode(params)
            return f"{self._api_endpoint_base_url}?{query_string}"
        return self._api_endpoint_base_url

    async def _connect(self):
        try:
            ws_url = self._build_ws_url()
            headers = {
                "Authorization": self._api_key,
                "User-Agent": f"AssemblyAI/1.0 (integration=Pipecat/{pipecat_version})",
            }
            self._websocket = await websocket_connect(
                ws_url,
                additional_headers=headers,
            )
            self._connected = True
            self._receive_task = self.create_task(self._receive_task_handler())

            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"Failed to connect to AssemblyAI: {e}")
            self._connected = False
            raise

    async def _disconnect(self):
        """Disconnect from AssemblyAI WebSocket and wait for termination message."""
        if not self._connected or not self._websocket:
            return

        try:
            self._termination_event.clear()
            self._received_termination = False

            if len(self._audio_buffer) > 0:
                await self._websocket.send(bytes(self._audio_buffer))
                self._audio_buffer.clear()

            try:
                await self._websocket.send(json.dumps({"type": "Terminate"}))

                try:
                    await asyncio.wait_for(self._termination_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Timed out waiting for termination message from server")

            except Exception as e:
                logger.warning(f"Error during termination handshake: {e}")

            if self._receive_task:
                await self.cancel_task(self._receive_task)

            await self._websocket.close()

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

        finally:
            self._websocket = None
            self._connected = False
            self._receive_task = None
            await self._call_event_handler("on_disconnected")

    async def _receive_task_handler(self):
        """Handle incoming WebSocket messages."""
        try:
            while self._connected:
                try:
                    message = await self._websocket.recv()
                    data = json.loads(message)
                    await self._handle_message(data)
                except websockets.exceptions.ConnectionClosedOK:
                    break
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    break

        except Exception as e:
            logger.error(f"Fatal error in receive handler: {e}")

    def _parse_message(self, message: Dict[str, Any]) -> BaseMessage:
        """Parse a raw message into the appropriate message type."""
        msg_type = message.get("type")

        if msg_type == "Begin":
            return BeginMessage.model_validate(message)
        elif msg_type == "Turn":
            return TurnMessage.model_validate(message)
        elif msg_type == "Termination":
            return TerminationMessage.model_validate(message)
        else:
            raise ValueError(f"Unknown message type: {msg_type}")

    async def _handle_message(self, message: Dict[str, Any]):
        """Handle AssemblyAI WebSocket messages."""
        try:
            parsed_message = self._parse_message(message)

            if isinstance(parsed_message, BeginMessage):
                logger.debug(
                    f"Session Begin: {parsed_message.id} (expires at {parsed_message.expires_at})"
                )
            elif isinstance(parsed_message, TurnMessage):
                await self._handle_transcription(parsed_message)
            elif isinstance(parsed_message, TerminationMessage):
                await self._handle_termination(parsed_message)
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_termination(self, message: TerminationMessage):
        """Handle termination message."""
        self._received_termination = True
        self._termination_event.set()

        logger.info(
            f"Session Terminated: Audio Duration={message.audio_duration_seconds}s, "
            f"Session Duration={message.session_duration_seconds}s"
        )
        await self.push_frame(EndFrame())

    async def _handle_transcription(self, message: TurnMessage):
        """Handle transcription results."""
        if not message.transcript:
            return
        await self.stop_ttfb_metrics()
        if message.end_of_turn and (
            not self._connection_params.formatted_finals or message.turn_is_formatted
        ):
            await self.push_frame(
                TranscriptionFrame(
                    message.transcript,
                    self._user_id,
                    time_now_iso8601(),
                    self._language,
                    message,
                )
            )
            await self._trace_transcription(message.transcript, True, self._language)
            await self.stop_processing_metrics()
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    message.transcript,
                    self._user_id,
                    time_now_iso8601(),
                    self._language,
                    message,
                )
            )
