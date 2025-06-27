#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use AssemblyAI, you need to `pip install "pipecat-ai[assemblyai]"`.')
    raise Exception(f"Missing module: {e}")


class AssemblyAISTTService(STTService):
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
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._chunk_size_bytes = int(self._chunk_size_ms * self._sample_rate * 2 / 1000)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        self._audio_buffer.extend(audio)

        while len(self._audio_buffer) >= self._chunk_size_bytes:
            chunk = bytes(self._audio_buffer[: self._chunk_size_bytes])
            self._audio_buffer = self._audio_buffer[self._chunk_size_bytes :]
            await self._websocket.send(chunk)

        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
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
        params = {
            k: str(v).lower() if isinstance(v, bool) else v
            for k, v in self._connection_params.model_dump().items()
            if v is not None
        }
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
            self._websocket = await websockets.connect(
                ws_url,
                extra_headers=headers,
            )
            self._connected = True
            self._receive_task = self.create_task(self._receive_task_handler())
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
                    await asyncio.wait_for(
                        self._termination_event.wait(),
                        timeout=5.0,
                    )
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

    async def _receive_task_handler(self):
        """Handle incoming WebSocket messages."""
        try:
            while self._connected:
                try:
                    message = await asyncio.wait_for(self._websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    await self._handle_message(data)
                except asyncio.TimeoutError:
                    self.reset_watchdog()
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
                    "",  # participant
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
                    "",  # participant
                    time_now_iso8601(),
                    self._language,
                    message,
                )
            )
