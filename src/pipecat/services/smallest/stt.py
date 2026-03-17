#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smallest AI speech-to-text service implementation.

This module provides a WebSocket-based real-time STT service using Smallest
AI's Waves API (Pulse model). Audio is streamed continuously over a WebSocket
connection and interim/final transcription results are received with low
latency.
"""

import asyncio
import json
from typing import AsyncGenerator, Optional
from urllib.parse import urlencode

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import SMALLEST_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")


class SmallestSTTService(WebsocketSTTService):
    """Smallest AI real-time speech-to-text service using the Pulse WebSocket API.

    Streams audio continuously over a WebSocket connection and receives
    interim and final transcription results with low latency. Best suited
    for real-time voice applications where immediate feedback is needed.

    Uses Pipecat's VAD to detect when the user stops speaking and sends
    a finalize message to flush the final transcript.

    Example::

        stt = SmallestSTTService(
            api_key="your-api-key",
            params=SmallestSTTService.InputParams(
                language="en",
                word_timestamps=True,
            ),
        )
    """

    class InputParams(BaseModel):
        """Configuration parameters for Smallest STT service.

        Parameters:
            language: Language code for transcription. Use "multi" for auto-detection.
                Defaults to "en".
            encoding: Audio encoding format. Defaults to "linear16".
            word_timestamps: Include word-level timestamps. Defaults to False.
            full_transcript: Include cumulative transcript. Defaults to False.
            sentence_timestamps: Include sentence-level timestamps. Defaults to False.
            redact_pii: Redact personally identifiable information. Defaults to False.
            redact_pci: Redact payment card information. Defaults to False.
            numerals: Convert spoken numerals to digits. Defaults to "auto".
            diarize: Enable speaker diarization. Defaults to False.
        """

        language: str = "en"
        encoding: str = "linear16"
        word_timestamps: bool = False
        full_transcript: bool = False
        sentence_timestamps: bool = False
        redact_pii: bool = False
        redact_pci: bool = False
        numerals: str = "auto"
        diarize: bool = False

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.smallest.ai",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        ttfs_p99_latency: Optional[float] = SMALLEST_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Smallest AI STT service.

        Args:
            api_key: Smallest AI API key for authentication.
            base_url: Base WebSocket URL for the Smallest API.
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline's rate.
            params: Configuration parameters for the STT service.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        self._stt_params = params or SmallestSTTService.InputParams()

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=10,
            keepalive_interval=5,
            settings=STTSettings(model="pulse", language=self._stt_params.language),
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._receive_task = None
        self._connected_event = asyncio.Event()
        self._connected_event.set()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service and connect to the WebSocket."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and disconnect from the WebSocket."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and disconnect from the WebSocket."""
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, handling VAD events for finalization."""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._websocket and self._websocket.state is State.OPEN:
                try:
                    await self._websocket.send(json.dumps({"type": "finalize"}))
                except Exception as e:
                    logger.warning(f"{self} failed to send finalize: {e}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio to the Smallest Pulse WebSocket for transcription.

        Args:
            audio: Raw PCM audio bytes.

        Yields:
            None -- transcription results arrive via WebSocket messages.
        """
        await self._connected_event.wait()

        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                yield ErrorFrame(error=f"Smallest STT error: {e}")

        yield None

    async def _connect(self):
        self._connected_event.clear()
        try:
            await self._connect_websocket()
            await super()._connect()

            if self._websocket and not self._receive_task:
                self._receive_task = self.create_task(
                    self._receive_task_handler(self._report_error)
                )
        finally:
            self._connected_event.set()

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to the Smallest Pulse STT API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Smallest STT")

            query_params = {
                "language": self._stt_params.language,
                "encoding": self._stt_params.encoding,
                "sample_rate": str(self.sample_rate),
                "word_timestamps": str(self._stt_params.word_timestamps).lower(),
                "full_transcript": str(self._stt_params.full_transcript).lower(),
                "sentence_timestamps": str(self._stt_params.sentence_timestamps).lower(),
                "redact_pii": str(self._stt_params.redact_pii).lower(),
                "redact_pci": str(self._stt_params.redact_pci).lower(),
                "numerals": self._stt_params.numerals,
                "diarize": str(self._stt_params.diarize).lower(),
            }

            ws_url = f"{self._base_url}/waves/v1/pulse/get_text?{urlencode(query_params)}"

            self._websocket = await websocket_connect(
                ws_url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )
            await self._call_event_handler("on_connected")
            logger.debug("Connected to Smallest STT")
        except Exception as e:
            await self.push_error(error_msg=f"Smallest STT connection error: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket connection."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Smallest STT")
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

    async def _receive_messages(self):
        """Receive and process messages from the Smallest Pulse WebSocket."""
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON message: {message}")
            except Exception as e:
                logger.error(f"{self} error processing message: {e}")

    async def _process_response(self, data: dict):
        """Process a transcription response from the Pulse API.

        Args:
            data: Parsed JSON response containing transcript data.
        """
        is_final = data.get("is_final", False)
        text = data.get("transcript", "").strip()

        if not text:
            return

        if is_final:
            await self.stop_processing_metrics()
            logger.debug(f"Smallest final transcript: [{text}]")
            await self._handle_transcription(text, True, data.get("language"))
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    data.get("language"),
                    result=data,
                )
            )
        else:
            logger.trace(f"Smallest interim transcript: [{text}]")
            await self.push_frame(
                InterimTranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    data.get("language"),
                    result=data,
                )
            )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing."""
        pass
