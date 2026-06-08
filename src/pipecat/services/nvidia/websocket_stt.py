#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Nemotron streaming Speech-to-Text service over a raw WebSocket.

This connects to the self-hosted Tone STT server (NeMo + FastAPI) that wraps
``nvidia/nemotron-speech-streaming-en-0.6b`` and exposes a ``/ws/asr`` endpoint.

Wire protocol (see ``nemotron-streaming/app/server.py``):

* Client sends raw 16 kHz mono **PCM16-LE binary** frames.
* Server emits JSON ``{"type": "partial", "text": ..., "ms": ..., "ttf": bool}``
  as audio accumulates, and ``{"type": "final", "text": ..., "ms": ...}`` after
  the client sends the text frame ``"eof"``. On error it emits
  ``{"type": "error", "detail": ...}``.
* After sending the final, the server closes the socket. Reconnection for the
  next utterance is handled by :class:`WebsocketService`.

``partial`` maps to :class:`InterimTranscriptionFrame`; ``final`` maps to
:class:`TranscriptionFrame`.
"""

import json
from typing import AsyncGenerator, Optional

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import websockets
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the NVIDIA Nemotron websocket STT, you need to `pip install websockets`."
    )
    raise Exception(f"Missing module: {e}")


class NvidiaWebSocketService(WebsocketSTTService):
    """Streaming STT over the Tone Nemotron ``/ws/asr`` WebSocket.

    Sends raw PCM16-LE audio to the self-hosted Nemotron streaming server and
    surfaces partial/final transcripts as pipecat frames. Provides automatic
    reconnection and error handling via :class:`WebsocketService`.

    Note:
        The current v1 server finalizes (and closes) only after receiving an
        ``"eof"`` text frame, which this service sends on stop. For per-utterance
        finals in a continuous pipeline, drive finalization from VAD or upgrade
        the server to emit incremental finals.
    """

    def __init__(
        self,
        *,
        url: str,
        sample_rate: Optional[int] = 16000,
        language: Language = Language.EN,
        **kwargs,
    ):
        """Initialize the NVIDIA Nemotron websocket STT service.

        Args:
            url: WebSocket URL of the Nemotron STT server, e.g.
                ``ws://stt-host:8000/ws/asr`` (or ``wss://`` behind TLS).
            sample_rate: Audio sample rate in Hz. The server expects 16 kHz.
            language: Transcript language tag attached to emitted frames.
            **kwargs: Additional arguments passed to the parent classes.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._url = url
        self._language = language
        self.set_model_name("nvidia/nemotron-speech-streaming-en-0.6b")

        self._receive_task = None
        # Tracks the latest partial so we can fall back to it if the server
        # never emits an explicit final.
        self._last_partial: str = ""

    def __str__(self):
        return f"{self.name}"

    def can_generate_metrics(self) -> bool:
        """Whether this service supports metrics generation.

        Returns:
            True, indicating processing metrics are emitted.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the STT service and open the websocket connection.

        Args:
            frame: The start frame triggering service startup.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Finalize, then stop the STT service and close the connection.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        await self._send_eof()
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close the connection.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Stream audio to the Nemotron server.

        Args:
            audio: Raw PCM16-LE audio bytes at the configured sample rate.

        Yields:
            None. Transcripts are pushed asynchronously from the receive task.
        """
        await self.start_processing_metrics()

        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"{self} websocket closed while sending audio: {e}")

        yield None

    async def _connect(self):
        """Connect to the Nemotron server and start the receive task."""
        await super()._connect()
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Cancel the receive task and close the websocket connection."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Open the websocket connection to the Nemotron server."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"{self} connecting to Nemotron WebSocket {self._url}")
            self._websocket = await websocket_connect(self._url)
            self._last_partial = ""
            await self._call_event_handler("on_connected")
            logger.debug(f"{self} connected to Nemotron WebSocket")
        except Exception as e:
            await self.push_error(error_msg=f"Unable to connect to Nemotron STT: {e}", exception=e)
            raise

    async def _disconnect_websocket(self):
        """Close the websocket connection to the Nemotron server."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug(f"{self} disconnecting from Nemotron WebSocket")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _send_eof(self):
        """Signal end-of-stream so the server emits a final transcript."""
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send("eof")
            except websockets.exceptions.ConnectionClosed:
                pass

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        await self.stop_processing_metrics()

    async def _receive_messages(self):
        """Receive and dispatch transcription messages from the server."""
        async for message in self._get_websocket():
            try:
                content = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"{self} received non-JSON message: {message}")
                continue

            msg_type = content.get("type")
            text = content.get("text", "")

            if msg_type == "partial":
                self._last_partial = text
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        self._language,
                        result=content,
                    )
                )
            elif msg_type == "final":
                final_text = text or self._last_partial
                self._last_partial = ""
                await self.push_frame(
                    TranscriptionFrame(
                        final_text,
                        self._user_id,
                        time_now_iso8601(),
                        self._language,
                        result=content,
                    )
                )
                await self._handle_transcription(
                    transcript=final_text, is_final=True, language=self._language
                )
            elif msg_type == "error":
                await self.push_error(
                    error_msg=f"Nemotron STT error: {content.get('detail')}"
                )
