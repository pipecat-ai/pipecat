#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""FastAPI WebSocket transport implementation for Pipecat.

This module provides WebSocket-based transport for real-time audio/video streaming
using FastAPI and WebSocket connections. Supports binary and text serialization
with configurable session timeouts and WAV header generation.
"""

import asyncio
import io
import time
import typing
import wave
from collections.abc import Awaitable, Callable

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.enums import EndTaskReason
from pipecat.utils.security.allowed_origins import default_allowed_origins, is_origin_allowed

try:
    from fastapi import WebSocket
    from starlette.websockets import WebSocketState
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use FastAPI websockets, you need to `uv add "pipecat-ai[websocket]"`.'
    )
    raise ImportError(f"Missing module: {e}") from e


# Default time (in seconds) to wait for the WebSocket close handshake in
# ``FastAPIWebsocketClient.disconnect()`` before proceeding with shutdown.
_WS_CLOSE_TIMEOUT_DEFAULT = 0.5


class FastAPIWebsocketParams(TransportParams):
    """Configuration parameters for FastAPI WebSocket transport.

    Parameters:
        add_wav_header: Whether to add WAV headers to audio frames.
        serializer: Frame serializer for encoding/decoding messages.
        session_timeout: Session timeout in seconds, None for no timeout.
        fixed_audio_packet_size: Optional fixed-size packetization for raw PCM audio payloads.
            Useful when the remote WebSocket media endpoint requires strict audio framing.
        allowed_origins: List of allowed origins. Empty list allows all
            origins. When set, connections with a missing or disallowed Origin header
            are rejected. Defaults to ``PIPECAT_ALLOWED_ORIGINS`` env var
            (comma-separated).
        ws_close_timeout: Maximum time, in seconds, to wait for the WebSocket
            close handshake during disconnect. The close is initiated in a
            background task before we start waiting, so the close frame is sent
            to the peer in the common case; this only bounds how long we wait
            for the peer to acknowledge it before letting shutdown proceed.
            Prevents a dead or half-closed peer (e.g. a telephony call already
            torn down on the provider's side) from stalling pipeline shutdown on
            the ASGI server's close-handshake timeout. Increase it for
            high-latency peers that need longer to complete a graceful close.
    """

    add_wav_header: bool = False
    serializer: FrameSerializer | None = None
    session_timeout: int | None = None
    fixed_audio_packet_size: int | None = None
    allowed_origins: list[str] = Field(default_factory=default_allowed_origins)
    ws_close_timeout: float = _WS_CLOSE_TIMEOUT_DEFAULT


class FastAPIWebsocketCallbacks(BaseModel):
    """Callback functions for WebSocket events.

    Parameters:
        on_client_connected: Called when a client connects to the WebSocket.
        on_client_disconnected: Called when a client disconnects from the WebSocket.
        on_session_timeout: Called when a session timeout occurs.
    """

    on_client_connected: Callable[[WebSocket], Awaitable[None]]
    on_client_disconnected: Callable[[WebSocket], Awaitable[None]]
    on_session_timeout: Callable[[WebSocket], Awaitable[None]]


class _WebSocketMessageIterator:
    """Async iterator for WebSocket messages that yields both binary and text."""

    def __init__(self, websocket: WebSocket):
        self._websocket = websocket

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes | str:
        message = await self._websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise StopAsyncIteration
        if "bytes" in message and message["bytes"] is not None:
            return message["bytes"]
        if "text" in message and message["text"] is not None:
            return message["text"]
        raise StopAsyncIteration


class FastAPIWebsocketClient:
    """WebSocket client wrapper for handling connections and message passing.

    Manages WebSocket state, message sending/receiving, and connection lifecycle
    with support for both binary and text message types.
    """

    def __init__(
        self,
        websocket: WebSocket,
        callbacks: FastAPIWebsocketCallbacks,
        ws_close_timeout: float = _WS_CLOSE_TIMEOUT_DEFAULT,
    ):
        """Initialize the WebSocket client.

        Args:
            websocket: The FastAPI WebSocket connection.
            callbacks: Event callback functions.
            ws_close_timeout: Maximum time, in seconds, to wait for the close
                handshake in ``disconnect()`` before proceeding.
        """
        self._websocket = websocket
        self._closing = False
        self._callbacks = callbacks
        self._leave_counter = 0
        self._transfer_in_progress = False
        self._ws_close_timeout = ws_close_timeout
        self._close_task: asyncio.Task | None = None

    async def setup(self, _: StartFrame):
        """Set up the WebSocket client.

        Args:
            _: The start frame (unused).
        """
        self._leave_counter += 1

    def receive(self) -> typing.AsyncIterator[bytes | str]:
        """Get an async iterator for receiving WebSocket messages.

        Returns:
            An async iterator yielding bytes or strings.
        """
        return _WebSocketMessageIterator(self._websocket)

    async def send(self, data: str | bytes):
        """Send data through the WebSocket connection.

        Args:
            data: The data to send (string or bytes).
        """
        try:
            if self._can_send():
                if isinstance(data, bytes):
                    await self._websocket.send_bytes(data)
                else:
                    await self._websocket.send_text(data)
        except Exception as e:
            logger.warning(
                f"{self} exception sending data: {e.__class__.__name__} ({e}), application_state: {self._websocket.application_state}"
            )

    async def disconnect(self):
        """Disconnect the WebSocket client.

        The close handshake is bounded by ``ws_close_timeout``. The close is
        initiated in a background task before we start waiting, so the close
        frame is sent to the peer in the common case; we then wait at most
        ``ws_close_timeout`` seconds for the peer to acknowledge it. If the peer
        never replies (e.g. a half-closed connection after the remote side
        already hung up), we stop waiting and let shutdown proceed instead of
        blocking on the ASGI server's close-handshake timeout. The close task is
        left running and its eventual result is logged, since the underlying
        ``close()`` may not respond to cancellation.
        """
        self._leave_counter -= 1
        if self._leave_counter > 0:
            return

        if self.is_connected and not self.is_closing:
            self._closing = True
            self._close_task = asyncio.create_task(self._websocket.close(), name="fastapi-ws-close")
            self._close_task.add_done_callback(self._on_close_done)
            done, _ = await asyncio.wait({self._close_task}, timeout=self._ws_close_timeout)
            if not done:
                logger.debug(
                    f"{self} WebSocket close exceeded {self._ws_close_timeout}s; "
                    "proceeding with shutdown"
                )

    def _on_close_done(self, task: asyncio.Task):
        """Log the outcome of the WebSocket close task.

        Runs whether the close completed within ``ws_close_timeout`` or
        afterwards, so an error raised by a slow close is still surfaced.
        """
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"{self} exception while closing the websocket: {e}")

    async def trigger_client_disconnected(self):
        """Trigger the client disconnected callback."""
        await self._callbacks.on_client_disconnected(self._websocket)

    async def trigger_client_connected(self):
        """Trigger the client connected callback."""
        await self._callbacks.on_client_connected(self._websocket)

    async def trigger_client_timeout(self):
        """Trigger the client timeout callback."""
        await self._callbacks.on_session_timeout(self._websocket)

    def _can_send(self):
        """Check if data can be sent through the WebSocket."""
        return (
            self.is_connected
            and not self.is_closing
            and self._websocket.application_state != WebSocketState.DISCONNECTED
        )

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket is currently connected.

        Returns:
            True if the WebSocket is in connected state.
        """
        return self._websocket.client_state == WebSocketState.CONNECTED

    @property
    def is_closing(self) -> bool:
        """Check if the WebSocket is currently closing.

        Returns:
            True if the WebSocket is in the process of closing.
        """
        return self._closing


class FastAPIWebsocketInputTransport(BaseInputTransport):
    """Input transport for FastAPI WebSocket connections.

    Handles incoming WebSocket messages, deserializes frames, and manages
    connection monitoring with optional session timeouts.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: FastAPIWebsocketClient,
        params: FastAPIWebsocketParams,
        **kwargs,
    ):
        """Initialize the WebSocket input transport.

        Args:
            transport: The parent transport instance.
            client: The WebSocket client wrapper.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._params = params
        self._receive_task = None
        self._monitor_websocket_task = None

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the input transport and begin message processing.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.setup(frame)
        if self._params.serializer:
            await self._params.serializer.setup(frame)
        if not self._monitor_websocket_task and self._params.session_timeout:
            self._monitor_websocket_task = self.create_task(
                self._monitor_websocket(self._params.session_timeout)
            )
        await self._client.trigger_client_connected()
        await self.push_frame(ClientConnectedFrame())
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_messages())
        await self.set_transport_ready(frame)

    async def _stop_tasks(self):
        """Stop all running tasks."""
        if self._monitor_websocket_task:
            await self.cancel_task(self._monitor_websocket_task)
            self._monitor_websocket_task = None
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def stop(self, frame: EndFrame):
        """Stop the input transport and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cleanup(self):
        """Clean up transport resources."""
        await super().cleanup()
        await self._transport.cleanup()

    async def _receive_messages(self):
        """Main message receiving loop for WebSocket messages."""
        try:
            async for message in self._client.receive():
                if not self._params.serializer:
                    continue

                frame = await self._params.serializer.deserialize(message)

                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    await self.push_audio_frame(frame)
                elif isinstance(frame, InputTransportMessageFrame):
                    await self.broadcast_frame(InputTransportMessageFrame, message=frame.message)
                else:
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        # Trigger `on_client_disconnected` if the client actually disconnects,
        # that is, we are not the ones disconnecting.
        if not self._client.is_closing and not self._client._transfer_in_progress:
            await self._client.trigger_client_disconnected()

    async def _monitor_websocket(self, timeout: int):
        """Wait for ``timeout`` seconds, then trigger the client-timeout event if still open."""
        await asyncio.sleep(timeout)
        await self._client.trigger_client_timeout()


class FastAPIWebsocketOutputTransport(BaseOutputTransport):
    """Output transport for FastAPI WebSocket connections.

    Handles outgoing frame serialization, audio streaming with timing simulation,
    and WebSocket message transmission with optional WAV header generation.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: FastAPIWebsocketClient,
        params: FastAPIWebsocketParams,
        **kwargs,
    ):
        """Initialize the WebSocket output transport.

        Args:
            transport: The parent transport instance.
            client: The WebSocket client wrapper.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._client = client
        self._params = params

        # write_audio_frame() is called quickly, as soon as we get audio
        # (e.g. from the TTS), and since this is just a network connection we
        # would be sending it to quickly. Instead, we want to block to emulate
        # an audio device, this is what the send interval is. It will be
        # computed on StartFrame.
        self._send_interval = 0
        self._next_send_time = 0

        # Buffer for optional protocol-level audio packetization.
        # Some serializers may emit arbitrarily sized raw PCM payloads, while
        # certain downstream transports or media endpoints require audio to be
        # sent in fixed-size frames. When `params.fixed_audio_packet_size` is set,
        # this buffer accumulates outgoing audio until a full packet can be
        # emitted, preserving any remainder for subsequent sends.
        self._audio_send_buffer = bytearray()

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the output transport and initialize timing.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.setup(frame)
        if self._params.serializer:
            await self._params.serializer.setup(frame)
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)

        # Transfer ends the current media WebSocket intentionally; don't report
        # it as a user hangup through on_client_disconnected.
        if isinstance(frame, EndFrame) and frame.reason == EndTaskReason.TRANSFER_CALL.value:
            self._client._transfer_in_progress = True

        await self._write_frame(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._write_frame(frame)
        await self._client.disconnect()

    async def cleanup(self):
        """Clean up transport resources."""
        await super().cleanup()
        await self._transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process outgoing frames with special handling for interruptions.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            # Drop any partially buffered audio to avoid replaying stale PCM
            if self._params.fixed_audio_packet_size:
                self._audio_send_buffer.clear()

            await self._write_frame(frame)
            self._next_send_time = 0

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message frame.

        Args:
            frame: The transport message frame to send.
        """
        await self._write_frame(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the WebSocket with timing simulation.

        Args:
            frame: The output audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        if self._client.is_closing or not self._client.is_connected:
            return False

        frame = OutputAudioRawFrame(
            audio=frame.audio,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        if self._params.add_wav_header:
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(frame.num_channels)
                    wf.setframerate(frame.sample_rate)
                    wf.writeframes(frame.audio)
                wav_frame = OutputAudioRawFrame(
                    buffer.getvalue(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
                frame = wav_frame

        await self._write_frame(frame)

        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

        return True

    async def _write_frame(self, frame: Frame):
        """Serialize and send a frame through the WebSocket."""
        if self._client.is_closing or not self._client.is_connected:
            return

        if not self._params.serializer:
            return

        try:
            payload = await self._params.serializer.serialize(frame)
            if payload:
                # Optional protocol-level audio packetization:
                # If a downstream WebSocket media endpoint requires fixed-size PCM frames,
                # configure params.fixed_audio_packet_size (e.g. 640 for 20ms @ 16kHz PCM16 mono).
                packet_bytes = self._params.fixed_audio_packet_size

                if packet_bytes and isinstance(payload, (bytes, bytearray)):
                    self._audio_send_buffer.extend(bytes(payload))

                    # Send only full frames; keep remainder for the next call.
                    while len(self._audio_send_buffer) >= packet_bytes:
                        chunk = bytes(self._audio_send_buffer[:packet_bytes])
                        del self._audio_send_buffer[:packet_bytes]
                        await self._client.send(chunk)
                    return

                await self._client.send(payload)
        except Exception as e:
            logger.error(f"{self} exception sending data: {e.__class__.__name__} ({e})")

    async def _write_audio_sleep(self):
        """Simulate audio playback timing with appropriate delays."""
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class FastAPIWebsocketTransport(BaseTransport):
    """FastAPI WebSocket transport for real-time audio/video streaming.

    Provides bidirectional WebSocket communication with frame serialization,
    session management, and event handling for client connections and timeouts.

    Event handlers available:

    - on_client_connected(transport, websocket): Client WebSocket connected
    - on_client_disconnected(transport, websocket): Client WebSocket disconnected
    - on_session_timeout(transport, websocket): Session timed out

    Example::

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, websocket):
            ...
    """

    def __init__(
        self,
        websocket: WebSocket,
        params: FastAPIWebsocketParams,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the FastAPI WebSocket transport.

        Raises ``ValueError`` if ``params.allowed_origins`` is set and the
        connection's Origin header is missing or not in the allowed list. The
        caller is responsible for closing the WebSocket in that case.

        Args:
            websocket: The FastAPI WebSocket connection.
            params: Transport configuration parameters.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        if params.allowed_origins:
            origin = websocket.headers.get("origin", "")
            if not is_origin_allowed(origin, params.allowed_origins):
                raise ValueError(f"WebSocket connection rejected: origin '{origin}' not allowed")

        super().__init__(input_name=input_name, output_name=output_name)

        self._params = params

        self._callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
        )

        self._client = FastAPIWebsocketClient(
            websocket, self._callbacks, ws_close_timeout=self._params.ws_close_timeout
        )

        self._input = FastAPIWebsocketInputTransport(
            self, self._client, self._params, name=self._input_name
        )
        self._output = FastAPIWebsocketOutputTransport(
            self, self._client, self._params, name=self._output_name
        )

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")

    def input(self) -> FastAPIWebsocketInputTransport:
        """Get the input transport processor.

        Returns:
            The WebSocket input transport instance.
        """
        return self._input

    def output(self) -> FastAPIWebsocketOutputTransport:
        """Get the output transport processor.

        Returns:
            The WebSocket output transport instance.
        """
        return self._output

    async def _on_client_connected(self, websocket):
        """Handle client connected event."""
        await self._call_event_handler("on_client_connected", websocket)

    async def _on_client_disconnected(self, websocket):
        """Handle client disconnected event."""
        await self._call_event_handler("on_client_disconnected", websocket)

    async def _on_session_timeout(self, websocket):
        """Handle session timeout event."""
        await self._call_event_handler("on_session_timeout", websocket)
