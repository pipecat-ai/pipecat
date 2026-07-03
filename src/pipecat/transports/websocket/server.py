#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket server transport implementation for Pipecat.

This module provides WebSocket server transport functionality for real-time
audio and data streaming, including client connection management, session
handling, and frame serialization.
"""

import asyncio
import io
import time
import wave
from collections.abc import Awaitable, Callable

import websockets
from loguru import logger
from pydantic import BaseModel, Field
from websockets.asyncio.server import serve as websocket_serve
from websockets.protocol import State

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
from pipecat.utils.deprecation import deprecated
from pipecat.utils.security.allowed_origins import default_allowed_origins


class SingleClientWebsocketServerParams(TransportParams):
    """Configuration parameters for :class:`SingleClientWebsocketServerTransport`.

    Parameters:
        add_wav_header: Whether to add WAV headers to audio frames.
        serializer: Frame serializer for message encoding/decoding.
        session_timeout: Timeout in seconds for client sessions.
        allowed_origins: List of allowed origins. Empty list allows all
            origins. When set, connections with a missing or disallowed Origin header
            are rejected. Defaults to ``PIPECAT_ALLOWED_ORIGINS`` env var
            (comma-separated).
    """

    add_wav_header: bool = False
    serializer: FrameSerializer | None = None
    session_timeout: int | None = None
    allowed_origins: list[str] = Field(default_factory=default_allowed_origins)


class SingleClientWebsocketServerCallbacks(BaseModel):
    """Callback functions for WebSocket server events.

    Parameters:
        on_client_connected: Called when a client connects to the server.
        on_client_disconnected: Called when a client disconnects from the server.
        on_session_timeout: Called when a client session times out.
        on_websocket_ready: Called when the WebSocket server is ready to accept connections.
    """

    on_client_connected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_client_disconnected: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_session_timeout: Callable[[websockets.WebSocketServerProtocol], Awaitable[None]]
    on_websocket_ready: Callable[[], Awaitable[None]]


class SingleClientWebsocketServerInputTransport(BaseInputTransport):
    """WebSocket server input transport for receiving client data.

    Handles incoming WebSocket connections, message processing, and client
    session management including timeout monitoring and connection lifecycle.
    """

    def __init__(
        self,
        transport: BaseTransport,
        host: str,
        port: int,
        params: SingleClientWebsocketServerParams,
        callbacks: SingleClientWebsocketServerCallbacks,
        **kwargs,
    ):
        """Initialize the WebSocket server input transport.

        Args:
            transport: The parent transport instance.
            host: Host address to bind the WebSocket server to.
            port: Port number to bind the WebSocket server to.
            params: WebSocket server configuration parameters.
            callbacks: Callback functions for WebSocket events.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._host = host
        self._port = port
        self._params = params
        self._callbacks = callbacks

        self._websocket: websockets.WebSocketServerProtocol | None = None

        self._server_task = None

        # This task will monitor the websocket connection periodically.
        self._monitor_task = None

        self._stop_server_event = asyncio.Event()

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the WebSocket server and initialize components.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.serializer:
            await self._params.serializer.setup(frame)
        if not self._server_task:
            self._server_task = self.create_task(self._server_task_handler())
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the WebSocket server and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        # Signal the server loop to exit and drain it gracefully before
        # cancelling whatever is left.
        self._stop_server_event.set()
        if self._server_task:
            await self._server_task
            self._server_task = None
        await self._stop_tasks()

    async def cancel(self, frame: CancelFrame):
        """Cancel the WebSocket server and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._stop_tasks()

    async def cleanup(self):
        """Release input transport resources at teardown."""
        await super().cleanup()
        await self._stop_tasks()
        await self._transport.cleanup()

    async def _stop_tasks(self):
        """Cancel the server and monitor tasks. Idempotent."""
        if self._monitor_task:
            await self.cancel_task(self._monitor_task)
            self._monitor_task = None
        if self._server_task:
            await self.cancel_task(self._server_task)
            self._server_task = None

    async def _server_task_handler(self):
        """Handle WebSocket server startup and client connections."""
        logger.info(f"Starting websocket server on {self._host}:{self._port}")
        origins = self._params.allowed_origins or None
        async with websocket_serve(
            self._client_handler, self._host, self._port, origins=origins
        ) as server:
            await self._callbacks.on_websocket_ready()
            await self._stop_server_event.wait()

    async def _client_handler(self, websocket: websockets.WebSocketServerProtocol):
        """Handle individual client connections and message processing."""
        logger.info(f"New client connection from {websocket.remote_address}")

        # This transport only serves a single client at a time. If we already
        # have a live connection, reject the new one and keep the existing
        # client. The current connection's reference is cleared when it
        # disconnects (or something goes wrong), so the next client can connect.
        if self._websocket and self._websocket.state is State.OPEN:
            logger.warning(
                f"Rejecting client {websocket.remote_address}: a client is already connected"
            )
            await websocket.close(code=1013, reason="Server already has a connected client")
            return

        self._websocket = websocket

        # Notify
        await self._callbacks.on_client_connected(websocket)

        # Create a task to monitor the websocket connection
        if not self._monitor_task and self._params.session_timeout:
            self._monitor_task = self.create_task(
                self._monitor_websocket(websocket, self._params.session_timeout)
            )

        # Handle incoming messages
        try:
            async for message in websocket:
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
        except websockets.ConnectionClosed:
            # The client closed the connection (clean or abrupt). Normal end of a
            # session, not an error.
            logger.debug(f"{self}: client disconnected while receiving")
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        # Notify disconnection
        await self._callbacks.on_client_disconnected(websocket)

        await websocket.close()
        # Only clear if it's still ours: the next client may have already
        # connected and replaced it (e.g. back-to-back evals on a kept-alive
        # server), and we must not null out their connection.
        if self._websocket is websocket:
            self._websocket = None

        logger.info(f"Client {websocket.remote_address} disconnected")

    async def _monitor_websocket(
        self, websocket: websockets.WebSocketServerProtocol, session_timeout: int
    ):
        """Monitor WebSocket connection for session timeout."""
        try:
            await asyncio.sleep(session_timeout)
            if websocket.state is not State.CLOSED:
                await self._callbacks.on_session_timeout(websocket)
        except asyncio.CancelledError:
            logger.info(f"Monitoring task cancelled for: {websocket.remote_address}")
            raise


class SingleClientWebsocketServerOutputTransport(BaseOutputTransport):
    """WebSocket server output transport for sending data to clients.

    Handles outgoing frame serialization, audio streaming with timing control,
    and client connection management for WebSocket communication.
    """

    def __init__(
        self, transport: BaseTransport, params: SingleClientWebsocketServerParams, **kwargs
    ):
        """Initialize the WebSocket server output transport.

        Args:
            transport: The parent transport instance.
            params: WebSocket server configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._params = params

        self._websocket: websockets.WebSocketServerProtocol | None = None

        # write_audio_frame() is called quickly, as soon as we get audio
        # (e.g. from the TTS), and since this is just a network connection we
        # would be sending it to quickly. Instead, we want to block to emulate
        # an audio device, this is what the send interval is. It will be
        # computed on StartFrame.
        self._send_interval = 0
        self._next_send_time = 0

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def set_client_connection(self, websocket: websockets.WebSocketServerProtocol | None):
        """Set the active client WebSocket connection.

        Args:
            websocket: The WebSocket connection to set as active, or None to clear.
        """
        # The input transport gates new connections, so by the time we set a new
        # client here the previous one (if any) is already gone. Close any stale
        # reference just in case before tracking the new connection.
        if self._websocket and self._websocket is not websocket:
            await self._websocket.close()
        self._websocket = websocket

    async def start(self, frame: StartFrame):
        """Start the output transport and initialize components.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        if self._params.serializer:
            await self._params.serializer.setup(frame)
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport and send final frame.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._write_frame(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and send cancellation frame.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._write_frame(frame)

    async def cleanup(self):
        """Cleanup resources and parent transport."""
        await super().cleanup()
        await self._transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle interruption timing.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._write_frame(frame)
            self._next_send_time = 0

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message frame to the client.

        Args:
            frame: The transport message frame to send.
        """
        await self._write_frame(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the WebSocket client with timing control.

        Args:
            frame: The output audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        if not self._websocket:
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
        """Serialize and send a frame to the WebSocket client."""
        if not self._params.serializer:
            return

        try:
            payload = await self._params.serializer.serialize(frame)
            if payload and self._websocket:
                await self._websocket.send(payload)
        except websockets.ConnectionClosed:
            # The client went away mid-send (a normal race on disconnect, e.g.
            # while still streaming TTS audio). Not an error.
            logger.debug(f"{self}: client disconnected while sending")
        except Exception as e:
            logger.error(f"{self} exception sending data: {e.__class__.__name__} ({e})")

    async def _write_audio_sleep(self):
        """Simulate audio device timing by sleeping between audio chunks."""
        # Simulate a clock.
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class SingleClientWebsocketServerTransport(BaseTransport):
    """WebSocket server transport that serves a single client at a time.

    Provides a complete WebSocket server implementation with separate input and
    output transports, client connection management, and event handling for
    real-time audio and data streaming applications.

    Only one client can be connected at a time. While a client is connected, new
    connection attempts are rejected and the existing client is kept; once that
    client disconnects, the server accepts a new one. This makes it well suited
    for local development and single-session bots, but not for serving multiple
    concurrent clients.

    Event handlers available:

    - on_client_connected(transport, websocket): Client WebSocket connected
    - on_client_disconnected(transport, websocket): Client WebSocket disconnected
    - on_session_timeout(transport, websocket): Session timed out
    - on_websocket_ready(transport): WebSocket server is ready to accept connections

    Example::

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, websocket):
            ...
    """

    def __init__(
        self,
        params: SingleClientWebsocketServerParams,
        host: str = "localhost",
        port: int = 8765,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the WebSocket server transport.

        Args:
            params: WebSocket server configuration parameters.
            host: Host address to bind the server to. Defaults to "localhost".
            port: Port number to bind the server to. Defaults to 8765.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._host = host
        self._port = port
        self._params = params

        self._callbacks = SingleClientWebsocketServerCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
            on_websocket_ready=self._on_websocket_ready,
        )
        self._input: SingleClientWebsocketServerInputTransport | None = None
        self._output: SingleClientWebsocketServerOutputTransport | None = None
        self._websocket: websockets.WebSocketServerProtocol | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")
        self._register_event_handler("on_websocket_ready")

    def input(self) -> SingleClientWebsocketServerInputTransport:
        """Get the input transport for receiving client data.

        Returns:
            The WebSocket server input transport instance.
        """
        if not self._input:
            self._input = SingleClientWebsocketServerInputTransport(
                self, self._host, self._port, self._params, self._callbacks, name=self._input_name
            )
        return self._input

    def output(self) -> SingleClientWebsocketServerOutputTransport:
        """Get the output transport for sending data to clients.

        Returns:
            The WebSocket server output transport instance.
        """
        if not self._output:
            self._output = SingleClientWebsocketServerOutputTransport(
                self, self._params, name=self._output_name
            )
        return self._output

    async def _on_client_connected(self, websocket):
        """Handle client connection events."""
        if self._output:
            await self._output.set_client_connection(websocket)
            await self._call_event_handler("on_client_connected", websocket)
            if self._input:
                await self._input.push_frame(ClientConnectedFrame())
        else:
            logger.error("A SingleClientWebsocketServerTransport output is missing in the pipeline")

    async def _on_client_disconnected(self, websocket):
        """Handle client disconnection events."""
        if self._output:
            await self._output.set_client_connection(None)
            await self._emit_client_disconnected(websocket)
        else:
            logger.error("A SingleClientWebsocketServerTransport output is missing in the pipeline")

    async def _emit_client_disconnected(self, websocket):
        """Fire the ``on_client_disconnected`` event.

        Split from the connection teardown above so subclasses can suppress the
        event without skipping that teardown.
        """
        await self._call_event_handler("on_client_disconnected", websocket)

    async def _on_session_timeout(self, websocket):
        """Handle client session timeout events."""
        await self._call_event_handler("on_session_timeout", websocket)

    async def _on_websocket_ready(self):
        """Handle WebSocket server ready events."""
        await self._call_event_handler("on_websocket_ready")


@deprecated(
    "`WebsocketServerParams` is deprecated since 1.4.0 and will be removed in 2.0.0. "
    "Use `SingleClientWebsocketServerParams` instead."
)
class WebsocketServerParams(SingleClientWebsocketServerParams):
    """Deprecated alias for :class:`SingleClientWebsocketServerParams`.

    .. deprecated:: 1.4.0
        Use :class:`SingleClientWebsocketServerParams` instead. Will be removed
        in 2.0.0.
    """

    pass


@deprecated(
    "`WebsocketServerCallbacks` is deprecated since 1.4.0 and will be removed in 2.0.0. "
    "Use `SingleClientWebsocketServerCallbacks` instead."
)
class WebsocketServerCallbacks(SingleClientWebsocketServerCallbacks):
    """Deprecated alias for :class:`SingleClientWebsocketServerCallbacks`.

    .. deprecated:: 1.4.0
        Use :class:`SingleClientWebsocketServerCallbacks` instead. Will be
        removed in 2.0.0.
    """

    pass


@deprecated(
    "`WebsocketServerInputTransport` is deprecated since 1.4.0 and will be removed in 2.0.0. "
    "Use `SingleClientWebsocketServerInputTransport` instead."
)
class WebsocketServerInputTransport(SingleClientWebsocketServerInputTransport):
    """Deprecated alias for :class:`SingleClientWebsocketServerInputTransport`.

    .. deprecated:: 1.4.0
        Use :class:`SingleClientWebsocketServerInputTransport` instead. Will be
        removed in 2.0.0.
    """

    pass


@deprecated(
    "`WebsocketServerOutputTransport` is deprecated since 1.4.0 and will be removed in 2.0.0. "
    "Use `SingleClientWebsocketServerOutputTransport` instead."
)
class WebsocketServerOutputTransport(SingleClientWebsocketServerOutputTransport):
    """Deprecated alias for :class:`SingleClientWebsocketServerOutputTransport`.

    .. deprecated:: 1.4.0
        Use :class:`SingleClientWebsocketServerOutputTransport` instead. Will be
        removed in 2.0.0.
    """

    pass


@deprecated(
    "`WebsocketServerTransport` is deprecated since 1.4.0 and will be removed in 2.0.0. "
    "Use `SingleClientWebsocketServerTransport` instead."
)
class WebsocketServerTransport(SingleClientWebsocketServerTransport):
    """Deprecated alias for :class:`SingleClientWebsocketServerTransport`.

    .. deprecated:: 1.4.0
        Use :class:`SingleClientWebsocketServerTransport` instead. The renamed
        class makes it explicit that the server handles a single client at a
        time. Will be removed in 2.0.0.
    """

    pass
