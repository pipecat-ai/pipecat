#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Asterisk chan_websocket server transport implementation for Pipecat.

This module provides transport of Asterisk chan_websocket channels (client on Asterisk side, server on Pipecat side).
It implementes audio buffering, flow-controlled audio streaming and and some basic signaling provided by chan_websocket.
"""

import asyncio
import io
import time
import wave
from typing import Awaitable, Callable, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
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
from pipecat.serializers.asterisk import AsteriskWsFrameSerializer
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.server import (
    WebsocketServerCallbacks,
)

try:
    import websockets
    from websockets.asyncio.server import serve as websocket_serve
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use websockets, you need to `pip install pipecat-ai[websocket]`.")
    raise Exception(f"Missing module: {e}")


class AsteriskWSServerParams(TransportParams):
    """Configuration parameters for Asterisk chan_websocket server transport.

    The transport has a local audio buffer to store audio frames before sending them to Asterisk.
    Here we define a number of audio frames to buffer locally `local_audio_buffer_frames`, it's important to know that size of the frame is defined by a TTS audio frame size, not ptime on Asterisk side.
    For example, if TTS produces audio frames of 3200 bytes (16000Hz, 16bit, mono, 20ms) and we set local buffer size to 3000 frames, the local buffer will be able to store up to 3000*3200=9.6MB of audio data.

    We send audio from local buffer to Asterisk as long as we have something in the local buffer and remote Asterisk buffer has enough space.
    In the very beginning of the call we might receiv an audio frame from TTS with 20ms of audio and send it right away to Asterisk,
    but if the next frame from TTS is delayed for more than 20ms Asterisk doesn't have enough audio in its media buffer, it will run out of audio to play and will produce glitches.
    To avoid that, the transport has an `initial_jitter_buffer_ms` that will accumulate a respective number of audio frames before starting to send them to Asterisk.
    But it comes with a trade-off of added latency in the beginning of the call. If you never have glitches at the beginning of the call, you can set it to 0 to have minimal latency.

    Asterisk has its own media buffer on chan_websocket, this transport is desinged to work with this meda buffer enabled.
    Asterisk's media buffer size is 1000 frames (it's 20 seconds of audio when ptime=20ms), by default we use half `max_remote_audio_buffer_frames=500` of it to be safe, we stop sending when we reach this limit

    We resume sending audio from local buffer when remote buffer fill is below 50% by default `remote_audio_buffer_resume_threshold` float in percentage (0.0 - 1.0), to don't abuse asyncio event loop with frequent pause/resume

    Parameters:
        serializer: Frame serializer for message encoding/decoding.
        session_timeout: Timeout in seconds for client sessions.
        local_audio_buffer_frames: Number of audio frames to buffer locally before sending to Asterisk.
        initial_jitter_buffer_ms: Initial jitter buffer size in milliseconds to accumulate audio before sending.
        max_remote_audio_buffer_frames: Maximum number of audio frames allowed in Asterisk's media buffer.
        remote_audio_buffer_resume_threshold: Threshold percentage to resume sending audio to Asterisk.
    """

    local_audio_buffer_frames: Optional[int] = 3000  # in frames
    initial_jitter_buffer_ms: Optional[int] = 80  # in milliseconds
    max_remote_audio_buffer_frames: Optional[int] = 500  # in frames
    remote_audio_buffer_resume_threshold: float = 0.5  # in percentage (0.0 - 1.0)

    port: Optional[int] = 8765
    host: Optional[str] = "localhost"
    serializer: Optional[FrameSerializer] = AsteriskWsFrameSerializer()
    session_timeout: Optional[int] = None


class AsteriskWSServerInputTransport(BaseInputTransport):
    """Asterisk channel WebSocket server input transport for receiving client data from Asterisk.

    Handles incoming WebSocket connections, message processing, and client
    session management including timeout monitoring and connection lifecycle.
    It's almost the same as generic WebSocket server transport, but adapted for Asterisk chan_websocket specifics.
    """

    def __init__(
        self,
        transport: BaseTransport,
        params: AsteriskWSServerParams,
        callbacks: WebsocketServerCallbacks,
        **kwargs,
    ):
        """Initialize the AsteriskWebSocket server input transport.

        Args:
            transport: The parent transport instance.
            params: Asterisk WebSocket server configuration parameters.
            callbacks: Callback functions for WebSocket events.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._params = params
        self._transport = transport
        self._host = self._params.host
        self._port = self._params.port
        self._callbacks = callbacks

        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        self._server_task = None

        # This task implements the session timeout monitoring.
        self._session_timer_task = None

        # Event to signal server shutdown. It keeps the server running until set.
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

    async def _terminate(self, gracefully: bool = False):
        """Terminate the WebSocket server and timeout task.

        Args:
            gracefully: Whether to terminate gracefully, allowing proper connection closure.
        """
        # Cancel the session timer task if it exists.
        if self._session_timer_task:
            await self.cancel_task(self._session_timer_task)
            self._session_timer_task = None

        if gracefully:
            # Will cause the server to exit the context manager and stop. Effectively it will return from _server_task_handler after closing all connections properly.
            self._stop_server_event.set()
            # Wait for the server task to close connections properly. It's needed as exiting the context manager might take some time, it's best to await it.
            if self._server_task:
                await self._server_task
                self._server_task = None
        else:
            # If not gracefully, just cancel the server task. No need to close connections properly.
            if self._server_task:
                await self.cancel_task(self._server_task)
                self._server_task = None

    async def stop(self, frame: EndFrame):
        """Stop the WebSocket server and cleanup resources.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._terminate(self, gracefully=True)

    async def cancel(self, frame: CancelFrame):
        """Cancel the WebSocket server and stop all processing.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._terminate(self)

    async def cleanup(self):
        """Cleanup resources and parent transport."""
        await super().cleanup()
        await self._transport.cleanup()

    async def _server_task_handler(self):
        """Handle WebSocket server startup and client connections."""
        logger.info(f"Starting websocket server on {self._host}:{self._port}")
        async with websocket_serve(self._client_handler, self._host, self._port) as _:
            await self._callbacks.on_websocket_ready()
            await self._stop_server_event.wait()

    async def _client_handler(self, websocket: websockets.WebSocketServerProtocol):
        """Handle individual client connections and message processing."""
        logger.info(f"New client connection from {websocket.remote_address}")
        if self._websocket != None and websocket != None:
            logger.warning("We already have a client connected, ignoring new connection")
            await websocket.close()
            return

        self._websocket = websocket

        # Notify connection
        await self._callbacks.on_client_connected(websocket)

        # Create a timer task if session timeout is set
        if not self._session_timer_task and self._params.session_timeout:
            self._session_timer_task = self.create_task(
                self._session_timer(websocket, self._params.session_timeout)
            )

        # Handle incoming messages
        try:
            async for message in websocket:
                if not self._params.serializer:
                    logger.error(f"{self} no serializer configured, cannot process messages")
                    continue

                frame = await self._params.serializer.deserialize(message)

                if not frame:
                    continue

                if isinstance(frame, InputAudioRawFrame):
                    await self.push_audio_frame(frame)
                else:
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        # Notify disconnection
        await self._callbacks.on_client_disconnected(websocket)

        await self._websocket.close()
        self._websocket = None

        logger.info(f"Client {websocket.remote_address} disconnected")

    async def _session_timer(
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


class AsteriskWSServerOutputTransport(BaseOutputTransport):
    """Asterisk WebSocket server output transport for sending data to clients.

    Handles outgoing frame serialization, audio streaming with buffer and flow control,
    and client connection management for WebSocket communication, support basic channel websocket signaling.
    """

    def __init__(self, transport: BaseTransport, params: AsteriskWSServerParams, **kwargs):
        """Initialize the WebSocket server output transport.

        Args:
            transport: The parent transport instance.
            params: WebSocket server configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)

        self._transport = transport
        self._params = params or AsteriskWSServerParams()
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        # TODO: Implement sending REPORT_QUEUE_DRAINED command and handling QUEUE_DRAINED event.
        # We need to send REPORT_QUEUE_DRAINED every time when we start populating EMPTY remote buffer.
        # Then asteirsk will send us QUEUE_DRAINED event when it finishes playing all the audio in its buffer.
        # So we can understand the precise moment when Asterisk finished playing all the TTS audio (bot stopped speaking from the remote user's perspective).
        # TODO: Implement jitter buffer not only in the very beginning of the call, but also every time after the local buffer sized hit 0 again.(e.g. after interruptions or pauses in bot's speech)
        # TODO: Implement warnings when local buffer is 90% full and errors when it is full.

        # Internal audio buffer is an asyncio Queue for storing serialized binary audio frames ready to be sent to Asterisk.
        self._audio_buffer: asyncio.Queue[OutputAudioRawFrame] = asyncio.Queue(
            maxsize=self._params.local_audio_buffer_frames
        )

        # Background tasks for buffer management
        self._buffer_consumer_task: Optional[asyncio.Task] = None
        self._buffer_state_monitor_task: Optional[asyncio.Task] = None

        # internal values and switches used by the buffer consumer and state monitor tasks
        self._audio_buffer_consumer_can_send = asyncio.Event()
        self._audio_buffer_bytes_buffered: int = (
            0  # current size of data buffered in _audio_buffer in bytes
        )
        self._initial_jitter_buffer_bytes: int = 0  # calculated based on optimal frame size
        self._initial_jitter_buffer_is_filled: bool = False
        self._max_remote_audio_buffer_bytes: int = (
            0  # calculated based on optimal frame size and _max_remote_audio_buffer_frames
        )
        self._optimal_frame_size: Optional[int] = None
        self._ptime: float = 20  # default to 20ms
        self._remote_audio_buffer_bytes: int = 0  # calculated based on optimal frame size
        self._remote_audio_buffer_is_full: bool = False
        self._remote_audio_buffer_resume_threshold_bytes: int = (
            0  # calculated based on optimal frame size and _remote_audio_buffer_resume_threshold
        )

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def set_client_connection(self, websocket: Optional[websockets.WebSocketServerProtocol]):
        """Set the active client WebSocket connection.

        Args:
            websocket: The WebSocket connection to set as active, or None to clear.
        """
        if self._websocket != None and websocket != None:
            logger.warning(
                f"We already have a client connected, ignoring the new connection from {websocket.remote_address}"
            )
            return
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

        await self._params.serializer.setup(frame)
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport and send final frame.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._terminate(self, gracefully=True)

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and send cancellation frame.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._terminate(self)

    async def cleanup(self):
        """Cleanup resources and parent transport."""
        await super().cleanup()
        await self._terminate(self)
        await self._transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle interruption timing.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._flush_buffers()

        elif (
            isinstance(frame, InputTransportMessageFrame)
            and frame.message.get("event", None) == "MEDIA_START"
        ):
            await self._handle_media_start(frame.message)

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
        return await self._write_to_buffer(frame)

    async def _handle_media_start(self, message: dict):
        """Handle MEDIA_START message to configure transport parameters.

        Args:
            message: The MEDIA_START message containing Asterisk's channel_websocket configuration data.
        """
        # extract optimal frame size and ptime from the message
        optimal_frame_size = message.get("optimal_frame_size", None)
        ptime = message.get("ptime", None)

        if optimal_frame_size is None:
            logger.warning(f"{self} MEDIA_START message missing optimal_frame_size")
            raise ValueError("MEDIA_START message missing optimal_frame_size")
        if ptime is None:
            logger.warning(f"{self} MEDIA_START message missing ptime")
            raise ValueError("MEDIA_START message missing ptime")

        # Update internal parameters
        self._optimal_frame_size = optimal_frame_size
        logger.debug(f"{self} optimal frame size set to {self._optimal_frame_size} bytes")

        if ptime / 1000 != self._ptime:
            self._ptime = ptime / 1000  # convert to seconds
            logger.debug(f"{self} ptime set to {self._ptime} seconds")

        # Calculate internal buffer parameters based on optimal frame size
        self._max_remote_audio_buffer_bytes = (
            self._params.max_remote_audio_buffer_frames * self._optimal_frame_size
        )
        logger.debug(
            f"{self} remote audio buffer max size set to {self._max_remote_audio_buffer_bytes} bytes"
        )

        # Calculate resume threshold in bytes
        self._remote_audio_buffer_resume_threshold_bytes = int(
            self._params.remote_audio_buffer_resume_threshold * self._max_remote_audio_buffer_bytes
        )
        logger.debug(
            f"{self} remote audio buffer resume threshold set to {self._remote_audio_buffer_resume_threshold_bytes} bytes"
        )

        # Calculate initial jitter buffer size in bytes if enabled
        if self._params.initial_jitter_buffer_ms > 0:
            self._initial_jitter_buffer_bytes = int(
                (self._params.initial_jitter_buffer_ms / ptime) * optimal_frame_size
            )
            logger.debug(
                f"{self} initial jitter buffer size set to {self._initial_jitter_buffer_bytes} bytes"
            )

        # Send START_MEDIA_BUFFERING command
        command = self._params.serializer.form_command("START_MEDIA_BUFFERING")
        logger.debug(f"{self} sending START_MEDIA_BUFFERING command to Asterisk")
        try:
            if self._websocket:
                await self._websocket.send(command)
        except Exception as e:
            logger.error(
                f"{self} exception sending START_MEDIA_BUFFERING command: {e.__class__.__name__} ({e})"
            )

        # Start the buffer consumer and state monitor tasks
        self._buffer_state_monitor_task = self.create_task(self._buffer_state_monitor())
        self._buffer_consumer_task = self.create_task(self._buffer_consumer())

    async def _write_to_buffer(self, frame: OutputAudioRawFrame):
        """Write an raw audio bytes to the local audio buffer."""
        try:
            payload = await self._params.serializer.serialize(frame)
        except Exception as e:
            logger.error(f"{self} exception serializing data: {e.__class__.__name__} ({e})")
            return False

        if payload:
            logger.trace(f"{self} buffering audio frame of size {len(payload)} bytes")
            try:
                self._audio_buffer.put_nowait(payload)
                if (
                    self._params.initial_jitter_buffer_ms > 0
                    and not self._initial_jitter_buffer_is_filled
                ):
                    self._audio_buffer_bytes_buffered = self._audio_buffer_bytes_buffered + len(
                        payload
                    )
            except Exception as e:
                logger.error(f"Error adding frame to buffer: {e}")
                return False

    async def _buffer_consumer(self):
        """Consume frames from the local buffer and send them to Asterisk."""
        # Check if we need to fill up the initial jitter buffer before stat sending audio to websocket
        if self._params.initial_jitter_buffer_ms > 0:
            # Wait till the initial jitter buffer is filled
            while not self._initial_jitter_buffer_is_filled:
                if self._audio_buffer_bytes_buffered < self._initial_jitter_buffer_bytes:
                    await asyncio.sleep(0.02)  # Avoid busy waiting
                else:
                    self._initial_jitter_buffer_is_filled = True
                    logger.debug(
                        f"{self} Initial jitter buffer filled, starting to send audio frames"
                    )

        # Allow sending data now
        self._audio_buffer_consumer_can_send.set()

        # Start consuming from the buffer
        # Read from the buffer and send to websocket if possible
        while True:
            # Wait until we have date in local buffer and space in remote buffer (to avoid overfilling remote buffer and busy waiting)
            await self._audio_buffer_consumer_can_send.wait()

            payload = await self._audio_buffer.get()
            logger.trace(f"{self} sending buffered data of size {len(payload)} bytes")

            try:
                # Send the data to websocket
                await self._websocket.send(payload)

                # Update the remote audio buffer counter
                self._remote_audio_buffer_bytes = self._remote_audio_buffer_bytes + len(payload)

                # Check if remote buffer is full now
                if self._remote_audio_buffer_bytes >= self._max_remote_audio_buffer_bytes:
                    self._remote_audio_buffer_is_full = True
                    # Pause the buffer consumer until remote buffer has space again
                    self._audio_buffer_consumer_can_send.clear()
                    logger.debug(f"{self} remote audio buffer is full, pausing buffer consumer")
            except Exception as e:
                logger.error(
                    f"{self} exception sending buffered data: {e.__class__.__name__} ({e})"
                )
            self._audio_buffer.task_done()

    async def _buffer_state_monitor(self):
        """Monitor the local and the remote audio buffers state."""
        while True:
            last_check_time = time.monotonic()

            await asyncio.sleep(
                self._ptime
            )  # Check every ptime seconds, it's Ok in term of frequency, but
            # the problem is that asyncio sleep doesn't guarantee exact timing, so we have some drift behind.
            # In reality we sleep a bit more than ptime each time, so we think Asterisk consumed less data than it actually did.
            # So we need to calculate how much time had effectively passed since the last check.
            current_time = time.monotonic()
            elapsed = current_time - last_check_time
            last_check_time = current_time

            # Remoter buffer counter update logic
            # Decrease the remote audio buffer bytes based on the elapsed time and optimal frame size
            bytes_consumed = round((elapsed / self._ptime) * self._optimal_frame_size)
            self._remote_audio_buffer_bytes = max(
                0, self._remote_audio_buffer_bytes - bytes_consumed
            )

            if self._remote_audio_buffer_bytes > 0:
                logger.trace(
                    f"{self} remote audio buffer size: {self._remote_audio_buffer_bytes} bytes"
                )
            self._remote_audio_buffer_is_full = (
                self._remote_audio_buffer_bytes >= self._max_remote_audio_buffer_bytes
            )

            # Based on the state decide whether to allow sending more data from _buffer_consumer loop
            # ON - when local buffer has data, remote buffer has space and self._audio_buffer_consumer_can_send.is_set(), otherwise OFF

            if self._audio_buffer_consumer_can_send.is_set():
                if self._remote_audio_buffer_is_full:
                    logger.trace(f"{self} pausing buffer consumer as remote buffer is full")
                    self._audio_buffer_consumer_can_send.clear()
                elif self._audio_buffer.empty():
                    logger.trace(f"{self} pausing buffer consumer as local buffer is empty")
                    self._audio_buffer_consumer_can_send.clear()
            elif (
                not (self._remote_audio_buffer_is_full or self._audio_buffer.empty())
                and self._remote_audio_buffer_bytes
                < self._remote_audio_buffer_resume_threshold_bytes
            ):
                logger.trace(
                    f"{self} resuming buffer consumer as remote buffer has enough space and local buffer has data"
                )
                self._audio_buffer_consumer_can_send.set()

    async def _flush_audio_buffer(self):
        """Flush the local audio buffer."""
        while not self._audio_buffer.empty():
            try:
                self._audio_buffer.get_nowait()
                self._audio_buffer.task_done()
            except asyncio.QueueEmpty:
                break

    async def _flush_remote_audio_buffer(self):
        """Flush the remote audio buffer state."""
        self._remote_audio_buffer_bytes = 0
        self._remote_audio_buffer_is_full = False
        payload = self._params.serializer.form_command("FLUSH_MEDIA")

        try:
            if self._websocket:
                await self._websocket.send(payload)
        except Exception as e:
            logger.error(
                f"{self} exception sending FLUSH_MEDIA command: {e.__class__.__name__} ({e})"
            )

    async def _flush_buffers(self):
        await self._flush_audio_buffer()
        await self._flush_remote_audio_buffer()

    async def _terminate(self, gracefully: bool = False):
        """Terminate the output transport.

        Args:
            gracefully: Whether to terminate gracefully, allowing proper connection closure.
        """
        if gracefully:
            # Wait till local audio buffer is empty and remote audio buffer is empty
            while not self._audio_buffer.empty() or self._remote_audio_buffer_bytes > 0:
                await asyncio.sleep(0.02)  # Avoid busy waiting
        else:
            # stop the buffers immediately
            await self._flush_buffers()

        await self.cancel_task(self._buffer_consumer_task)
        await self.cancel_task(self._buffer_state_monitor_task)


class AsteriskWSServerTransport(BaseTransport):
    """WebSocket server transport for bidirectional real-time communication.

    Provides a complete WebSocket server implementation with separate input and
    output transports, client connection management, and event handling for
    real-time audio and data streaming applications.
    """

    def __init__(
        self,
        params: AsteriskWSServerParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
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
        self._params = params
        self._host = self._params.host
        self._port = self._params.port

        self._callbacks = WebsocketServerCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
            on_websocket_ready=self._on_websocket_ready,
        )
        self._input: Optional[AsteriskWSServerInputTransport] = None
        self._output: Optional[AsteriskWSServerOutputTransport] = None
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")
        self._register_event_handler("on_websocket_ready")

    def input(self) -> AsteriskWSServerInputTransport:
        """Get the input transport for receiving client data.

        Returns:
            The WebSocket server input transport instance.
        """
        if not self._input:
            self._input = AsteriskWSServerInputTransport(
                self, self._params, self._callbacks, name=self._input_name
            )
        return self._input

    def output(self) -> AsteriskWSServerOutputTransport:
        """Get the output transport for sending data to clients.

        Returns:
            The WebSocket server output transport instance.
        """
        if not self._output:
            self._output = AsteriskWSServerOutputTransport(
                self, self._params, name=self._output_name
            )
        return self._output

    async def _on_client_connected(self, websocket):
        """Handle client connection events."""
        if self._output:
            await self._output.set_client_connection(websocket)
            await self._call_event_handler("on_client_connected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_client_disconnected(self, websocket):
        """Handle client disconnection events."""
        if self._output:
            await self._output.set_client_connection(None)
            await self._output._terminate()
            await self._call_event_handler("on_client_disconnected", websocket)
        else:
            logger.error("A WebsocketServerTransport output is missing in the pipeline")

    async def _on_session_timeout(self, websocket):
        """Handle client session timeout events."""
        await self._call_event_handler("on_session_timeout", websocket)

    async def _on_websocket_ready(self):
        """Handle WebSocket server ready events."""
        await self._call_event_handler("on_websocket_ready")
