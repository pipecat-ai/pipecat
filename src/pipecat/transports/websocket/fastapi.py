#
# Copyright (c) 2024–2025, Daily
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
from typing import Awaitable, Callable, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    from fastapi import WebSocket
    from starlette.websockets import WebSocketState
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use FastAPI websockets, you need to `pip install pipecat-ai[websocket]`."
    )
    raise Exception(f"Missing module: {e}")


class FastAPIWebsocketParams(TransportParams):
    """Configuration parameters for FastAPI WebSocket transport.

    Parameters:
        add_wav_header: Whether to add WAV headers to audio frames.
        serializer: Frame serializer for encoding/decoding messages.
        session_timeout: Session timeout in seconds, None for no timeout.
    """

    add_wav_header: bool = False
    serializer: Optional[FrameSerializer] = None
    session_timeout: Optional[int] = None


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


class FastAPIWebsocketClient:
    """WebSocket client wrapper for handling connections and message passing.

    Manages WebSocket state, message sending/receiving, and connection lifecycle
    with support for both binary and text message types.
    """

    def __init__(self, websocket: WebSocket, is_binary: bool, callbacks: FastAPIWebsocketCallbacks):
        """Initialize the WebSocket client.

        Args:
            websocket: The FastAPI WebSocket connection.
            is_binary: Whether to use binary message format.
            callbacks: Event callback functions.
        """
        self._websocket = websocket
        self._closing = False
        self._is_binary = is_binary
        self._callbacks = callbacks
        self._leave_counter = 0

    async def setup(self, _: StartFrame):
        """Set up the WebSocket client.

        Args:
            _: The start frame (unused).
        """
        self._leave_counter += 1

    def receive(self) -> typing.AsyncIterator[bytes | str]:
        """Get an async iterator for receiving WebSocket messages.

        Returns:
            An async iterator yielding bytes or strings based on message type.
        """
        return self._websocket.iter_bytes() if self._is_binary else self._websocket.iter_text()

    async def send(self, data: str | bytes):
        """Send data through the WebSocket connection.

        Args:
            data: The data to send (string or bytes).
        """
        try:
            if self._can_send():
                if self._is_binary:
                    await self._websocket.send_bytes(data)
                else:
                    await self._websocket.send_text(data)
        except Exception as e:
            logger.error(
                f"{self} exception sending data: {e.__class__.__name__} ({e}), application_state: {self._websocket.application_state}"
            )
            # For some reason the websocket is disconnected, and we are not able to send data
            # So let's properly handle it and disconnect the transport if it is not already disconnecting
            if (
                self._websocket.application_state == WebSocketState.DISCONNECTED
                and not self.is_closing
            ):
                logger.warning("Closing already disconnected websocket!")
                self._closing = True

    async def disconnect(self):
        """Disconnect the WebSocket client."""
        self._leave_counter -= 1
        if self._leave_counter > 0:
            return

        if self.is_connected and not self.is_closing:
            self._closing = True
            try:
                await self._websocket.close()
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
        return self.is_connected and not self.is_closing

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
            self._monitor_websocket_task = self.create_task(self._monitor_websocket())
        await self._client.trigger_client_connected()
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
                else:
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

        # Trigger `on_client_disconnected` if the client actually disconnects,
        # that is, we are not the ones disconnecting.
        if not self._client.is_closing:
            await self._client.trigger_client_disconnected()

    async def _monitor_websocket(self):
        """Wait for self._params.session_timeout seconds, if the websocket is still open, trigger timeout event."""
        await asyncio.sleep(self._params.session_timeout)
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
    """

    def __init__(
        self,
        websocket: WebSocket,
        params: FastAPIWebsocketParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the FastAPI WebSocket transport.

        Args:
            websocket: The FastAPI WebSocket connection.
            params: Transport configuration parameters.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)

        self._params = params

        self._callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
        )

        is_binary = False
        if self._params.serializer:
            is_binary = self._params.serializer.type == FrameSerializerType.BINARY
        self._client = FastAPIWebsocketClient(websocket, is_binary, self._callbacks)

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
