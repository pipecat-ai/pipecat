#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Jambonz transport implementation for Pipecat.

This module provides comprehensive Jambonz integration for as transport.
"""

import asyncio
import time

from aiohttp import web
from typing import Optional, Awaitable, Callable, AsyncIterator
from pydantic import BaseModel
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams


class JambonzTransportParams(TransportParams):
    """Configuration parameters for Jambonz transport.

    Parameters:
        serializer: Frame serializer for encoding/decoding messages.
        session_timeout: Session timeout in seconds, None for no timeout.
    """

    serializer: Optional[FrameSerializer] = None
    session_timeout: Optional[int] = None


class JambonzTransportCallbacks(BaseModel):
    """Callback functions for Jambonz events.

    Parameters:
        on_client_connected: Called when a client connects to the WebSocket.
        on_client_disconnected: Called when a client disconnects from the WebSocket.
        on_session_timeout: Called when a session timeout occurs.
    """

    on_client_connected: Callable[[web.WebSocketResponse], Awaitable[None]]
    on_client_disconnected: Callable[[web.WebSocketResponse], Awaitable[None]]
    on_session_timeout: Callable[[web.WebSocketResponse], Awaitable[None]]


class JambonzTransportClient:
    """Jambonz client wrapper for handling connections and message passing.

    Manages Jambonz state, message sending/receiving, and connection lifecycle
    with support for both binary and text message types.
    """

    def __init__(
        self,
        websocket: web.WebSocketResponse,
        callbacks: JambonzTransportCallbacks,
        params: JambonzTransportParams,
    ):
        self._websocket = websocket
        self._callbacks = callbacks
        self._params = params
        self._closing = False

        self._disconnected_emitted = False

    async def send(self, data: str | bytes | dict):
        """Send data through the WebSocket connection."""
        try:
            if self._closing:
                return
            if isinstance(data, dict):
                await self._websocket.send_json(data)
            elif isinstance(data, bytes):
                await self._websocket.send_bytes(data)
            else:
                raise ValueError(f"Invalid data type: {type(data)}")
        except Exception as e:
            logger.error(f"Exception sending data: {e}")

    async def receive(self) -> AsyncIterator[bytes | str]:
        """Receive data from the WebSocket connection."""
        async for msg in self._websocket:
            yield msg

    async def disconnect(self):
        """Disconnect the websocket client."""
        self._closing = True

        await self.trigger_client_disconnected()

    async def trigger_client_disconnected(self):
        """Trigger the client disconnected callback."""
        if self._disconnected_emitted:
            return
        self._disconnected_emitted = True
        logger.info(f"Connection with Jambonz disconnected.")
        await self._callbacks.on_client_disconnected(self._websocket)

    async def trigger_client_connected(self):
        """Trigger the client connected callback."""
        logger.info(f"Connection with Jambonz established.")
        await self._callbacks.on_client_connected(self._websocket)

    async def trigger_client_timeout(self):
        """Trigger the client timeout callback."""
        logger.warning(f"Connection with Jambonz timed out.")
        await self._callbacks.on_session_timeout(self._websocket)

    @property
    def is_connected(self) -> bool:
        return self._websocket.closed is False

    @property
    def is_closing(self) -> bool:
        """Check if the WebSocket is in the process of closing.

        Returns:
            True if the WebSocket is in closing state.
        """
        return self._closing


class JambonzInputTransport(BaseInputTransport):
    """Input transport for Jambonz WebSocket connections.

    Handles incoming WebSocket messages, deserializes frames, and manages
    connection monitoring with optional session timeouts.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: JambonzTransportClient,
        params: JambonzTransportParams,
        **kwargs,
    ):
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

        # Propagate resolved input sample rate back to StartFrame
        frame.audio_in_sample_rate = self.sample_rate
        frame.audio_out_sample_rate = self.sample_rate

        if self._initialized:
            return

        self._initialized = True

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
                    logger.info(f"[AioHTTP WS] received frame={frame}")
                    await self.push_frame(frame)
        except Exception as e:
            logger.error(
                f"{self} exception receiving data: {e.__class__.__name__} ({e})"
            )

        await self._client.trigger_client_disconnected()

    async def _monitor_websocket(self):
        """Wait for self._params.session_timeout seconds, if the websocket is still open, trigger timeout event."""
        await asyncio.sleep(self._params.session_timeout)
        await self._client.trigger_client_timeout()


class JambonzOutputTransport(BaseOutputTransport):
    """Output transport for Jambonz WebSocket connections.

    Handles outgoing frame serialization, audio streaming with timing simulation,
    and WebSocket message transmission with optional WAV header generation.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: JambonzTransportClient,
        params: JambonzTransportParams,
        **kwargs,
    ):
        """Initialize the WebSocket output transport."""
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._params = params

        self._send_interval = 0
        self._next_send_time = 0

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

        # Propagate resolved output sample rate back to StartFrame so serializers see it
        frame.audio_out_sample_rate = self.sample_rate

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
        """Process outgoing frames with special handling for interruptions."""
        await super().process_frame(frame, direction)
        if isinstance(frame, StartInterruptionFrame):
            await self._write_frame(frame)
            self._next_send_time = 0

    async def send_message(
        self, frame: TransportMessageFrame | TransportMessageUrgentFrame
    ):
        """Send a transport message frame.

        Args:
            frame: The transport message frame to send.
        """
        await self._write_frame(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame):
        """Write an audio frame to the WebSocket with timing simulation.

        Args:
            frame: The output audio frame to write.
        """
        if self._client.is_closing or not self._client.is_connected:
            return

        frame = OutputAudioRawFrame(
            audio=frame.audio,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        await self._write_frame(frame)

        # Simulate audio playback with a sleep.
        await self._write_audio_sleep()

    async def _write_frame(self, frame: Frame):
        """Serialize and send a frame through the WebSocket."""
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


class JambonzTransport(BaseTransport):
    """Jambonz WebSocket transport for real-time audio/video streaming.

    Provides bidirectional WebSocket communication with frame serialization,
    session management, and event handling for client connections and timeouts.
    """

    def __init__(
        self,
        websocket: web.WebSocketResponse,
        params: JambonzTransportParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the Jambonz WebSocket transport.

        Args:
            websocket: The Jambonz WebSocket connection.
            params: Transport configuration parameters.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)

        self._params = params

        self._callbacks = JambonzTransportCallbacks(
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_session_timeout=self._on_session_timeout,
        )

        self._client = JambonzTransportClient(websocket, self._callbacks, self._params)

        self._input = JambonzInputTransport(
            self, self._client, self._params, name=self._input_name
        )
        self._output = JambonzOutputTransport(
            self, self._client, self._params, name=self._output_name
        )

        self.call_info = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_session_timeout")

    def input(self) -> JambonzInputTransport:
        """Get the input transport processor.

        Returns:
            The WebSocket input transport instance.
        """
        return self._input

    def output(self) -> JambonzOutputTransport:
        """Get the output transport processor.

        Returns:
            The WebSocket output transport instance.
        """
        return self._output

    def set_call_info(self, call_info: dict):
        """Set the call info for the transport."""
        self.call_info = call_info

    def get_call_info(self) -> dict:
        """Get the call info for the transport."""
        return self.call_info

    async def _on_client_connected(self, websocket):
        """Handle client connected event."""
        await self._call_event_handler("on_client_connected", websocket)

    async def _on_client_disconnected(self, websocket):
        """Handle client disconnected event."""
        await self._call_event_handler("on_client_disconnected", websocket)

    async def _on_session_timeout(self, websocket):
        """Handle session timeout event."""
        await self._call_event_handler("on_session_timeout", websocket)
