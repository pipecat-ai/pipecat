#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket-based LLM service base class.

This module depends on the optional ``websockets`` package and is kept
separate from :mod:`pipecat.services.llm_service` so that importing the core
LLM service base classes does not require ``websockets`` to be installed.
"""

from __future__ import annotations

import json
from typing import Generic

from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, StartFrame
from pipecat.services.llm_service import LLMService, TAdapter
from pipecat.services.websocket_service import WebsocketService


class WebsocketReconnectedError(Exception):
    """Raised by ``_ws_send``/``_ws_recv`` after a transparent reconnection.

    Signals that the WebSocket connection was lost and automatically
    re-established.  The current inference should be restarted — any
    connection-local state on the server (e.g. cached responses) is gone.
    """

    pass


class WebsocketLLMService(LLMService[TAdapter], WebsocketService, Generic[TAdapter]):
    """Base class for websocket-based LLM services.

    Each LLM inference is a discrete request/response exchange: send one
    request, receive events inline until a terminal event, then wait for
    the next frame to trigger an inference.  This contrasts with
    ``WebsocketTTSService`` / ``WebsocketSTTService`` which stream data
    continuously via a background receive loop
    (``_receive_task_handler``).  This class does **not** start a
    background receive loop.

    Provides connection lifecycle management (connect on start, disconnect
    on stop/cancel), automatic reconnection with exponential backoff, and
    three helpers for running each inference:

    1. ``_ensure_connected()`` — verify the websocket is alive, reconnect
       with exponential backoff if not.
    2. ``_ws_send(message)`` — send the inference request as JSON.
    3. ``_ws_recv()`` — receive and parse response events one at a time
       until the caller sees a terminal event.

    ``_ws_send`` and ``_ws_recv`` catch ``ConnectionClosed`` transparently,
    auto-reconnect via ``_try_reconnect``, and raise
    ``WebsocketReconnectedError`` so callers know the inference must be
    restarted.  If reconnection fails, the original ``ConnectionClosed``
    propagates.

    Subclasses must implement:
        ``_connect_websocket()``: Establish the websocket connection.
        ``_disconnect_websocket()``: Close the websocket and clean up.

    Event handlers:
        on_connection_error: Called when a websocket connection error occurs.

    Example::

        @llm.event_handler("on_connection_error")
        async def on_connection_error(llm: LLMService, error: str):
            logger.error(f"LLM connection error: {error}")
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the Websocket LLM service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        # pyright stumbles here because the TypeVar default makes
        # `LLMService` resolve to `LLMService[BaseLLMAdapter]` invariantly,
        # while `self` is `WebsocketLLMService[TAdapter]` for an arbitrary
        # TAdapter. The runtime call is fine — generics are erased.
        LLMService.__init__(self, **kwargs)  # pyright: ignore[reportArgumentType]
        WebsocketService.__init__(self, reconnect_on_error=reconnect_on_error, **kwargs)
        self._register_event_handler("on_connection_error")

    # -- lifecycle ------------------------------------------------------------

    async def _connect(self):
        """Connect: reset flags and establish the websocket."""
        await super()._connect()
        await self._connect_websocket()

    async def _disconnect(self):
        """Disconnect: set flags and close the websocket."""
        await super()._disconnect()
        await self._disconnect_websocket()

    async def start(self, frame: StartFrame):
        """Start the service and establish WebSocket connection.

        Args:
            frame: The start frame triggering service initialization.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close WebSocket connection.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close WebSocket connection.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    # -- per-inference helpers ------------------------------------------------

    async def _ws_send(self, message: dict):
        """Send a JSON message over the websocket.

        Guards against sends during intentional disconnect.  If the send
        fails with ``ConnectionClosed``, attempts to reconnect and raises
        ``WebsocketReconnectedError`` on success so the caller can restart
        the inference.  If reconnection fails, the original
        ``ConnectionClosed`` propagates.

        Args:
            message: The message dict to serialize and send.
        """
        if self._disconnecting or not self._websocket:
            return
        try:
            await self._websocket.send(json.dumps(message))
        except ConnectionClosed:
            if self._disconnecting:
                return
            success = await self._try_reconnect(report_error=self._report_error)
            if success:
                raise WebsocketReconnectedError()
            raise

    async def _ws_recv(self) -> dict:
        """Receive and parse a JSON message from the websocket.

        If the receive fails with ``ConnectionClosed``, attempts to
        reconnect and raises ``WebsocketReconnectedError`` on success.
        If reconnection fails, the original ``ConnectionClosed``
        propagates.

        Returns:
            The parsed JSON message as a dict.
        """
        # Should never happen — `_ensure_connected` (which callers must invoke
        # first) raises ConnectionError if it can't establish a websocket.
        # Match that contract here.
        if self._websocket is None:
            raise ConnectionError(f"{self} _ws_recv called without a websocket")
        try:
            raw = await self._websocket.recv()
            return json.loads(raw)
        except ConnectionClosed:
            if self._disconnecting:
                raise
            success = await self._try_reconnect(report_error=self._report_error)
            if success:
                raise WebsocketReconnectedError()
            raise

    async def _ensure_connected(self):
        """Ensure the websocket is connected, reconnecting if needed.

        Uses ``_try_reconnect`` with exponential backoff.

        Raises:
            ConnectionError: If the connection could not be established.
        """
        if self._websocket and self._websocket.state is not State.CLOSED:
            return
        success = await self._try_reconnect(report_error=self._report_error)
        if not success:
            raise ConnectionError(f"{self} failed to establish WebSocket connection")

    # -- WebsocketService interface -------------------------------------------

    async def _receive_messages(self):
        """Not used — messages are received inline during each inference.

        This satisfies the ``WebsocketService`` abstract method but is never
        called because ``_receive_task_handler`` is never started.
        """
        raise NotImplementedError(
            "WebsocketLLMService receives messages inline during inference, "
            "not via a continuous background loop"
        )

    async def _report_error(self, error: ErrorFrame):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error_frame(error)
