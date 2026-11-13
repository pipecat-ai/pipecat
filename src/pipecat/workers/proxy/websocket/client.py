#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket client proxy that forwards bus messages to a remote server."""

import asyncio

import websockets
from loguru import logger
from websockets.asyncio.client import connect

from pipecat.bus import BusMessage, BusWorkerRegistryMessage
from pipecat.bus.messages import BusLocalMessage
from pipecat.bus.serializers import JSONMessageSerializer
from pipecat.bus.serializers.base import MessageSerializer
from pipecat.workers.base_worker import BaseWorker


class WebSocketProxyClient(BaseWorker):
    """Forwards bus messages to a remote worker over WebSocket.

    Connects to a WebSocket URL and forwards messages between a local
    worker and a remote worker. Only messages targeted at the remote worker
    are sent. Only messages targeted at the local worker are accepted.

    Event handlers available:

    - on_connected: Fired when the WebSocket connection is established.
    - on_disconnected: Fired when the WebSocket connection is closed.

    Example::

        proxy = WebSocketProxyClient(
            "proxy",
            url="ws://remote-server:8765/ws",
            remote_worker_name="worker",
            local_worker_name="voice",
        )

        @proxy.event_handler("on_connected")
        async def on_connected(worker, websocket):
            logger.info("Connected to remote server")

        @proxy.event_handler("on_disconnected")
        async def on_disconnected(worker, websocket):
            logger.info("Disconnected from remote server")

        await runner.add_workers(proxy)
    """

    def __init__(
        self,
        name: str,
        *,
        url: str,
        remote_worker_name: str,
        local_worker_name: str,
        forward_messages: tuple[type[BusMessage], ...] = (),
        headers: dict[str, str] | None = None,
        serializer: MessageSerializer | None = None,
        active: bool = False,
    ):
        """Initialize the WebSocketProxyClient.

        Args:
            name: Unique name for this worker.
            url: The WebSocket URL to connect to.
            remote_worker_name: Name of the worker on the remote server.
                Only messages targeted at this worker are forwarded.
            local_worker_name: Name of the local worker that should
                receive responses. Only inbound messages targeted at
                this worker are accepted.
            forward_messages: Additional message types to forward from
                the local worker (e.g. ``(BusFrameMessage,)`` for frame
                routing). These are forwarded based on source worker name
                only, regardless of target.
            headers: Optional HTTP headers sent with the WebSocket
                handshake (e.g. for authentication).
            serializer: Serializer for bus messages. Defaults to
                `JSONMessageSerializer`.
            active: Whether the worker starts active. Defaults to ``False``
                because ``on_activated`` opens the WebSocket connection,
                which is almost always a deliberate action triggered by an
                upstream event (e.g. the local client connecting). Pass
                ``True`` to connect as soon as the worker starts.
        """
        super().__init__(name, active=active)
        self._url = url
        self._remote_worker_name = remote_worker_name
        self._local_worker_name = local_worker_name
        self._forward_messages = forward_messages
        self._headers = headers or {}
        self._serializer = serializer or JSONMessageSerializer()
        self._ws = None
        self._receive_task: asyncio.Task | None = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")

    async def on_activated(self, args: dict | None) -> None:
        """Connect to the remote WebSocket server."""
        await super().on_activated(args)

        logger.debug(f"Worker '{self}': connecting to {self._url}")

        self._ws = await connect(self._url, additional_headers=self._headers)

        logger.debug(f"Worker '{self}': connected to {self._url}")

        await self._call_event_handler("on_connected", self._ws)

        self._receive_task = self.create_task(self._receive_loop())

        # Schedule worker right away.
        await asyncio.sleep(0)

    async def stop(self) -> None:
        """Cancel the receive loop and close the WebSocket connection."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._ws:
            await self._ws.close()
            logger.debug(f"Worker '{self}': WebSocket connection closed")
            self._ws = None
        await super().stop()

    async def on_bus_message(self, message: BusMessage) -> None:
        """Forward messages targeted at the remote worker.

        Args:
            message: The bus message to process.
        """
        await super().on_bus_message(message)

        if not self._ws:
            return

        if isinstance(message, BusLocalMessage):
            return

        # Forward targeted messages to the remote worker.
        if message.target == self._remote_worker_name:
            await self._send_ws(message)
        # Forward additional message types from the local worker.
        elif isinstance(message, self._forward_messages):
            if message.source == self._local_worker_name:
                await self._send_ws(message)

    async def _send_ws(self, message: BusMessage) -> None:
        """Serialize and send a message over the WebSocket."""
        if not self._ws:
            return
        try:
            data = self._serializer.serialize(message)
            await self._ws.send(data)
            logger.trace(f"Worker '{self}': sent {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Worker '{self}': connection closed, stopping forwarding")
            ws = self._ws
            self._ws = None
            await self._call_event_handler("on_disconnected", ws)

    async def _receive_loop(self) -> None:
        """Read messages from the WebSocket and put them on the local bus."""
        assert self._ws is not None, "on_activated() must run before _receive_loop"
        try:
            async for data in self._ws:
                try:
                    payload = data if isinstance(data, bytes) else data.encode()
                    message = self._serializer.deserialize(payload)
                    if not message:
                        continue

                    # Accept registry messages (target=None) for worker discovery.
                    if isinstance(message, BusWorkerRegistryMessage):
                        logger.trace(
                            f"Worker '{self}': received registry from remote: {message.workers}"
                        )
                        await self.send_bus_message(message)
                        continue

                    # Accept additional message types (e.g. BusFrameMessage).
                    if self._forward_messages and isinstance(message, self._forward_messages):
                        logger.trace(f"Worker '{self}': received {message} from remote")
                        await self.send_bus_message(message)
                        continue

                    # Only accept other messages targeted at the local worker.
                    if message.target != self._local_worker_name:
                        logger.warning(
                            f"Worker '{self}': dropped inbound message with "
                            f"unexpected target '{message.target}'"
                        )
                        continue

                    logger.trace(f"Worker '{self}': received {message} from remote")
                    await self.send_bus_message(message)
                except Exception:
                    logger.exception(f"Worker '{self}': failed to deserialize remote message")
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Worker '{self}': WebSocket connection closed")
            ws = self._ws
            self._ws = None
            await self._call_event_handler("on_disconnected", ws)
        except asyncio.CancelledError:
            pass
