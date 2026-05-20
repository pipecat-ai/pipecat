#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket client proxy that forwards bus messages to a remote server."""

import asyncio

from loguru import logger

from pipecat.bus import BusMessage, BusTaskRegistryMessage
from pipecat.bus.messages import BusLocalMessage
from pipecat.bus.serializers import JSONMessageSerializer
from pipecat.bus.serializers.base import MessageSerializer
from pipecat.pipeline.base_task import BaseTask

try:
    import websockets
    from websockets.asyncio.client import connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use WebSocketProxyClientTask, you need to `pip install pipecat-ai[websockets-base]`."
    )
    raise Exception(f"Missing module: {e}")


class WebSocketProxyClientTask(BaseTask):
    """Forwards bus messages to a remote task over WebSocket.

    Connects to a WebSocket URL and forwards messages between a local
    task and a remote task. Only messages targeted at the remote task
    are sent. Only messages targeted at the local task are accepted.

    Event handlers available:

    - on_connected: Fired when the WebSocket connection is established.
    - on_disconnected: Fired when the WebSocket connection is closed.

    Example::

        proxy = WebSocketProxyClientTask(
            "proxy",
            url="ws://remote-server:8765/ws",
            remote_task_name="worker",
            local_task_name="voice",
        )

        @proxy.event_handler("on_connected")
        async def on_connected(task, websocket):
            logger.info("Connected to remote server")

        @proxy.event_handler("on_disconnected")
        async def on_disconnected(task, websocket):
            logger.info("Disconnected from remote server")

        await runner.spawn(proxy)
    """

    def __init__(
        self,
        name: str,
        *,
        url: str,
        remote_task_name: str,
        local_task_name: str,
        forward_messages: tuple[type[BusMessage], ...] = (),
        headers: dict[str, str] | None = None,
        serializer: MessageSerializer | None = None,
        active: bool = False,
    ):
        """Initialize the WebSocketProxyClientTask.

        Args:
            name: Unique name for this task.
            url: The WebSocket URL to connect to.
            remote_task_name: Name of the task on the remote server.
                Only messages targeted at this task are forwarded.
            local_task_name: Name of the local task that should
                receive responses. Only inbound messages targeted at
                this task are accepted.
            forward_messages: Additional message types to forward from
                the local task (e.g. ``(BusFrameMessage,)`` for frame
                routing). These are forwarded based on source task name
                only, regardless of target.
            headers: Optional HTTP headers sent with the WebSocket
                handshake (e.g. for authentication).
            serializer: Serializer for bus messages. Defaults to
                `JSONMessageSerializer`.
            active: Whether the task starts active. Defaults to ``False``
                because ``on_activated`` opens the WebSocket connection,
                which is almost always a deliberate action triggered by an
                upstream event (e.g. the local client connecting). Pass
                ``True`` to connect as soon as the task starts.
        """
        super().__init__(name, active=active)
        self._url = url
        self._remote_task_name = remote_task_name
        self._local_task_name = local_task_name
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

        logger.debug(f"Task '{self}': connecting to {self._url}")

        self._ws = await connect(self._url, additional_headers=self._headers)

        logger.debug(f"Task '{self}': connected to {self._url}")

        await self._call_event_handler("on_connected", self._ws)

        self._receive_task = self.create_task(self._receive_loop())

        # Schedule task right away.
        await asyncio.sleep(0)

    async def stop(self) -> None:
        """Cancel the receive loop and close the WebSocket connection."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._ws:
            await self._ws.close()
            logger.debug(f"Task '{self}': WebSocket connection closed")
            self._ws = None
        await super().stop()

    async def on_bus_message(self, message: BusMessage) -> None:
        """Forward messages targeted at the remote task.

        Args:
            message: The bus message to process.
        """
        await super().on_bus_message(message)

        if not self._ws:
            return

        if isinstance(message, BusLocalMessage):
            return

        # Forward targeted messages to the remote task.
        if message.target == self._remote_task_name:
            await self._send_ws(message)
        # Forward additional message types from the local task.
        elif isinstance(message, self._forward_messages):
            if message.source == self._local_task_name:
                await self._send_ws(message)

    async def _send_ws(self, message: BusMessage) -> None:
        """Serialize and send a message over the WebSocket."""
        if not self._ws:
            return
        try:
            data = self._serializer.serialize(message)
            await self._ws.send(data)
            logger.trace(f"Task '{self}': sent {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Task '{self}': connection closed, stopping forwarding")
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

                    # Accept registry messages (target=None) for task discovery.
                    if isinstance(message, BusTaskRegistryMessage):
                        logger.trace(
                            f"Task '{self}': received registry from remote: {message.tasks}"
                        )
                        await self.send_bus_message(message)
                        continue

                    # Accept additional message types (e.g. BusFrameMessage).
                    if self._forward_messages and isinstance(message, self._forward_messages):
                        logger.trace(f"Task '{self}': received {message} from remote")
                        await self.send_bus_message(message)
                        continue

                    # Only accept other messages targeted at the local task.
                    if message.target != self._local_task_name:
                        logger.warning(
                            f"Task '{self}': dropped inbound message with "
                            f"unexpected target '{message.target}'"
                        )
                        continue

                    logger.trace(f"Task '{self}': received {message} from remote")
                    await self.send_bus_message(message)
                except Exception:
                    logger.exception(f"Task '{self}': failed to deserialize remote message")
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Task '{self}': WebSocket connection closed")
            ws = self._ws
            self._ws = None
            await self._call_event_handler("on_disconnected", ws)
        except asyncio.CancelledError:
            pass
