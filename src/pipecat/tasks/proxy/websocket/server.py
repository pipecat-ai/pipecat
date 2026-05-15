#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket server proxy that receives bus messages from a remote client."""

import asyncio

from loguru import logger

from pipecat.bus import BusMessage, BusTaskRegistryMessage
from pipecat.bus.messages import BusLocalMessage
from pipecat.bus.serializers import JSONMessageSerializer
from pipecat.bus.serializers.base import MessageSerializer
from pipecat.pipeline.base_task import BaseTask
from pipecat.registry.types import TaskReadyData, TaskRegistryEntry

try:
    from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use WebSocketProxyServerTask, you need to `pip install starlette`.")
    raise Exception(f"Missing module: {e}")


class WebSocketProxyServerTask(BaseTask):
    """Receives bus messages from a remote client over WebSocket.

    Accepts a FastAPI/Starlette WebSocket connection and forwards
    messages between the remote client and a local task. Only messages
    from the local task targeted at the remote task are sent. Only
    inbound messages targeted at the local task are accepted.

    Event handlers available:

    - on_client_connected: Fired when the WebSocket client connects and the proxy is ready.
    - on_client_disconnected: Fired when the WebSocket client disconnects.

    Example::

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            proxy = WebSocketProxyServerTask(
                "gateway",
                websocket=websocket,
                task_name="worker",
                remote_task_name="voice",
            )

            @proxy.event_handler("on_client_connected")
            async def on_client_connected(task, websocket):
                logger.info("Client connected")

            @proxy.event_handler("on_client_disconnected")
            async def on_client_disconnected(task, websocket):
                logger.info("Client disconnected")

            await runner.spawn(proxy)
    """

    def __init__(
        self,
        name: str,
        *,
        websocket: WebSocket,
        task_name: str,
        remote_task_name: str,
        forward_messages: tuple[type[BusMessage], ...] = (),
        serializer: MessageSerializer | None = None,
    ):
        """Initialize the WebSocketProxyServerTask.

        Args:
            name: Unique name for this task.
            websocket: An accepted FastAPI/Starlette WebSocket connection.
            task_name: Name of the local task to route messages to/from.
                Only messages from this task are forwarded to the client.
            remote_task_name: Name of the task on the remote client.
                Only outbound messages targeted at this task are sent.
                Only inbound messages targeted at the local task are accepted.
            forward_messages: Additional message types to forward from
                the local task (e.g. ``(BusFrameMessage,)`` for frame
                routing). These are forwarded based on source task name
                only, regardless of target.
            serializer: Serializer for bus messages. Defaults to
                `JSONMessageSerializer`.
        """
        super().__init__(name)
        self._ws: WebSocket | None = websocket
        self._task_name = task_name
        self._remote_task_name = remote_task_name
        self._forward_messages = forward_messages
        self._serializer = serializer or JSONMessageSerializer()
        self._receive_task: asyncio.Task | None = None

        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    async def start(self) -> None:
        """Start the WebSocket receive loop and watch the local task."""
        await super().start()

        logger.debug(f"Task '{self}': WebSocket proxy server ready")

        await self._call_event_handler("on_client_connected", self._ws)

        self._receive_task = self.create_task(self._receive_loop(), f"{self.name}::ws_receive")

        # Schedule task right away.
        await asyncio.sleep(0)

        # Watch the local task so we can notify the remote side when it's ready.
        await self.watch_task(self._task_name)

    async def stop(self) -> None:
        """Cancel the receive loop and close the WebSocket connection."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._ws and self._ws.client_state == WebSocketState.CONNECTED:
            await self._ws.close()
            logger.debug(f"Task '{self}': WebSocket connection closed")
        await super().stop()

    async def on_task_ready(self, data: TaskReadyData) -> None:
        """Notify the remote client that the local task is ready."""
        if not self._ws:
            return

        if data.task_name != self._task_name:
            return

        logger.debug(f"Task '{self}': local task '{self._task_name}' ready, notifying remote")

        try:
            msg = BusTaskRegistryMessage(
                source=self.name,
                runner=data.runner,
                tasks=[TaskRegistryEntry(name=self._task_name)],
            )
            await self._send_ws(msg)
        except Exception:
            logger.exception(f"Task '{self}': failed to send registry to remote")

    async def on_bus_message(self, message: BusMessage) -> None:
        """Forward messages from the local task to the remote client.

        Args:
            message: The bus message to process.
        """
        await super().on_bus_message(message)

        if not self._ws:
            return

        if isinstance(message, BusLocalMessage):
            return

        if message.source != self._task_name:
            return

        # Forward targeted messages from the local task to the remote task.
        if message.target == self._remote_task_name:
            await self._send_ws(message)
        # Forward additional message types from the local task.
        elif isinstance(message, self._forward_messages):
            await self._send_ws(message)

    async def _send_ws(self, message: BusMessage) -> None:
        """Serialize and send a message over the WebSocket."""
        if not self._ws:
            return
        try:
            data = self._serializer.serialize(message)
            await self._ws.send_bytes(data)
            logger.trace(f"Task '{self}': sent {message}")
        except WebSocketDisconnect:
            logger.warning(f"Task '{self}': connection closed, stopping forwarding")
            ws = self._ws
            self._ws = None
            await self._call_event_handler("on_client_disconnected", ws)

    async def _receive_loop(self) -> None:
        """Read messages from the WebSocket and put them on the local bus."""
        assert self._ws is not None, "start() must run before _receive_loop"
        try:
            while True:
                data = await self._ws.receive_bytes()
                try:
                    message = self._serializer.deserialize(data)
                    if not message:
                        continue

                    # Accept additional message types (e.g. BusFrameMessage).
                    if self._forward_messages and isinstance(message, self._forward_messages):
                        logger.trace(f"Task '{self}': received {message} from client")
                        await self.send_message(message)
                        continue

                    # Only accept other messages targeted at the local task.
                    if message.target != self._task_name:
                        logger.warning(
                            f"Task '{self}': dropped inbound message with "
                            f"unexpected target '{message.target}'"
                        )
                        continue

                    logger.trace(f"Task '{self}': received {message} from client")
                    await self.send_message(message)
                except Exception:
                    logger.exception(f"Task '{self}': failed to deserialize client message")
        except WebSocketDisconnect:
            logger.warning(f"Task '{self}': client disconnected")
            ws = self._ws
            self._ws = None
            await self._call_event_handler("on_client_disconnected", ws)
        except asyncio.CancelledError:
            pass
