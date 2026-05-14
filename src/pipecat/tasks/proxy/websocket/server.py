#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket server proxy that receives bus messages from a remote client."""

import asyncio

from loguru import logger

from pipecat.bus import BusMessage, BusTaskRegistryMessage, TaskBus
from pipecat.bus.messages import BusLocalMessage
from pipecat.bus.serializers import JSONMessageSerializer
from pipecat.bus.serializers.base import MessageSerializer
from pipecat.pipeline.base_task import BaseTask
from pipecat.registry.types import TaskReadyData, TaskRegistryEntry

try:
    from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use WebSocketProxyServerAgent, you need to `pip install pipecat-ai-subagents[websocket]`."
    )
    raise Exception(f"Missing module: {e}")


class WebSocketProxyServerAgent(BaseTask):
    """Receives bus messages from a remote client over WebSocket.

    Accepts a FastAPI/Starlette WebSocket connection and forwards
    messages between the remote client and a local agent. Only messages
    from the local agent targeted at the remote agent are sent. Only
    inbound messages targeted at the local agent are accepted.

    Event handlers available:

    - on_client_connected: Fired when the WebSocket client connects and the proxy is ready.
    - on_client_disconnected: Fired when the WebSocket client disconnects.

    Example::

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            proxy = WebSocketProxyServerAgent(
                "gateway",
                bus=runner.bus,
                websocket=websocket,
                agent_name="worker",
                remote_agent_name="voice",
            )

            @proxy.event_handler("on_client_connected")
            async def on_client_connected(agent, websocket):
                logger.info("Client connected")

            @proxy.event_handler("on_client_disconnected")
            async def on_client_disconnected(agent, websocket):
                logger.info("Client disconnected")

            await runner.add_task(proxy)
    """

    def __init__(
        self,
        name: str,
        *,
        bus: TaskBus,
        websocket: WebSocket,
        agent_name: str,
        remote_agent_name: str,
        forward_messages: tuple[type[BusMessage], ...] = (),
        serializer: MessageSerializer | None = None,
    ):
        """Initialize the WebSocketProxyServerAgent.

        Args:
            name: Unique name for this agent.
            bus: The `TaskBus` for inter-agent communication.
            websocket: An accepted FastAPI/Starlette WebSocket connection.
            agent_name: Name of the local agent to route messages to/from.
                Only messages from this agent are forwarded to the client.
            remote_agent_name: Name of the agent on the remote client.
                Only outbound messages targeted at this agent are sent.
                Only inbound messages targeted at the local agent are accepted.
            forward_messages: Additional message types to forward from
                the local agent (e.g. ``(BusFrameMessage,)`` for frame
                routing). These are forwarded based on source agent name
                only, regardless of target.
            serializer: Serializer for bus messages. Defaults to
                `JSONMessageSerializer`.
        """
        super().__init__(name, bus=bus)
        self._ws = websocket
        self._agent_name = agent_name
        self._remote_agent_name = remote_agent_name
        self._forward_messages = forward_messages
        self._serializer = serializer or JSONMessageSerializer()
        self._receive_task: asyncio.Task | None = None

        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    async def cleanup(self):
        """Cancel the receive loop task and release resources."""
        await super().cleanup()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def on_ready(self) -> None:
        """Start receiving messages from the WebSocket and watch the local agent."""
        await super().on_ready()

        logger.debug(f"Agent '{self}': WebSocket proxy server ready")

        await self._call_event_handler("on_client_connected", self._ws)

        self._receive_task = self.create_task(self._receive_loop(), f"{self.name}::ws_receive")

        # Schedule task right away.
        await asyncio.sleep(0)

        # Watch the local agent so we can notify the remote side when it's ready
        await self.watch_task(self._agent_name)

    async def on_task_ready(self, data: TaskReadyData) -> None:
        """Notify the remote client that the local agent is ready."""
        if not self._ws:
            return

        if data.agent_name != self._agent_name:
            return

        logger.debug(f"Agent '{self}': local agent '{self._agent_name}' ready, notifying remote")

        try:
            msg = BusTaskRegistryMessage(
                source=self.name,
                runner=data.runner,
                agents=[TaskRegistryEntry(name=self._agent_name)],
            )

            await self._send_ws(msg)
        except Exception:
            logger.exception(f"Agent '{self}': failed to send registry to remote")

    async def on_bus_message(self, message: BusMessage) -> None:
        """Forward messages from the local agent to the remote client.

        Args:
            message: The bus message to process.
        """
        await super().on_bus_message(message)

        if not self._ws:
            return

        if isinstance(message, BusLocalMessage):
            return

        if message.source != self._agent_name:
            return

        # Forward targeted messages from the local agent to the remote agent
        if message.target == self._remote_agent_name:
            await self._send_ws(message)
        # Forward additional message types from the local agent
        elif isinstance(message, self._forward_messages):
            await self._send_ws(message)

    async def _stop(self) -> None:
        """Close the WebSocket connection and stop."""
        if self._ws and self._ws.client_state == WebSocketState.CONNECTED:
            await self._ws.close()
            logger.debug(f"Agent '{self}': WebSocket connection closed")
        await super()._stop()

    async def _send_ws(self, message: BusMessage) -> None:
        """Serialize and send a message over the WebSocket."""
        if not self._ws:
            return
        try:
            data = self._serializer.serialize(message)
            await self._ws.send_bytes(data)
            logger.trace(f"Agent '{self}': sent {message}")
        except (WebSocketDisconnect, Exception):
            logger.warning(f"Agent '{self}': connection closed, stopping forwarding")
            ws = self._ws
            self._ws = None
            await self._call_event_handler("on_client_disconnected", ws)

    async def _receive_loop(self) -> None:
        """Read messages from the WebSocket and put them on the local bus."""
        try:
            while True:
                data = await self._ws.receive_bytes()
                try:
                    message = self._serializer.deserialize(data)
                    if not message:
                        continue

                    # Accept additional message types (e.g. BusFrameMessage)
                    if self._forward_messages and isinstance(message, self._forward_messages):
                        logger.trace(f"Agent '{self}': received {message} from client")
                        await self.send_message(message)
                        continue

                    # Only accept other messages targeted at the local agent
                    if message.target != self._agent_name:
                        logger.warning(
                            f"Agent '{self}': dropped inbound message with "
                            f"unexpected target '{message.target}'"
                        )
                        continue

                    logger.trace(f"Agent '{self}': received {message} from client")
                    await self.send_message(message)
                except Exception:
                    logger.exception(f"Agent '{self}': failed to deserialize client message")
        except WebSocketDisconnect:
            logger.warning(f"Agent '{self}': client disconnected")
            ws = self._ws
            self._ws = None
            await self._call_event_handler("on_client_disconnected", ws)
        except asyncio.CancelledError:
            pass
