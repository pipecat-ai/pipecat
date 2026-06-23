#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A Pipecat pipeline acting as an RTVI *client*.

`RTVIClientTransport` connects to a bot that runs an RTVI *server* transport
(e.g. the eval transport) and exchanges frames with it: the bot's RTVI server
messages arrive as pipeline frames, and outgoing frames (audio, pre-built RTVI
client messages) are sent to the bot. It is the client-side counterpart to a bot
running an RTVI server transport, and the foundation for the eval harness and
eval simulations (see ``docs/design/eval-simulations.md``).

It wraps :class:`~pipecat.transports.websocket.client.WebsocketClientTransport`,
defaulting its serializer to
:class:`~pipecat.serializers.rtvi_client.RTVIClientSerializer` and adding the
``client-ready`` / ``bot-ready`` handshake.
"""

import asyncio
import json

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.serializers.rtvi_client import RTVIClientSerializer
from pipecat.transports.websocket.client import (
    WebsocketClientParams,
    WebsocketClientTransport,
)


class RTVIClientTransport(WebsocketClientTransport):
    """WebSocket transport that speaks RTVI as a client to a bot.

    Event handlers available:

    - on_connected: the WebSocket connection was established
    - on_disconnected: the WebSocket connection was closed
    - on_bot_ready: the bot answered the ``client-ready`` handshake with ``bot-ready``

    Example::

        transport = RTVIClientTransport("ws://localhost:7860")
        ...
        await transport.wait_for_bot_ready()
    """

    def __init__(self, uri: str, params: WebsocketClientParams | None = None):
        """Initialize the RTVI client transport.

        Args:
            uri: The WebSocket URI of the bot's RTVI server transport.
            params: Optional transport parameters. The serializer defaults to
                :class:`RTVIClientSerializer` when not provided.
        """
        params = params or WebsocketClientParams()
        params.serializer = params.serializer or RTVIClientSerializer()
        super().__init__(uri, params)

        self._bot_ready_event = asyncio.Event()
        self._register_event_handler("on_bot_ready")

    @property
    def bot_ready(self) -> bool:
        """Whether the bot has answered the handshake with ``bot-ready``."""
        return self._bot_ready_event.is_set()

    async def wait_for_bot_ready(self, timeout: float = 10.0) -> None:
        """Block until the bot answers with ``bot-ready``.

        Args:
            timeout: Seconds to wait before raising.

        Raises:
            TimeoutError: If ``bot-ready`` doesn't arrive within ``timeout``.
        """
        await asyncio.wait_for(self._bot_ready_event.wait(), timeout=timeout)

    async def _on_connected(self, websocket):
        """On connect, fire the user's handler and start the RTVI handshake."""
        await super()._on_connected(websocket)
        await self._send_client_ready()

    async def _on_message(self, websocket, message):
        """Watch for ``bot-ready`` to complete the handshake, then dispatch normally."""
        if not self._bot_ready_event.is_set() and self._is_bot_ready(message):
            self._bot_ready_event.set()
            await self._call_event_handler("on_bot_ready")
        await super()._on_message(websocket, message)

    async def _send_client_ready(self) -> None:
        """Send the RTVI ``client-ready`` message that starts the handshake."""
        message = RTVI.Message(
            type="client-ready",
            id="client-ready",
            data=RTVI.ClientReadyData(
                version=RTVI.PROTOCOL_VERSION,
                about=RTVI.AboutClientData(library="pipecat"),
            ).model_dump(),
        )
        await self._session.send(message.model_dump_json())

    @staticmethod
    def _is_bot_ready(message) -> bool:
        """Return True if ``message`` is an RTVI ``bot-ready`` message."""
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return False
        return (
            isinstance(data, dict)
            and data.get("label") == RTVI.MESSAGE_LABEL
            and data.get("type") == "bot-ready"
        )
