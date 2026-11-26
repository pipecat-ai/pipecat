#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A Pipecat pipeline acting as an RTVI *client*.

`RTVIClientTransport` connects to a bot that runs an RTVI *server* transport
(e.g. the eval transport) and exchanges frames with it: the bot's RTVI server
messages reach the pipeline as raw messages, ``InputTransportMessageFrame``s
pushed downstream, and outgoing frames (audio, pre-built RTVI client messages)
are sent to the bot. It is the client-side counterpart to a bot
running an RTVI server transport, and the foundation for the eval harness and
eval simulations (see ``docs/design/eval-simulations.md``).

It wraps :class:`~pipecat.transports.websocket.client.WebsocketClientTransport`,
defaulting its serializer to
:class:`~pipecat.serializers.rtvi_client.RTVIClientSerializer` and adding the
``client-ready`` / ``bot-ready`` handshake.
"""

import json

from pydantic import ValidationError

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import InputTransportMessageFrame
from pipecat.pipeline.capabilities import BotCapabilities
from pipecat.serializers.rtvi_client import RTVIClientSerializer
from pipecat.transports.websocket.client import (
    WebsocketClientInputTransport,
    WebsocketClientParams,
    WebsocketClientTransport,
)


class RTVIClientInputTransport(WebsocketClientInputTransport):
    """Input transport that hands the bot's RTVI messages to the pipeline downstream only.

    A ``PipelineWorker`` puts its own ``RTVIProcessor`` ahead of the pipeline,
    and it takes any ``InputTransportMessageFrame`` reaching it for a client's
    message to it. The bot's messages must not reach it: it would answer them
    with errors sent back to the bot.
    """

    async def _push_transport_message(self, frame: InputTransportMessageFrame):
        """Push the bot's message downstream, past nothing upstream of the input.

        Args:
            frame: The bot's message, as the serializer deserialized it.
        """
        await self.push_frame(frame)


class RTVIClientTransport(WebsocketClientTransport):
    """WebSocket transport that speaks RTVI as a client to a bot.

    Event handlers available:

    - on_connected: the WebSocket connection was established
    - on_disconnected: the WebSocket connection was closed
    - on_bot_ready: the bot answered the ``client-ready`` handshake with ``bot-ready``.
      The handler runs before any later message from the bot is dispatched, so
      it acts on what the bot reported before it hears anything else from it.
      It runs in the loop that reads from the bot, so it must return quickly
      and must not wait for anything the bot sends next.

    Example::

        transport = RTVIClientTransport("ws://localhost:7860")

        @transport.event_handler("on_bot_ready")
        async def on_bot_ready(transport):
            ...
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
        # An RTVI raw-audio message carries bare PCM with its own sample rate and
        # channel count, so a WAV header is never valid on this wire.
        params.add_wav_header = False
        super().__init__(uri, params)

        self._bot_ready = False
        self._bot_capabilities: BotCapabilities | None = None
        # Run inline: the bot can send its first reports right after bot-ready,
        # and they must not reach the pipeline before the handler has run.
        self._register_event_handler("on_bot_ready", sync=True)

    def input(self) -> WebsocketClientInputTransport:
        """Get the input transport, which hands the bot's messages downstream only.

        Returns:
            The RTVI client input transport instance.
        """
        if not self._input:
            self._input = RTVIClientInputTransport(self, self._session, self._params)
        return self._input

    @property
    def bot_ready(self) -> bool:
        """Whether the bot has answered the handshake with ``bot-ready``."""
        return self._bot_ready

    @property
    def bot_capabilities(self) -> BotCapabilities | None:
        """What the bot reported it does in ``bot-ready``.

        ``None`` before ``bot-ready``, and when the bot reported nothing
        readable, as bots on Pipecat before 1.13 do.
        """
        return self._bot_capabilities

    async def _on_connected(self, websocket):
        """On connect, fire the user's handler and start the RTVI handshake."""
        await super()._on_connected(websocket)
        await self._send_client_ready()

    async def _on_message(self, websocket, message):
        """Watch for ``bot-ready`` to complete the handshake, then dispatch normally."""
        if not self._bot_ready and self._is_bot_ready(message):
            self._bot_ready = True
            self._bot_capabilities = self._capabilities(message)
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

    @staticmethod
    def _capabilities(bot_ready) -> BotCapabilities | None:
        """Return the capabilities a ``bot-ready`` message reports, or ``None`` if none."""
        data = json.loads(bot_ready).get("data")
        capabilities = data.get("capabilities") if isinstance(data, dict) else None
        if capabilities is None:
            return None
        try:
            return BotCapabilities.model_validate(capabilities)
        except ValidationError:
            return None
