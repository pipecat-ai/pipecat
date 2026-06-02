#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""WebSocket server transport for the eval harness.

A thin subclass of :class:`~pipecat.transports.websocket.server.WebsocketServerTransport`
that, when a client connects with ``?skip_tts=true``, pushes an
:class:`~pipecat.frames.frames.LLMConfigureOutputFrame` (``skip_tts=True``) into
the pipeline *before* firing ``on_client_connected``.

This is deliberately done at connect time rather than as a client message.
pipecat processes frames strictly in order, and a bot that greets on
``on_client_connected`` queues its greeting ``LLMRunFrame`` there. A config frame
sent afterwards (as a client message) would sit *behind* that greeting and only
take effect once the greeting's whole inference finished — too late to keep it
silent. Pushing the config before the ``on_client_connected`` event guarantees it
reaches the LLM ahead of the greeting, so a text-mode eval's greeting is silent.

The harness sets the query parameter per connection based on the scenario's
``bot_audio`` (see :class:`~pipecat.evals.harness.EvalSession`).
"""

from urllib.parse import parse_qs, urlsplit

from loguru import logger

from pipecat.frames.frames import LLMConfigureOutputFrame
from pipecat.transports.websocket.server import WebsocketServerTransport

SKIP_TTS_QUERY_PARAM = "skip_tts"


def _connection_wants_skip_tts(websocket) -> bool:
    """Whether the client's connection URL requested ``skip_tts``."""
    # websockets exposes the request target as ``.path`` (legacy) or
    # ``.request.path`` (newer); both include the query string.
    path = getattr(websocket, "path", None)
    if path is None:
        request = getattr(websocket, "request", None)
        path = getattr(request, "path", "") if request is not None else ""
    values = parse_qs(urlsplit(path or "").query).get(SKIP_TTS_QUERY_PARAM, [])
    return bool(values) and values[0].strip().lower() in ("1", "true", "yes")


class EvalWebsocketServerTransport(WebsocketServerTransport):
    """WebSocket server transport used by the eval harness.

    Identical to the base transport except that a ``?skip_tts=true`` connection
    silences the bot's output for that session, including any on-connect greeting
    (see the module docstring for why this must happen before
    ``on_client_connected``).
    """

    async def _on_client_connected(self, websocket):
        """Push a skip-TTS config (if requested) before the greeting, then proceed."""
        if self._input is not None and _connection_wants_skip_tts(websocket):
            logger.debug(f"{self}: eval client requested skip_tts; configuring LLM output")
            await self._input.push_frame(LLMConfigureOutputFrame(skip_tts=True))
        await super()._on_client_connected(websocket)
