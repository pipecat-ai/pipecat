#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for :class:`pipecat.transports.websocket.rtvi_client.RTVIClientTransport`.

``TestRTVIClientTransportHandshake`` covers the RTVI-specific bits added on top of
the WebSocket client transport (the ``client-ready`` handshake message and
``bot-ready`` detection). ``TestRTVIClientTransportIntegration`` runs the transport
in a real pipeline against a tiny RTVI WebSocket server and asserts the bot's
server messages arrive as raw messages, in order and downstream only.
"""

import asyncio
import base64
import json
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import websockets

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    InputTransportMessageFrame,
    OutputAudioRawFrame,
)
from pipecat.pipeline.capabilities import BotCapabilities
from pipecat.serializers.rtvi_client import RTVIClientSerializer
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.websocket.client import WebsocketClientParams
from pipecat.transports.websocket.rtvi_client import RTVIClientTransport


def _rtvi(msg_type: str, data: dict | None = None) -> str:
    return json.dumps({"label": RTVI.MESSAGE_LABEL, "type": msg_type, "id": "x", "data": data})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class TestRTVIClientTransportHandshake(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.transport = RTVIClientTransport("ws://localhost:0")

    def test_defaults_to_rtvi_client_serializer(self):
        self.assertIsInstance(self.transport._params.serializer, RTVIClientSerializer)

    async def test_raw_audio_carries_bare_pcm(self):
        """The audio write path sends the PCM as given: no WAV header on the wire."""
        pcm = bytes(range(256)) * 4
        output = self.transport.output()
        output._session = SimpleNamespace(is_closing=False, is_connected=True, send=AsyncMock())
        output._sample_rate = 16000

        self.assertTrue(
            await output.write_audio_frame(
                OutputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)
            )
        )

        sent = json.loads(output._session.send.call_args.args[0])
        self.assertEqual(sent["type"], "raw-audio")
        self.assertEqual(base64.b64decode(sent["data"]["base64Audio"]), pcm)
        self.assertEqual(sent["data"]["sampleRate"], 16000)
        self.assertEqual(sent["data"]["numChannels"], 1)

    async def test_send_client_ready_message(self):
        self.transport._session.send = AsyncMock()
        await self.transport._send_client_ready()

        self.transport._session.send.assert_awaited_once()
        sent = json.loads(self.transport._session.send.call_args.args[0])
        self.assertEqual(sent["label"], RTVI.MESSAGE_LABEL)
        self.assertEqual(sent["type"], "client-ready")
        self.assertEqual(sent["data"]["version"], RTVI.PROTOCOL_VERSION)
        self.assertIn("library", sent["data"]["about"])

    def test_is_bot_ready(self):
        bot_ready = json.dumps(
            {
                "label": RTVI.MESSAGE_LABEL,
                "type": "bot-ready",
                "id": "1",
                "data": {"version": "2.0.0"},
            }
        )
        not_ready = json.dumps({"label": RTVI.MESSAGE_LABEL, "type": "bot-llm-text"})
        no_label = json.dumps({"type": "bot-ready"})

        self.assertTrue(RTVIClientTransport._is_bot_ready(bot_ready))
        self.assertFalse(RTVIClientTransport._is_bot_ready(not_ready))
        self.assertFalse(RTVIClientTransport._is_bot_ready(no_label))
        self.assertFalse(RTVIClientTransport._is_bot_ready("not json"))

    def test_capabilities(self):
        def bot_ready(data):
            return json.dumps({"label": RTVI.MESSAGE_LABEL, "type": "bot-ready", "data": data})

        reported = RTVIClientTransport._capabilities(
            bot_ready({"version": "2.0.0", "capabilities": {"audio_in": False, "audio_out": True}})
        )

        self.assertEqual(reported, BotCapabilities(audio_in=False, audio_out=True))
        self.assertIsNone(RTVIClientTransport._capabilities(bot_ready({"version": "1.0.0"})))
        self.assertIsNone(RTVIClientTransport._capabilities(bot_ready(None)))
        self.assertIsNone(
            RTVIClientTransport._capabilities(bot_ready({"capabilities": {"audio_in": "maybe"}}))
        )

    async def test_bot_ready_flag_set_on_message(self):
        self.assertFalse(self.transport.bot_ready)
        msg = json.dumps(
            {
                "label": RTVI.MESSAGE_LABEL,
                "type": "bot-ready",
                "id": "1",
                "data": {"version": "2.0.0"},
            }
        )
        await self.transport._on_message(None, msg)
        self.assertTrue(self.transport.bot_ready)


class _BotServer:
    """A tiny RTVI server: on client-ready, reply bot-ready then a scripted turn."""

    def __init__(self, port: int, replies: list[str]):
        self.port = port
        self._replies = replies
        self.received: list[dict] = []
        self._server = None

    async def _handler(self, ws):
        async for raw in ws:
            self.received.append(json.loads(raw))
            if json.loads(raw).get("type") == "client-ready":
                await ws.send(_rtvi("bot-ready", {"version": RTVI.PROTOCOL_VERSION}))
                for reply in self._replies:
                    await ws.send(reply)

    async def __aenter__(self):
        self._server = await websockets.serve(self._handler, "localhost", self.port)
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://localhost:{self.port}"


class TestRTVIClientTransportIntegration(unittest.IsolatedAsyncioTestCase):
    """Run the transport in a pipeline against a real RTVI server."""

    async def test_bot_messages_arrive_as_raw_messages(self):
        replies = [
            _rtvi("bot-llm-started"),
            _rtvi("bot-llm-text", {"text": "Paris"}),
            _rtvi("bot-llm-stopped"),
            _rtvi(
                "user-transcription",
                {"text": "hello", "user_id": "u", "timestamp": "t", "final": True},
            ),
        ]
        async with _BotServer(_free_port(), replies) as server:
            transport = RTVIClientTransport(server.url)
            down, up = await run_test(
                transport.input(),
                frames_to_send=[SleepFrame(sleep=0.5)],  # let connect/handshake/replies flow
                start_timeout=5.0,  # the input transport connects before the pipeline starts
            )

        # The bot's reports arrived as raw messages, in the order it sent them.
        reports = [f.message for f in down if isinstance(f, InputTransportMessageFrame)]
        self.assertEqual(
            [m["type"] for m in reports],
            [
                "bot-ready",
                "bot-llm-started",
                "bot-llm-text",
                "bot-llm-stopped",
                "user-transcription",
            ],
        )
        self.assertEqual(reports[2]["data"]["text"], "Paris")
        # Downstream only: the worker's RTVIProcessor, ahead of the pipeline,
        # would take them for a client's messages to it.
        self.assertFalse(any(isinstance(f, InputTransportMessageFrame) for f in up))

        # The transport completed the handshake (the server saw client-ready).
        self.assertTrue(any(m.get("type") == "client-ready" for m in server.received))
        self.assertTrue(transport.bot_ready)

    async def test_bot_ready_is_handled_before_the_bots_next_message(self):
        # The bot reports right after bot-ready; the handler, checking what the
        # bot reported, runs before any of that reaches the pipeline.
        order: list[str] = []

        class _Recording(RTVIClientSerializer):
            async def deserialize(self, data):
                order.append(json.loads(data)["type"])
                return await super().deserialize(data)

        async with _BotServer(_free_port(), [_rtvi("bot-llm-started")]) as server:
            transport = RTVIClientTransport(
                server.url, WebsocketClientParams(serializer=_Recording())
            )

            @transport.event_handler("on_bot_ready")
            async def _on_bot_ready(_transport):
                await asyncio.sleep(0.1)
                order.append("on_bot_ready")

            await run_test(
                transport.input(), frames_to_send=[SleepFrame(sleep=0.5)], start_timeout=5.0
            )

        self.assertEqual(order, ["on_bot_ready", "bot-ready", "bot-llm-started"])


if __name__ == "__main__":
    unittest.main()
