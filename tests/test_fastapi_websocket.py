#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.websockets import WebSocketState

from pipecat.frames.frames import StartFrame
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketCallbacks,
    FastAPIWebsocketClient,
    _WebSocketMessageIterator,
)


class TestWebSocketMessageIterator(unittest.IsolatedAsyncioTestCase):
    async def test_yields_binary_message(self):
        mock_websocket = AsyncMock()
        mock_websocket.receive.side_effect = [
            {"type": "websocket.receive", "bytes": b"binary data", "text": None},
            {"type": "websocket.disconnect"},
        ]

        iterator = _WebSocketMessageIterator(mock_websocket)
        messages = [msg async for msg in iterator]

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0], b"binary data")

    async def test_yields_text_message(self):
        mock_websocket = AsyncMock()
        mock_websocket.receive.side_effect = [
            {"type": "websocket.receive", "bytes": None, "text": "text data"},
            {"type": "websocket.disconnect"},
        ]

        iterator = _WebSocketMessageIterator(mock_websocket)
        messages = [msg async for msg in iterator]

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0], "text data")

    async def test_yields_mixed_messages(self):
        mock_websocket = AsyncMock()
        mock_websocket.receive.side_effect = [
            {"type": "websocket.receive", "bytes": b"binary", "text": None},
            {"type": "websocket.receive", "bytes": None, "text": "text"},
            {"type": "websocket.receive", "bytes": b"more binary", "text": None},
            {"type": "websocket.disconnect"},
        ]

        iterator = _WebSocketMessageIterator(mock_websocket)
        messages = [msg async for msg in iterator]

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0], b"binary")
        self.assertEqual(messages[1], "text")
        self.assertEqual(messages[2], b"more binary")

    async def test_stops_on_disconnect(self):
        mock_websocket = AsyncMock()
        mock_websocket.receive.side_effect = [
            {"type": "websocket.disconnect"},
        ]

        iterator = _WebSocketMessageIterator(mock_websocket)
        messages = [msg async for msg in iterator]

        self.assertEqual(len(messages), 0)


class TestFastAPIWebsocketClient:
    """Tests for FastAPIWebsocketClient focusing on _closing flag behaviour."""

    def _make_client(self, ws=None):
        if ws is None:
            ws = MagicMock()
            ws.client_state = WebSocketState.CONNECTED
            ws.application_state = WebSocketState.CONNECTED
        callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=AsyncMock(),
            on_client_disconnected=AsyncMock(),
            on_session_timeout=AsyncMock(),
        )
        return FastAPIWebsocketClient(ws, callbacks)

    @pytest.mark.asyncio
    async def test_send_exception_does_not_set_closing(self):
        """When send raises and the socket is already DISCONNECTED, _closing must stay False."""
        ws = MagicMock()
        ws.client_state = WebSocketState.CONNECTED
        ws.application_state = WebSocketState.DISCONNECTED
        ws.send_bytes = AsyncMock(side_effect=RuntimeError("connection lost"))

        client = self._make_client(ws)
        assert not client.is_closing

        await client.send(b"hello")

        # _closing must remain False so the receive loop can still fire on_client_disconnected
        assert not client.is_closing

    @pytest.mark.asyncio
    async def test_disconnect_sets_closing(self):
        """Calling disconnect() must set _closing to True."""
        ws = MagicMock()
        ws.client_state = WebSocketState.CONNECTED
        ws.application_state = WebSocketState.CONNECTED
        ws.close = AsyncMock()

        client = self._make_client(ws)
        await client.setup(MagicMock(spec=StartFrame))

        assert not client.is_closing

        await client.disconnect()

        assert client.is_closing


if __name__ == "__main__":
    unittest.main()
