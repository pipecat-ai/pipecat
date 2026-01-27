#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock

from pipecat.transports.websocket.fastapi import _WebSocketMessageIterator


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


if __name__ == "__main__":
    unittest.main()
