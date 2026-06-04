#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from unittest.mock import AsyncMock, PropertyMock

from starlette.websockets import WebSocketState

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


class TestSendDisconnectRace(unittest.IsolatedAsyncioTestCase):
    """Tests for the race condition in issue #3912.

    When the remote side disconnects while send() is in flight, send() should
    not set _closing = True, because that flag means "we initiated the close."
    Setting it from send() prevents the receive loop from firing
    on_client_disconnected, which can cause the pipeline to hang.
    """

    def _make_client(self, mock_ws):
        callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=AsyncMock(),
            on_client_disconnected=AsyncMock(),
            on_session_timeout=AsyncMock(),
        )
        client = FastAPIWebsocketClient(mock_ws, callbacks)
        return client, callbacks

    async def test_send_disconnect_does_not_set_closing(self):
        """send() should not set _closing when the remote side disconnects."""
        mock_ws = AsyncMock()
        type(mock_ws).client_state = PropertyMock(return_value=WebSocketState.CONNECTED)
        type(mock_ws).application_state = PropertyMock(return_value=WebSocketState.DISCONNECTED)
        mock_ws.send_bytes.side_effect = Exception("connection closed")

        client, _ = self._make_client(mock_ws)

        await client.send(b"audio data")

        self.assertFalse(client.is_closing)

    async def test_send_suppressed_after_disconnect(self):
        """After a failed send, _can_send() returns False via application_state.

        Simulates real Starlette behavior: application_state starts CONNECTED,
        transitions to DISCONNECTED when send_bytes raises (Starlette does this
        internally on OSError before re-raising as WebSocketDisconnect).
        """
        mock_ws = AsyncMock()
        type(mock_ws).client_state = PropertyMock(return_value=WebSocketState.CONNECTED)

        # application_state transitions from CONNECTED → DISCONNECTED on send failure
        app_state = {"state": WebSocketState.CONNECTED}
        type(mock_ws).application_state = PropertyMock(side_effect=lambda: app_state["state"])

        def fail_and_transition(data):
            app_state["state"] = WebSocketState.DISCONNECTED
            raise Exception("connection closed")

        mock_ws.send_bytes.side_effect = fail_and_transition

        client, _ = self._make_client(mock_ws)

        # First send: _can_send() passes (app_state CONNECTED), send_bytes raises,
        # Starlette sets app_state to DISCONNECTED
        await client.send(b"audio data")
        # Second send: _can_send() returns False (app_state now DISCONNECTED)
        await client.send(b"more audio")

        # send_bytes was only called once (the first attempt)
        mock_ws.send_bytes.assert_called_once()

    async def test_disconnect_callback_fires_when_send_races_receive(self):
        """Regression test for issue #3912.

        The receive loop is blocked waiting for the next message. Meanwhile,
        send() is called and hits an exception because the remote side closed.
        Then the receive loop unblocks and sees the disconnect.

        on_client_disconnected must still fire, because the remote side
        initiated the close — not us.
        """
        send_done = asyncio.Event()

        mock_ws = AsyncMock()
        type(mock_ws).client_state = PropertyMock(return_value=WebSocketState.CONNECTED)
        type(mock_ws).application_state = PropertyMock(return_value=WebSocketState.DISCONNECTED)
        mock_ws.send_bytes.side_effect = Exception("connection closed")

        # receive() blocks until send has completed, then returns disconnect.
        # This enforces the exact ordering that causes the bug.
        async def mock_receive():
            await send_done.wait()
            return {"type": "websocket.disconnect"}

        mock_ws.receive = mock_receive

        client, callbacks = self._make_client(mock_ws)

        # Simulate the _receive_messages logic from FastAPIWebsocketInputTransport
        async def receive_loop():
            try:
                async for _ in _WebSocketMessageIterator(mock_ws):
                    pass
            except Exception:
                pass
            if not client.is_closing:
                await client.trigger_client_disconnected()

        recv_task = asyncio.create_task(receive_loop())

        # Let the receive loop start and block on receive()
        await asyncio.sleep(0)

        # send() races — hits exception but does NOT set _closing
        await client.send(b"audio data")
        self.assertFalse(client.is_closing)

        # Unblock the receive loop — it sees the disconnect
        send_done.set()
        await recv_task

        # The callback fires because _closing was not poisoned by send()
        callbacks.on_client_disconnected.assert_called_once()

    async def test_send_text_disconnect_does_not_set_closing(self):
        """Same as test_send_disconnect_does_not_set_closing but with text data."""
        mock_ws = AsyncMock()
        type(mock_ws).client_state = PropertyMock(return_value=WebSocketState.CONNECTED)
        type(mock_ws).application_state = PropertyMock(return_value=WebSocketState.DISCONNECTED)
        mock_ws.send_text.side_effect = Exception("connection closed")

        client, _ = self._make_client(mock_ws)

        await client.send("text data")

        self.assertFalse(client.is_closing)


if __name__ == "__main__":
    unittest.main()
