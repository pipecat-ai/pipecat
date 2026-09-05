#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import contextlib
import io
import time
import unittest
from unittest.mock import AsyncMock, PropertyMock

from loguru import logger
from starlette.websockets import WebSocketState

from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketCallbacks,
    FastAPIWebsocketClient,
    FastAPIWebsocketOutputTransport,
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
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


class TestDisconnectCloseTimeout(unittest.IsolatedAsyncioTestCase):
    """Tests for issue #4528.

    ``disconnect()`` must not block indefinitely on a half-closed peer that
    never acknowledges the WebSocket close handshake (e.g. a telephony call
    already torn down on the provider's side). The close should be bounded by
    ``ws_close_timeout`` so pipeline shutdown can proceed.
    """

    def _make_client(self, mock_ws, ws_close_timeout=0.5):
        callbacks = FastAPIWebsocketCallbacks(
            on_client_connected=AsyncMock(),
            on_client_disconnected=AsyncMock(),
            on_session_timeout=AsyncMock(),
        )
        client = FastAPIWebsocketClient(mock_ws, callbacks, ws_close_timeout=ws_close_timeout)
        # setup() bumps the leave counter to 1; disconnect() decrements to 0
        # and then performs the close.
        client._leave_counter = 1
        return client, callbacks

    @contextlib.contextmanager
    def _capture_logs(self, level):
        """Capture loguru output (pipecat uses loguru, not stdlib logging)."""
        sink = io.StringIO()
        handler_id = logger.add(sink, level=level, format="{message}")
        try:
            yield sink
        finally:
            logger.remove(handler_id)

    async def test_disconnect_bounded_when_close_hangs(self):
        """disconnect() returns within ws_close_timeout if close() never completes."""
        never = asyncio.Event()

        async def hanging_close():
            await never.wait()  # peer never ACKs the close handshake

        mock_ws = AsyncMock()
        type(mock_ws).client_state = PropertyMock(return_value=WebSocketState.CONNECTED)
        mock_ws.close = hanging_close

        client, _ = self._make_client(mock_ws, ws_close_timeout=0.1)

        start = time.monotonic()
        with self._capture_logs("DEBUG") as logs:
            # wait_for is the regression guard: against the old unbounded code
            # disconnect() never returns, so this fails fast with TimeoutError
            # instead of hanging CI on the ~10s ASGI close-handshake timeout.
            await asyncio.wait_for(client.disconnect(), timeout=5.0)
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 2.0)
        self.assertTrue(client.is_closing)
        self.assertIn("WebSocket close exceeded", logs.getvalue())

        # The close task outlives the timeout; cancel it to clean up.
        client._close_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await client._close_task

    async def test_disconnect_completes_when_close_succeeds(self):
        """Happy path: a peer that ACKs the close lets disconnect() finish fast."""
        mock_ws = AsyncMock()
        type(mock_ws).client_state = PropertyMock(return_value=WebSocketState.CONNECTED)
        mock_ws.close = AsyncMock()

        client, _ = self._make_client(mock_ws, ws_close_timeout=5.0)

        await client.disconnect()
        await asyncio.sleep(0)  # let the done callback run

        mock_ws.close.assert_awaited_once()
        self.assertTrue(client.is_closing)
        self.assertTrue(client._close_task.done())

    async def test_disconnect_noop_when_other_holders_remain(self):
        """disconnect() only closes once the last holder leaves."""
        mock_ws = AsyncMock()
        type(mock_ws).client_state = PropertyMock(return_value=WebSocketState.CONNECTED)
        mock_ws.close = AsyncMock()

        client, _ = self._make_client(mock_ws)
        client._leave_counter = 2  # input + output both hold the client

        await client.disconnect()  # one leaves; one holder remains

        mock_ws.close.assert_not_called()
        self.assertFalse(client.is_closing)

    async def test_close_error_is_logged_not_raised(self):
        """An exception from close() is swallowed (logged), not propagated."""
        mock_ws = AsyncMock()
        type(mock_ws).client_state = PropertyMock(return_value=WebSocketState.CONNECTED)
        mock_ws.close = AsyncMock(side_effect=RuntimeError("already closed"))

        client, _ = self._make_client(mock_ws, ws_close_timeout=5.0)

        with self._capture_logs("ERROR") as logs:
            await client.disconnect()  # must not raise
            # The done callback runs via call_soon (not synchronously), so yield
            # once to let it consume and log the exception before we assert.
            await asyncio.sleep(0)

        self.assertTrue(client._close_task.done())
        self.assertIsInstance(client._close_task.exception(), RuntimeError)
        self.assertIn("exception while closing the websocket", logs.getvalue())

    async def test_transport_passes_ws_close_timeout_to_client(self):
        """The transport wires params.ws_close_timeout through to its client."""
        mock_ws = AsyncMock()
        params = FastAPIWebsocketParams(ws_close_timeout=1.25)

        transport = FastAPIWebsocketTransport(mock_ws, params)

        self.assertEqual(transport._client._ws_close_timeout, 1.25)


if __name__ == "__main__":
    unittest.main()


class TestAudioPacingWhenTheSerializerEmitsNothing(unittest.IsolatedAsyncioTestCase):
    """Tests for issue #5592.

    ``_write_audio_sleep`` is a clock: it advances ``_next_send_time`` by one send interval per
    call so the transport emulates an audio device. A stream resampler accumulates across calls
    and emits nothing for most of them, so the serializer returns no payload for a frame it
    accepted. Skipping the clock on those frames lets the output queue, and the
    ``TTSStoppedFrame`` behind it, drain faster than playout.
    """

    class _BufferingSerializer(FrameSerializer):
        """Emits a payload only every ``emit_every`` calls, as a stream resampler does."""

        def __init__(self, emit_every: int | None):
            super().__init__()
            self._emit_every = emit_every
            self.calls = 0

        async def serialize(self, frame):
            self.calls += 1
            if self._emit_every is None:
                return None
            return b"payload" if self.calls % self._emit_every == 0 else None

        async def deserialize(self, data):
            return None

    def _make_output(self, emit_every):
        client = AsyncMock()
        type(client).is_closing = PropertyMock(return_value=False)
        type(client).is_connected = PropertyMock(return_value=True)

        serializer = self._BufferingSerializer(emit_every)

        params = FastAPIWebsocketParams(audio_out_enabled=True, serializer=serializer)
        output = FastAPIWebsocketOutputTransport(
            transport=AsyncMock(), client=client, params=params
        )
        output._send_interval = 0.04
        output._next_send_time = 0
        return output

    async def _write(self, output, count):
        paced = 0

        async def counting_sleep():
            nonlocal paced
            paced += 1

        output._write_audio_sleep = counting_sleep

        results = []
        for _ in range(count):
            results.append(
                await output.write_audio_frame(
                    OutputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=8000, num_channels=1)
                )
            )
        return paced, results

    async def test_a_frame_with_no_payload_is_still_paced(self):
        output = self._make_output(None)

        paced, results = await self._write(output, 3)

        self.assertEqual(paced, 3, "the clock must advance for every frame the transport took")
        # The frame did not reach the wire on this call, and #5424 is right that the return
        # value should say so. Pacing and delivery are separate questions.
        self.assertEqual(results, [False, False, False])

    async def test_a_frame_that_is_sent_is_still_paced(self):
        output = self._make_output(1)

        paced, results = await self._write(output, 3)

        self.assertEqual(paced, 3)
        self.assertEqual(results, [True, True, True])

    async def test_every_frame_is_paced_when_the_resampler_emits_one_in_three(self):
        """The reported shape: most calls accumulate, one in N flushes a larger block."""
        output = self._make_output(3)

        paced, results = await self._write(output, 6)

        self.assertEqual(paced, 6, "all six frames were accepted, so all six must be paced")
        self.assertEqual(results, [False, False, True, False, False, True])
