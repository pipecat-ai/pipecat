#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for WebsocketService reconnection and lifecycle behavior."""

import asyncio
import base64
import hashlib
import io
import time
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from pipecat.frames.frames import ErrorFrame
from pipecat.services.websocket_service import (
    WS_CLOSE_TIMEOUT,
    WebsocketService,
    _BoundedCloseConnection,
)

# Magic value RFC 6455 requires when deriving the handshake accept header.
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# The _no_sleep fixture below stubs out asyncio.sleep for the whole module, so
# tests that need real elapsed time use this reference instead.
_real_sleep = asyncio.sleep


class ConcreteWebsocketService(WebsocketService):
    """Minimal concrete implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._receive_messages_impl: AsyncMock | None = None

    async def _connect_websocket(self):
        pass

    async def _disconnect_websocket(self):
        pass

    async def _receive_messages(self):
        if self._receive_messages_impl:
            await self._receive_messages_impl()


@pytest.fixture
def service():
    return ConcreteWebsocketService()


@pytest.fixture
def report_error():
    return AsyncMock()


@pytest.fixture(autouse=True)
def _no_sleep():
    """Patch asyncio.sleep globally to avoid real backoff waits."""
    with patch("pipecat.services.websocket_service.asyncio.sleep", new_callable=AsyncMock):
        yield


# ---------------------------------------------------------------------------
# Receive loop — how each exception type is handled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_closed_ok_exits_cleanly(service, report_error):
    """ConnectionClosedOK exits the loop with no error and no reconnection."""
    service._receive_messages_impl = AsyncMock(
        side_effect=ConnectionClosedOK(Close(1000, "Normal closure"), None)
    )
    service._try_reconnect = AsyncMock()

    await service._receive_task_handler(report_error)

    report_error.assert_not_called()
    service._try_reconnect.assert_not_called()


@pytest.mark.asyncio
async def test_connection_closed_error_triggers_reconnect(service, report_error):
    """ConnectionClosedError triggers reconnection; loop continues after success."""
    call_count = 0

    async def fail_then_exit():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionClosedError(Close(1006, "Abnormal closure"), None)
        service._disconnecting = True

    service._receive_messages_impl = AsyncMock(side_effect=fail_then_exit)
    service._try_reconnect = AsyncMock(return_value=True)

    await service._receive_task_handler(report_error)

    assert call_count == 2
    service._try_reconnect.assert_called_once()


@pytest.mark.asyncio
async def test_graceful_server_close_triggers_reconnect(service, report_error):
    """Normal return from _receive_messages (server close frame) triggers reconnection."""
    call_count = 0

    async def return_then_exit():
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            service._disconnecting = True

    service._receive_messages_impl = AsyncMock(side_effect=return_then_exit)
    service._try_reconnect = AsyncMock(return_value=True)

    await service._receive_task_handler(report_error)

    assert call_count == 2
    service._try_reconnect.assert_called_once()


@pytest.mark.asyncio
async def test_general_exception_triggers_reconnect(service, report_error):
    """A general exception in _receive_messages triggers reconnection."""
    call_count = 0

    async def fail_then_exit():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("something broke")
        service._disconnecting = True

    service._receive_messages_impl = AsyncMock(side_effect=fail_then_exit)
    service._try_reconnect = AsyncMock(return_value=True)

    await service._receive_task_handler(report_error)

    assert call_count == 2
    service._try_reconnect.assert_called_once()


# ---------------------------------------------------------------------------
# Exponential backoff — server unreachable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_succeeds_on_later_attempt(service, report_error):
    """_try_reconnect retries and succeeds on a later attempt."""
    service._reconnect_websocket = AsyncMock(
        side_effect=[ConnectionError("fail"), ConnectionError("fail"), True]
    )

    result = await service._try_reconnect(report_error=report_error)

    assert result is True
    assert service._reconnect_websocket.call_count == 3


@pytest.mark.asyncio
async def test_reconnect_exhausted_emits_error(service, report_error):
    """Exhausting all retries returns False and emits an ErrorFrame."""
    service._reconnect_websocket = AsyncMock(side_effect=ConnectionError("Connection refused"))

    result = await service._try_reconnect(report_error=report_error)

    assert result is False
    assert service._reconnect_websocket.call_count == 3
    final_error = report_error.call_args_list[-1][0][0]
    assert isinstance(final_error, ErrorFrame)
    assert "Connection refused" in final_error.error


@pytest.mark.asyncio
async def test_reconnect_exhausted_when_connect_does_not_raise(service, report_error):
    """A non-raising failed connect is treated as a failed reconnect attempt."""
    result = await service._try_reconnect(report_error=report_error)

    assert result is False
    assert report_error.call_count == 4
    final_error = report_error.call_args_list[-1][0][0]
    assert isinstance(final_error, ErrorFrame)
    assert "websocket reconnection failed verification" in final_error.error


# ---------------------------------------------------------------------------
# Quick failure detection — accept then immediately close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quick_failures_emit_error(service, report_error):
    """Connections failing immediately after establishment emit error after 3 cycles."""
    call_count = 0

    async def fail_immediately():
        nonlocal call_count
        call_count += 1
        raise ConnectionClosedError(Close(1008, "Invalid API key"), None)

    service._receive_messages_impl = AsyncMock(side_effect=fail_immediately)
    service._try_reconnect = AsyncMock(return_value=True)

    await service._receive_task_handler(report_error)

    assert call_count == service._quick_failure_tracker.max_consecutive_failures
    report_error.assert_called_once()
    error_frame = report_error.call_args[0][0]
    assert isinstance(error_frame, ErrorFrame)
    assert "failed 3 times immediately after connecting" in error_frame.error


@pytest.mark.asyncio
async def test_stable_connection_resets_quick_failure_counter(service, report_error):
    """A stable connection resets the quick failure counter; needs 3 new failures to trigger."""
    call_count = 0

    async def always_fail():
        nonlocal call_count
        call_count += 1
        raise ConnectionClosedError(Close(1006, "Abnormal closure"), None)

    service._receive_messages_impl = AsyncMock(side_effect=always_fail)
    service._try_reconnect = AsyncMock(return_value=True)

    base_time = 1000.0
    time_values = iter(
        [
            # Call 1: set _last_connect_time, check in _maybe_try_reconnect (quick) -> count=1
            base_time,
            base_time,
            # Call 2: quick -> count=2
            base_time + 1.0,
            base_time + 1.0,
            # Call 3: stable (10s elapsed) -> count=0
            base_time + 2.0,
            base_time + 12.0,
            # Call 4: quick -> count=1
            base_time + 13.0,
            base_time + 13.0,
            # Call 5: quick -> count=2
            base_time + 14.0,
            base_time + 14.0,
            # Call 6: quick -> count=3 -> error emitted, loop stops
            base_time + 15.0,
            base_time + 15.0,
        ]
    )

    with patch("pipecat.services.websocket_service.time") as mock_time:
        mock_time.monotonic = lambda: next(time_values)
        await service._receive_task_handler(report_error)

    assert call_count == 6
    report_error.assert_called_once()
    assert isinstance(report_error.call_args[0][0], ErrorFrame)


# ---------------------------------------------------------------------------
# Lifecycle and guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_prevents_reconnection(service, report_error):
    """After _disconnect(), errors exit the loop without reconnecting or emitting errors."""
    await service._disconnect()

    service._receive_messages_impl = AsyncMock(
        side_effect=ConnectionClosedError(Close(1006, "Abnormal closure"), None)
    )
    service._try_reconnect = AsyncMock()

    await service._receive_task_handler(report_error)

    report_error.assert_not_called()
    service._try_reconnect.assert_not_called()


@pytest.mark.asyncio
async def test_connect_resets_state(service):
    """_connect() resets _disconnecting and the quick-failure tracker."""
    service._disconnecting = True
    service._quick_failure_tracker.count = 5

    await service._connect()

    assert service._disconnecting is False
    assert service._quick_failure_tracker.count == 0


# ---------------------------------------------------------------------------
# Close timeout — bounding the closing handshake
# ---------------------------------------------------------------------------


@pytest.fixture
def connect_mock():
    """Patch the underlying websockets connect() and capture its kwargs."""
    with patch(
        "pipecat.services.websocket_service.websocket_connect", new_callable=AsyncMock
    ) as mock:
        yield mock


@pytest.mark.asyncio
async def test_websocket_connect_applies_default_close_timeout(service, connect_mock):
    """Connections get the default close timeout without the caller asking."""
    await service._websocket_connect("wss://example.test")

    assert connect_mock.await_args.kwargs["close_timeout"] == WS_CLOSE_TIMEOUT


@pytest.mark.asyncio
async def test_websocket_connect_honors_constructor_override(connect_mock):
    """ws_close_timeout passed at construction reaches the connection."""
    service = ConcreteWebsocketService(ws_close_timeout=7.5)

    await service._websocket_connect("wss://example.test")

    assert connect_mock.await_args.kwargs["close_timeout"] == 7.5


@pytest.mark.asyncio
async def test_websocket_connect_honors_per_call_override(service, connect_mock):
    """An explicit close_timeout wins over the service default.

    Used by services whose peer never acknowledges the closing handshake.
    """
    await service._websocket_connect("wss://example.test", close_timeout=0)

    assert connect_mock.await_args.kwargs["close_timeout"] == 0


@pytest.mark.asyncio
async def test_websocket_connect_forwards_arguments(service, connect_mock):
    """The URI and caller kwargs are passed through untouched."""
    headers = {"Authorization": "Bearer token"}

    await service._websocket_connect("wss://example.test", additional_headers=headers)

    assert connect_mock.await_args.args == ("wss://example.test",)
    assert connect_mock.await_args.kwargs["additional_headers"] is headers


@pytest.mark.asyncio
async def test_websocket_connect_installs_bounded_close_connection(service, connect_mock):
    """Connections are created as the class that reports an overrunning close."""
    await service._websocket_connect("wss://example.test")

    assert connect_mock.await_args.kwargs["create_connection"] is _BoundedCloseConnection


@pytest.fixture
def log_sink():
    """Capture loguru output for the duration of a test."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="DEBUG", format="{message}")
    try:
        yield sink
    finally:
        logger.remove(handler_id)


@pytest.mark.asyncio
async def test_bounded_close_logs_when_handshake_overruns(log_sink):
    """An unacknowledged close is logged so a silent teardown cost leaves a trace."""
    conn = _BoundedCloseConnection.__new__(_BoundedCloseConnection)
    conn.close_timeout = 0.05

    async def slow_close(self, code=1000, reason=""):
        await _real_sleep(0.1)

    with patch.object(ClientConnection, "close", slow_close):
        await conn.close()

    assert "did not acknowledge the websocket close" in log_sink.getvalue()


@pytest.mark.asyncio
async def test_bounded_close_silent_when_handshake_completes(log_sink):
    """A clean close logs nothing."""
    conn = _BoundedCloseConnection.__new__(_BoundedCloseConnection)
    conn.close_timeout = 5.0

    async def fast_close(self, code=1000, reason=""):
        return None

    with patch.object(ClientConnection, "close", fast_close):
        await conn.close()

    assert "did not acknowledge the websocket close" not in log_sink.getvalue()


@pytest.mark.asyncio
async def test_bounded_close_against_unresponsive_peer(log_sink):
    """End to end: a peer that never acknowledges the close is bounded and logged.

    Serves a raw WebSocket handshake and then goes silent, which is the condition
    that makes the closing handshake overrun.
    """
    handshake_done = asyncio.Event()

    async def deaf_peer(reader, writer):
        request = await reader.readuntil(b"\r\n\r\n")
        key = next(
            line.split(":", 1)[1].strip()
            for line in request.decode().split("\r\n")
            if line.lower().startswith("sec-websocket-key:")
        )
        accept = base64.b64encode(
            hashlib.sha1((key + _WS_GUID).encode()).digest()  # noqa: S324 - required by RFC 6455
        ).decode()
        writer.write(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            b"Sec-WebSocket-Accept: " + accept.encode() + b"\r\n\r\n"
        )
        await writer.drain()
        await handshake_done.wait()

    server = await asyncio.start_server(deaf_peer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    service = ConcreteWebsocketService(ws_close_timeout=0.3)
    try:
        websocket = await service._websocket_connect(f"ws://127.0.0.1:{port}", ping_interval=None)
        assert isinstance(websocket, _BoundedCloseConnection)

        started = time.monotonic()
        await websocket.close()
        elapsed = time.monotonic() - started

        # Bounded by ws_close_timeout rather than the websockets default of 10s,
        # and 1006 confirms the peer never sent its close frame.
        assert 0.3 <= elapsed < 3.0
        assert websocket.close_code == 1006
        assert "did not acknowledge the websocket close" in log_sink.getvalue()
    finally:
        handshake_done.set()
        server.close()
