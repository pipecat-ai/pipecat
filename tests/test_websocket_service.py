#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for WebsocketService reconnection and lifecycle behavior."""

from unittest.mock import AsyncMock, patch

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from pipecat.frames.frames import ErrorFrame
from pipecat.services.websocket_service import WebsocketService


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
async def test_reconnect_exhausted_emits_non_fatal_error(service, report_error):
    """Exhausting all retries returns False and emits a non-fatal ErrorFrame."""
    service._reconnect_websocket = AsyncMock(side_effect=ConnectionError("Connection refused"))

    result = await service._try_reconnect(report_error=report_error)

    assert result is False
    assert service._reconnect_websocket.call_count == 3
    final_error = report_error.call_args_list[-1][0][0]
    assert isinstance(final_error, ErrorFrame)
    assert final_error.fatal is False
    assert "Connection refused" in final_error.error


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

    assert call_count == service._MAX_CONSECUTIVE_QUICK_FAILURES
    report_error.assert_called_once()
    error_frame = report_error.call_args[0][0]
    assert isinstance(error_frame, ErrorFrame)
    assert error_frame.fatal is False
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
    error_frame = report_error.call_args[0][0]
    assert error_frame.fatal is False


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
    """_connect() resets _disconnecting and _quick_failure_count."""
    service._disconnecting = True
    service._quick_failure_count = 5

    await service._connect()

    assert service._disconnecting is False
    assert service._quick_failure_count == 0
