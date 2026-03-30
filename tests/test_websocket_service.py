#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for WebsocketService quick failure detection."""

import time
from unittest.mock import AsyncMock, patch

import pytest
from websockets.exceptions import ConnectionClosedError
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


@pytest.mark.asyncio
async def test_quick_failures_emit_error(service, report_error):
    """Connections that fail immediately after being established should emit an error
    after MAX_CONSECUTIVE_QUICK_FAILURES consecutive quick failures."""
    call_count = 0

    async def fail_immediately():
        nonlocal call_count
        call_count += 1
        raise ConnectionClosedError(Close(1008, "Invalid API key"), None)

    service._receive_messages_impl = AsyncMock(side_effect=fail_immediately)
    # Mock _try_reconnect to succeed (handshake passes but connection dies right away)
    service._try_reconnect = AsyncMock(return_value=True)

    await service._receive_task_handler(report_error)

    # Should have called _receive_messages MAX_RAPID_FAILURES times
    assert call_count == service._MAX_CONSECUTIVE_QUICK_FAILURES
    # Should have emitted a fatal error
    report_error.assert_called_once()
    error_frame = report_error.call_args[0][0]
    assert isinstance(error_frame, ErrorFrame)
    assert error_frame.fatal is False
    assert "failed 3 times immediately after connecting" in error_frame.error


@pytest.mark.asyncio
async def test_stable_connection_resets_quick_failure_counter(service, report_error):
    """A connection that survives beyond the threshold should reset the quick failure counter."""
    call_count = 0

    async def fail_then_stable_then_fail():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            # First two calls: quick failures
            raise ConnectionClosedError(Close(1006, "Abnormal closure"), None)
        elif call_count == 3:
            # Third call: simulate a stable connection by advancing time past threshold
            raise ConnectionClosedError(Close(1006, "Abnormal closure"), None)
        else:
            # Fourth and beyond: quick failures again
            raise ConnectionClosedError(Close(1006, "Abnormal closure"), None)

    service._receive_messages_impl = AsyncMock(side_effect=fail_then_stable_then_fail)
    service._try_reconnect = AsyncMock(return_value=True)

    # Patch time.monotonic to control timing
    base_time = 1000.0
    time_values = iter(
        [
            # Call 1: set _last_connect_time
            base_time,
            # Call 1: check in _maybe_try_reconnect (rapid: 0s elapsed)
            base_time,
            # Call 2: set _last_connect_time
            base_time + 1.0,
            # Call 2: check in _maybe_try_reconnect (rapid: 0s elapsed)
            base_time + 1.0,
            # Call 3: set _last_connect_time
            base_time + 2.0,
            # Call 3: check in _maybe_try_reconnect (stable: 10s elapsed)
            base_time + 12.0,
            # Call 4: set _last_connect_time
            base_time + 13.0,
            # Call 4: check in _maybe_try_reconnect (rapid: 0s elapsed)
            base_time + 13.0,
            # Call 5: set _last_connect_time
            base_time + 14.0,
            # Call 5: check in _maybe_try_reconnect (rapid: 0s elapsed)
            base_time + 14.0,
            # Call 6: set _last_connect_time
            base_time + 15.0,
            # Call 6: check in _maybe_try_reconnect (rapid: 0s elapsed)
            base_time + 15.0,
        ]
    )

    with patch("pipecat.services.websocket_service.time") as mock_time:
        mock_time.monotonic = lambda: next(time_values)

        await service._receive_task_handler(report_error)

    # After the stable connection (call 3), counter resets to 0.
    # Then calls 4, 5, 6 are quick failures (counter: 1, 2, 3) -> error emitted
    assert call_count == 6
    report_error.assert_called_once()
    error_frame = report_error.call_args[0][0]
    assert error_frame.fatal is False


@pytest.mark.asyncio
async def test_graceful_close_counts_toward_quick_failures(service, report_error):
    """A _receive_messages that returns normally (graceful close) should also count
    toward quick failures if it happens immediately."""
    call_count = 0

    async def return_immediately():
        nonlocal call_count
        call_count += 1

    service._receive_messages_impl = AsyncMock(side_effect=return_immediately)
    service._try_reconnect = AsyncMock(return_value=True)

    await service._receive_task_handler(report_error)

    assert call_count == service._MAX_CONSECUTIVE_QUICK_FAILURES
    report_error.assert_called_once()
    error_frame = report_error.call_args[0][0]
    assert isinstance(error_frame, ErrorFrame)
    assert error_frame.fatal is False


@pytest.mark.asyncio
async def test_connect_resets_quick_failure_counter(service):
    """Calling _connect() should reset the quick failure counter."""
    service._quick_failure_count = 5
    await service._connect()
    assert service._quick_failure_count == 0


@pytest.mark.asyncio
async def test_intentional_disconnect_skips_quick_failure_logic(service, report_error):
    """When _disconnecting is True, quick failure detection should not run."""
    service._disconnecting = True
    service._quick_failure_count = 0
    service._last_connect_time = time.monotonic()

    result = await service._maybe_try_reconnect("test error", report_error)

    assert result is False
    # Counter should not have been incremented
    assert service._quick_failure_count == 0
    # No error frame should have been emitted
    report_error.assert_not_called()
