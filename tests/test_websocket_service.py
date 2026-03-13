#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from dataclasses import dataclass
from unittest.mock import AsyncMock

from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from pipecat.frames.frames import ErrorFrame
from pipecat.services.websocket_service import WebsocketService


class MockWebsocketService(WebsocketService):
    """Concrete subclass for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect_count = 0
        self.disconnect_count = 0
        self.receive_side_effect = None

    async def _connect_websocket(self):
        self.connect_count += 1

    async def _disconnect_websocket(self):
        self.disconnect_count += 1

    async def _receive_messages(self):
        if self.receive_side_effect:
            effect = self.receive_side_effect
            raise effect


def make_close_error(code: int, reason: str = "") -> ConnectionClosedError:
    """Create a ConnectionClosedError with a specific close code."""
    rcvd = Close(code, reason)
    sent = Close(1000, "")
    return ConnectionClosedError(rcvd, sent, rcvd_then_sent=True)


class TestWebsocketServiceNonRecoverableCodes(unittest.IsolatedAsyncioTestCase):
    async def test_policy_violation_stops_reconnection(self):
        """Close code 1008 (policy violation) should emit fatal error, not reconnect."""
        service = MockWebsocketService()
        errors = []

        async def report_error(frame: ErrorFrame):
            errors.append(frame)

        error = make_close_error(1008, "Invalid API key")
        service.receive_side_effect = error

        await service._receive_task_handler(report_error)

        # Should have emitted exactly one fatal error
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].fatal)
        self.assertIn("non-recoverable", errors[0].error)
        # Should NOT have attempted reconnection (connect_count stays at 0)
        self.assertEqual(service.connect_count, 0)

    async def test_private_close_code_stops_reconnection(self):
        """Close codes 4000-4999 (private/application) should not reconnect."""
        service = MockWebsocketService()
        errors = []

        async def report_error(frame: ErrorFrame):
            errors.append(frame)

        error = make_close_error(4001, "Custom application error")
        service.receive_side_effect = error

        await service._receive_task_handler(report_error)

        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].fatal)
        self.assertEqual(service.connect_count, 0)

    async def test_protocol_error_stops_reconnection(self):
        """Close code 1002 (protocol error) should not reconnect."""
        service = MockWebsocketService()
        errors = []

        async def report_error(frame: ErrorFrame):
            errors.append(frame)

        error = make_close_error(1002, "Protocol error")
        service.receive_side_effect = error

        await service._receive_task_handler(report_error)

        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].fatal)


class TestWebsocketServiceRapidFailureDetection(unittest.IsolatedAsyncioTestCase):
    async def test_rapid_failures_stop_reconnection(self):
        """Connections that die immediately after reconnecting should trigger fatal error."""
        service = MockWebsocketService()
        errors = []
        call_count = 0

        async def report_error(frame: ErrorFrame):
            errors.append(frame)

        # Simulate a recoverable error (code 1006 = abnormal closure)
        error = make_close_error(1006, "Abnormal closure")

        original_reconnect = service._reconnect_websocket

        async def mock_reconnect(attempt):
            nonlocal call_count
            call_count += 1
            # Always "succeed" to simulate the handshake-succeeds-but-immediately-drops pattern
            return True

        service._reconnect_websocket = mock_reconnect
        # Pre-set the last connect time to "now" so elapsed time is ~0
        service._last_connect_time = __import__("time").monotonic()

        # First rapid failure
        service.receive_side_effect = error
        result = await service._maybe_try_reconnect("error 1", report_error, error)
        self.assertTrue(result)  # First rapid failure, still retries

        service._last_connect_time = __import__("time").monotonic()
        result = await service._maybe_try_reconnect("error 2", report_error, error)
        self.assertTrue(result)  # Second rapid failure, still retries

        service._last_connect_time = __import__("time").monotonic()
        result = await service._maybe_try_reconnect("error 3", report_error, error)
        self.assertFalse(result)  # Third rapid failure, gives up

        # Should have emitted a fatal error
        fatal_errors = [e for e in errors if e.fatal]
        self.assertEqual(len(fatal_errors), 1)
        self.assertIn("rapid failures", fatal_errors[0].error)

    async def test_stable_connection_resets_rapid_failure_count(self):
        """A connection that lasts long enough should reset the failure counter."""
        service = MockWebsocketService()
        errors = []

        async def report_error(frame: ErrorFrame):
            errors.append(frame)

        async def mock_reconnect(attempt):
            return True

        service._reconnect_websocket = mock_reconnect
        error = make_close_error(1006, "Abnormal closure")

        # Simulate a connection that was established long enough ago (stable)
        service._last_connect_time = __import__("time").monotonic() - 10.0

        result = await service._maybe_try_reconnect("error", report_error, error)
        self.assertTrue(result)
        # Rapid failure count should be 0 (reset by stable connection)
        self.assertEqual(service._rapid_failure_count, 0)


class TestWebsocketServiceDisconnectFlag(unittest.IsolatedAsyncioTestCase):
    async def test_intentional_disconnect_does_not_reconnect(self):
        """Setting _disconnecting should prevent reconnection attempts."""
        service = MockWebsocketService()
        service._disconnecting = True
        errors = []

        async def report_error(frame: ErrorFrame):
            errors.append(frame)

        result = await service._maybe_try_reconnect("error", report_error)
        self.assertFalse(result)
        self.assertEqual(len(errors), 0)

    async def test_connect_resets_rapid_failure_count(self):
        """Calling _connect should reset the rapid failure counter."""
        service = MockWebsocketService()
        service._rapid_failure_count = 5
        await service._connect()
        self.assertEqual(service._rapid_failure_count, 0)
        self.assertFalse(service._disconnecting)


if __name__ == "__main__":
    unittest.main()
