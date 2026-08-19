#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import patch

from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame
from pipecat.services.ai_service import AIService
from pipecat.services.websocket_service import WebsocketService
from pipecat.utils.errors import ErrorCategory


def websocket_rejection(status_code: int) -> InvalidStatus:
    """Build the exception `websockets` raises when a handshake is rejected."""
    return InvalidStatus(Response(status_code, "", Headers()))


class FakeWebsocket:
    """Stands in for a connected websocket that answers pings."""

    state = State.OPEN

    def __init__(self):
        self.sent: list = []

    async def ping(self):
        pass

    async def send(self, message):
        self.sent.append(message)


class FakeWebsocketService(AIService, WebsocketService):
    """Websocket service whose connection attempts succeed or fail on demand."""

    def __init__(self, connect_error: Exception | None = None, **kwargs):
        AIService.__init__(self, **kwargs)
        WebsocketService.__init__(self, **kwargs)
        self.connect_error = connect_error
        self.connect_attempts = 0
        self.reported: list[ErrorFrame] = []

    async def _connect_websocket(self):
        self.connect_attempts += 1
        if self.connect_error:
            raise self.connect_error
        self._websocket = FakeWebsocket()

    async def _disconnect_websocket(self):
        self._websocket = None

    async def _receive_messages(self):
        pass

    async def report_error(self, error: ErrorFrame, force_treat_as_permanent: bool = False):
        self.reported.append(error)
        await self.push_error_frame(error, force_treat_as_permanent=force_treat_as_permanent)


class BareWebsocketService(WebsocketService):
    """Websocket service mixed into something that isn't a frame processor."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect_attempts = 0

    async def _connect_websocket(self):
        self.connect_attempts += 1
        self._websocket = FakeWebsocket()

    async def _disconnect_websocket(self):
        self._websocket = None

    async def _receive_messages(self):
        pass


class TestReconnection(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Retry without the real backoff waits.
        backoff = patch(
            "pipecat.services.websocket_service.exponential_backoff_time", return_value=0
        )
        backoff.start()
        self.addCleanup(backoff.stop)

    async def test_an_unusable_service_does_not_reconnect(self):
        service = FakeWebsocketService()
        await service.set_usable(False)

        reconnected = await service._try_reconnect(report_error=service.report_error)

        self.assertFalse(reconnected)
        self.assertEqual(service.connect_attempts, 0)

    async def test_a_working_service_reconnects(self):
        service = FakeWebsocketService()

        reconnected = await service._try_reconnect(report_error=service.report_error)

        self.assertTrue(reconnected)
        self.assertEqual(service.connect_attempts, 1)
        self.assertTrue(service.is_usable)

    async def test_rejected_credentials_stop_further_attempts(self):
        service = FakeWebsocketService(connect_error=websocket_rejection(401))

        reconnected = await service._try_reconnect(report_error=service.report_error)

        self.assertFalse(reconnected)
        # Reporting the first rejection costs the service its usability, so the
        # remaining attempts are abandoned.
        self.assertEqual(service.connect_attempts, 1)
        self.assertFalse(service.is_usable)

    async def test_transient_failures_use_every_attempt(self):
        service = FakeWebsocketService(connect_error=websocket_rejection(503))

        reconnected = await service._try_reconnect(max_retries=2, report_error=service.report_error)

        self.assertFalse(reconnected)
        self.assertEqual(service.connect_attempts, 2)

    async def test_exhausted_attempts_leave_the_service_unusable(self):
        service = FakeWebsocketService(connect_error=websocket_rejection(503))

        await service._try_reconnect(max_retries=2, report_error=service.report_error)

        # Each attempt is reported as it fails; the last error is the one that
        # says the service has run out of ways to come back.
        self.assertFalse(service.is_usable)
        self.assertIn("failed to reconnect", service.reported[-1].error)

    async def test_a_failed_attempt_alone_leaves_the_service_usable(self):
        service = FakeWebsocketService(connect_error=websocket_rejection(503))

        await service._try_reconnect(max_retries=2, report_error=service.report_error)

        # The service was still worth retrying when the first attempt failed.
        self.assertEqual(service.reported[0].category, ErrorCategory.SERVER)

    async def test_reported_errors_carry_the_exception(self):
        error = websocket_rejection(401)
        service = FakeWebsocketService(connect_error=error)

        await service._try_reconnect(report_error=service.report_error)

        self.assertEqual(service.reported[0].exception, error)
        self.assertEqual(service.reported[0].category, ErrorCategory.AUTHENTICATION)

    async def test_only_giving_up_reports_the_service_as_spent(self):
        """Test which reported errors carry the flag that ends the service.

        A failed attempt is still worth retrying, so only the error that gives
        up says the service can no longer be given work.
        """
        service = FakeWebsocketService(connect_error=websocket_rejection(503))
        flags = []

        async def report_error(error, force_treat_as_permanent=False):
            flags.append(force_treat_as_permanent)
            await service.push_error_frame(error, force_treat_as_permanent=force_treat_as_permanent)

        await service._try_reconnect(max_retries=2, report_error=report_error)

        self.assertEqual(flags, [False, False, True])
        self.assertFalse(service.is_usable)

    async def test_a_service_that_tracks_nothing_still_reconnects(self):
        service = BareWebsocketService()

        reconnected = await service._try_reconnect()

        self.assertTrue(reconnected)
        self.assertEqual(service.connect_attempts, 1)
        self.assertTrue(service._is_service_usable)

    async def test_a_dropped_connection_leaves_the_receive_loop_alone(self):
        """Test that ending the receive loop doesn't end the service.

        Turning off reconnection here means the receive loop won't do it, not
        that the service is out of ways to come back.
        """
        service = FakeWebsocketService()
        service._reconnect_on_error = False

        await service._receive_task_handler(service.report_error)

        self.assertTrue(service.is_usable)
        self.assertEqual(service.connect_attempts, 0)

    async def test_a_dropped_connection_still_reconnects_on_demand(self):
        service = FakeWebsocketService()
        service._reconnect_on_error = False
        await service._receive_task_handler(service.report_error)

        await service.send_with_retry("hello", service.report_error)

        self.assertEqual(service.connect_attempts, 1)
        self.assertEqual(service._websocket.sent, ["hello"])
