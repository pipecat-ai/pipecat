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
from pipecat.services.status import ServiceStatus
from pipecat.services.websocket_service import WebsocketService
from pipecat.utils.errors import ErrorCategory


def websocket_rejection(status_code: int) -> InvalidStatus:
    """Build the exception `websockets` raises when a handshake is rejected."""
    return InvalidStatus(Response(status_code, "", Headers()))


class FakeWebsocket:
    """Stands in for a connected websocket that answers pings."""

    state = State.OPEN

    async def ping(self):
        pass


class FakeWebsocketService(AIService, WebsocketService):
    """Websocket service whose connection attempts succeed or fail on demand."""

    def __init__(self, connect_error: Exception | None = None, **kwargs):
        AIService.__init__(self, **kwargs)
        WebsocketService.__init__(self, **kwargs)
        self.connect_error = connect_error
        self.connect_attempts = 0
        self.reported: list[ErrorFrame] = []
        self.observed_statuses: list[ServiceStatus] = []

    async def _set_status(self, status: ServiceStatus):
        await super()._set_status(status)
        self.observed_statuses.append(status)

    async def _connect_websocket(self):
        self.connect_attempts += 1
        if self.connect_error:
            raise self.connect_error
        self._websocket = FakeWebsocket()

    async def _disconnect_websocket(self):
        self._websocket = None

    async def _receive_messages(self):
        pass

    async def report_error(self, error: ErrorFrame):
        self.reported.append(error)
        await self.push_error_frame(error)


class BareWebsocketService(WebsocketService):
    """Websocket service mixed into something that doesn't track status."""

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


class TestReconnectionGate(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Retry without the real backoff waits.
        backoff = patch(
            "pipecat.services.websocket_service.exponential_backoff_time", return_value=0
        )
        backoff.start()
        self.addCleanup(backoff.stop)

    async def test_misconfigured_service_does_not_reconnect(self):
        service = FakeWebsocketService()
        await service._set_status(ServiceStatus.MISCONFIGURED)

        reconnected = await service._try_reconnect(report_error=service.report_error)

        self.assertFalse(reconnected)
        self.assertEqual(service.connect_attempts, 0)

    async def test_healthy_service_reconnects(self):
        service = FakeWebsocketService()

        reconnected = await service._try_reconnect(report_error=service.report_error)

        self.assertTrue(reconnected)
        self.assertEqual(service.connect_attempts, 1)
        self.assertEqual(service.status, ServiceStatus.READY)

    async def test_rejected_credentials_stop_further_attempts(self):
        service = FakeWebsocketService(connect_error=websocket_rejection(401))

        reconnected = await service._try_reconnect(report_error=service.report_error)

        self.assertFalse(reconnected)
        # The first rejection identifies the service as misconfigured, so the
        # remaining attempts are abandoned.
        self.assertEqual(service.connect_attempts, 1)
        self.assertEqual(service.status, ServiceStatus.MISCONFIGURED)

    async def test_transient_failures_use_every_attempt(self):
        service = FakeWebsocketService(connect_error=websocket_rejection(503))

        reconnected = await service._try_reconnect(max_retries=2, report_error=service.report_error)

        self.assertFalse(reconnected)
        self.assertEqual(service.connect_attempts, 2)
        self.assertEqual(service.status, ServiceStatus.UNAVAILABLE)

    async def test_reported_errors_carry_the_exception(self):
        error = websocket_rejection(401)
        service = FakeWebsocketService(connect_error=error)

        await service._try_reconnect(report_error=service.report_error)

        self.assertEqual(service.reported[0].exception, error)
        self.assertEqual(service.reported[0].category, ErrorCategory.AUTHENTICATION)

    async def test_service_without_status_tracking_reconnects(self):
        service = BareWebsocketService()

        reconnected = await service._try_reconnect()

        self.assertTrue(reconnected)
        self.assertEqual(service.connect_attempts, 1)
        self.assertEqual(service._connection_status, ServiceStatus.UNKNOWN)


class TestConnectionStatusTransitions(unittest.IsolatedAsyncioTestCase):
    async def test_misconfigured_outranks_usable_statuses(self):
        service = FakeWebsocketService()
        await service._set_status(ServiceStatus.MISCONFIGURED)

        for status in (ServiceStatus.READY, ServiceStatus.DEGRADED, ServiceStatus.UNAVAILABLE):
            await service._set_connection_status(status)
            self.assertEqual(service.status, ServiceStatus.MISCONFIGURED)

    async def test_usable_statuses_replace_each_other(self):
        service = FakeWebsocketService()

        await service._set_connection_status(ServiceStatus.DEGRADED)
        self.assertEqual(service.status, ServiceStatus.DEGRADED)

        await service._set_connection_status(ServiceStatus.READY)
        self.assertEqual(service.status, ServiceStatus.READY)

    async def test_receive_loop_reports_a_working_connection(self):
        service = FakeWebsocketService()
        # The receive loop ends once the connection closes and reconnection is
        # disabled, after marking the established connection as ready.
        service._reconnect_on_error = False

        await service._receive_task_handler(service.report_error)

        self.assertEqual(service.status, ServiceStatus.UNAVAILABLE)
        self.assertIn(ServiceStatus.READY, service.observed_statuses)
