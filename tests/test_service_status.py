#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from pipecat.frames.frames import ErrorFrame, Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings
from pipecat.services.status import ServiceStatus, status_for_category
from pipecat.tests.utils import run_test
from pipecat.utils.errors import (
    ErrorCategory,
    classify_exception,
    classify_status_code,
    extract_status_code,
)


def websocket_rejection(status_code: int) -> InvalidStatus:
    """Build the exception `websockets` raises when a handshake is rejected."""
    return InvalidStatus(Response(status_code, "", Headers()))


class StatusService(AIService):
    """Service that reports a given exception whenever it sees a `TextFrame`."""

    def __init__(self, exception: Exception | None = None, **kwargs):
        super().__init__(**kwargs)
        self._exception = exception

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self.push_error("service failed", exception=self._exception)
        else:
            await self.push_frame(frame, direction)


class TestErrorClassification(unittest.TestCase):
    def test_status_codes_map_to_categories(self):
        self.assertEqual(classify_status_code(400), ErrorCategory.INVALID_REQUEST)
        self.assertEqual(classify_status_code(401), ErrorCategory.AUTHENTICATION)
        self.assertEqual(classify_status_code(402), ErrorCategory.QUOTA)
        self.assertEqual(classify_status_code(403), ErrorCategory.AUTHORIZATION)
        self.assertEqual(classify_status_code(404), ErrorCategory.INVALID_REQUEST)
        self.assertEqual(classify_status_code(422), ErrorCategory.INVALID_REQUEST)
        self.assertEqual(classify_status_code(429), ErrorCategory.RATE_LIMIT)

    def test_server_errors_map_to_server_category(self):
        for status_code in (500, 502, 503, 599):
            self.assertEqual(classify_status_code(status_code), ErrorCategory.SERVER)

    def test_unremarkable_status_codes_are_unknown(self):
        for status_code in (200, 301, 418, 600):
            self.assertEqual(classify_status_code(status_code), ErrorCategory.UNKNOWN)

    def test_extracts_status_code_from_websocket_rejection(self):
        self.assertEqual(extract_status_code(websocket_rejection(401)), 401)

    def test_extracts_status_code_from_attribute_shapes(self):
        class Nested:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        # httpx-style: exception.response.status_code
        self.assertEqual(extract_status_code(Nested(response=Nested(status_code=403))), 403)
        # aiohttp-style: exception.response.status
        self.assertEqual(extract_status_code(Nested(response=Nested(status=429))), 429)
        # SDK-style: the code directly on the exception
        self.assertEqual(extract_status_code(Nested(status_code=500)), 500)
        self.assertEqual(extract_status_code(Nested(status=404)), 404)

    def test_exceptions_without_a_status_code(self):
        self.assertIsNone(extract_status_code(ValueError("nope")))
        self.assertEqual(classify_exception(ValueError("nope")), ErrorCategory.UNKNOWN)

    def test_connectivity_exceptions(self):
        self.assertEqual(classify_exception(ConnectionError()), ErrorCategory.CONNECTIVITY)
        self.assertEqual(classify_exception(TimeoutError()), ErrorCategory.CONNECTIVITY)
        self.assertEqual(classify_exception(OSError()), ErrorCategory.UNKNOWN)

    def test_configuration_errors(self):
        for category in (
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.AUTHORIZATION,
            ErrorCategory.INVALID_REQUEST,
        ):
            self.assertTrue(category.is_configuration_error)

        for category in (
            ErrorCategory.UNKNOWN,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.QUOTA,
            ErrorCategory.CONNECTIVITY,
            ErrorCategory.SERVER,
        ):
            self.assertFalse(category.is_configuration_error)

    def test_only_configuration_errors_imply_a_status(self):
        self.assertEqual(
            status_for_category(ErrorCategory.AUTHENTICATION), ServiceStatus.MISCONFIGURED
        )
        self.assertIsNone(status_for_category(ErrorCategory.SERVER))
        self.assertIsNone(status_for_category(ErrorCategory.UNKNOWN))

    def test_is_misconfigured_flags_only_that_status(self):
        self.assertTrue(ServiceStatus.MISCONFIGURED.is_misconfigured)
        for status in (
            ServiceStatus.UNKNOWN,
            ServiceStatus.READY,
            ServiceStatus.DEGRADED,
            ServiceStatus.UNAVAILABLE,
        ):
            self.assertFalse(status.is_misconfigured)


class TestErrorFrame(unittest.TestCase):
    def test_defaults_to_unknown_category(self):
        self.assertEqual(ErrorFrame("boom").category, ErrorCategory.UNKNOWN)

    def test_str_omits_unknown_category(self):
        self.assertNotIn("category", str(ErrorFrame("boom")))

    def test_str_includes_known_category(self):
        frame = ErrorFrame("boom", category=ErrorCategory.AUTHENTICATION)
        self.assertIn("category: authentication", str(frame))


class TestServiceStatus(unittest.IsolatedAsyncioTestCase):
    async def test_services_start_unknown(self):
        self.assertEqual(StatusService().status, ServiceStatus.UNKNOWN)

    async def test_service_specific_classification_can_keep_a_service_usable(self):
        class RefreshingCredentialsService(StatusService):
            def _classify_error(self, exception: Exception) -> ErrorCategory | None:
                return ErrorCategory.CONNECTIVITY

        service = RefreshingCredentialsService(exception=websocket_rejection(401))

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.CONNECTIVITY)
        self.assertEqual(service.status, ServiceStatus.UNKNOWN)

    async def test_rejected_credentials_misconfigure_the_service(self):
        service = StatusService(exception=websocket_rejection(401))

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.AUTHENTICATION)
        self.assertEqual(service.status, ServiceStatus.MISCONFIGURED)
        self.assertTrue(service.status.is_misconfigured)

    async def test_server_errors_leave_the_status_alone(self):
        service = StatusService(exception=websocket_rejection(503))

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.SERVER)
        self.assertEqual(service.status, ServiceStatus.UNKNOWN)

    async def test_unclassifiable_exceptions_leave_the_status_alone(self):
        service = StatusService(exception=ValueError("nope"))

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.UNKNOWN)
        self.assertEqual(service.status, ServiceStatus.UNKNOWN)

    async def test_errors_without_an_exception_are_not_classified(self):
        service = StatusService()

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.UNKNOWN)
        self.assertEqual(service.status, ServiceStatus.UNKNOWN)

    async def test_service_specific_classification_takes_precedence(self):
        class ProviderService(StatusService):
            def _classify_error(self, exception: Exception) -> ErrorCategory | None:
                return ErrorCategory.AUTHORIZATION

        service = ProviderService(exception=websocket_rejection(503))

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.AUTHORIZATION)
        self.assertEqual(service.status, ServiceStatus.MISCONFIGURED)

    async def test_explicit_category_needs_no_opt_in(self):
        class ExplicitService(StatusService):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await AIService.process_frame(self, frame, direction)

                if isinstance(frame, TextFrame):
                    await self.push_error("bad key", category=ErrorCategory.AUTHENTICATION)
                else:
                    await self.push_frame(frame, direction)

        service = ExplicitService()

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.AUTHENTICATION)
        self.assertEqual(service.status, ServiceStatus.MISCONFIGURED)

    async def test_status_change_notifies_listeners(self):
        service = StatusService(exception=websocket_rejection(401))
        changed = asyncio.Event()
        transitions = []

        @service.event_handler("on_status_changed")
        async def on_status_changed(service, previous, current):
            transitions.append((previous, current))
            changed.set()

        await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        async with asyncio.timeout(5):
            await changed.wait()

        self.assertEqual(transitions, [(ServiceStatus.UNKNOWN, ServiceStatus.MISCONFIGURED)])

    async def test_repeated_errors_notify_once(self):
        service = StatusService(exception=websocket_rejection(401))
        transitions = []

        @service.event_handler("on_status_changed")
        async def on_status_changed(service, previous, current):
            transitions.append((previous, current))

        await run_test(
            service,
            frames_to_send=[TextFrame("one"), TextFrame("two"), TextFrame("three")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame, ErrorFrame, ErrorFrame],
        )

        self.assertEqual(len(transitions), 1)

    async def test_updated_settings_give_the_service_another_chance(self):
        service = StatusService(exception=websocket_rejection(401))
        await service._set_status(ServiceStatus.MISCONFIGURED)

        await service._update_settings(ServiceSettings(model="another-model"))

        self.assertEqual(service.status, ServiceStatus.UNKNOWN)

    async def test_unchanged_settings_leave_the_status_alone(self):
        service = StatusService(exception=websocket_rejection(401))
        await service._set_status(ServiceStatus.MISCONFIGURED)

        await service._update_settings(ServiceSettings())

        self.assertEqual(service.status, ServiceStatus.MISCONFIGURED)
