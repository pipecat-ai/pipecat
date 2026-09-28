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
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_service import AIService
from pipecat.services.settings import ServiceSettings
from pipecat.tests.utils import run_test
from pipecat.utils.errors import (
    ErrorCategory,
    classify_http_exception,
    classify_http_status_code,
    extract_http_status_code,
)


def websocket_rejection(status_code: int) -> InvalidStatus:
    """Build the exception `websockets` raises when a handshake is rejected."""
    return InvalidStatus(Response(status_code, "", Headers()))


class ReportingService(AIService):
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
        self.assertEqual(classify_http_status_code(400), ErrorCategory.INVALID_REQUEST)
        self.assertEqual(classify_http_status_code(401), ErrorCategory.AUTHENTICATION)
        self.assertEqual(classify_http_status_code(402), ErrorCategory.QUOTA)
        self.assertEqual(classify_http_status_code(403), ErrorCategory.AUTHORIZATION)
        self.assertEqual(classify_http_status_code(404), ErrorCategory.INVALID_REQUEST)
        self.assertEqual(classify_http_status_code(422), ErrorCategory.INVALID_REQUEST)
        self.assertEqual(classify_http_status_code(429), ErrorCategory.RATE_LIMIT)

    def test_server_errors_map_to_server_category(self):
        for status_code in (500, 502, 503, 599):
            self.assertEqual(classify_http_status_code(status_code), ErrorCategory.SERVER)

    def test_unremarkable_status_codes_are_unknown(self):
        for status_code in (200, 301, 418, 600):
            self.assertEqual(classify_http_status_code(status_code), ErrorCategory.UNKNOWN)

    def test_extracts_status_code_from_websocket_rejection(self):
        self.assertEqual(extract_http_status_code(websocket_rejection(401)), 401)

    def test_extracts_status_code_from_attribute_shapes(self):
        class Nested:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        # httpx-style: exception.response.status_code
        self.assertEqual(extract_http_status_code(Nested(response=Nested(status_code=403))), 403)
        # aiohttp-style: exception.response.status
        self.assertEqual(extract_http_status_code(Nested(response=Nested(status=429))), 429)
        # SDK-style: the code directly on the exception
        self.assertEqual(extract_http_status_code(Nested(status_code=500)), 500)
        self.assertEqual(extract_http_status_code(Nested(status=404)), 404)

    def test_exceptions_without_a_status_code(self):
        self.assertIsNone(extract_http_status_code(ValueError("nope")))
        self.assertEqual(classify_http_exception(ValueError("nope")), ErrorCategory.UNKNOWN)

    def test_connectivity_exceptions(self):
        self.assertEqual(classify_http_exception(ConnectionError()), ErrorCategory.CONNECTIVITY)
        self.assertEqual(classify_http_exception(TimeoutError()), ErrorCategory.CONNECTIVITY)
        self.assertEqual(classify_http_exception(OSError()), ErrorCategory.UNKNOWN)

    def test_permanent_categories(self):
        for category in (
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.AUTHORIZATION,
            ErrorCategory.INVALID_REQUEST,
        ):
            self.assertTrue(category.is_permanent)

        for category in (
            ErrorCategory.UNKNOWN,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.QUOTA,
            ErrorCategory.CONNECTIVITY,
            ErrorCategory.SERVER,
        ):
            self.assertFalse(category.is_permanent)


class TestErrorFrame(unittest.TestCase):
    def test_category_starts_unset(self):
        # Unset invites whoever reports the error to work the cause out.
        self.assertIsNone(ErrorFrame("boom").category)

    def test_str_omits_an_unset_category(self):
        self.assertNotIn("category", str(ErrorFrame("boom")))

    def test_str_omits_an_unknown_category(self):
        self.assertNotIn("category", str(ErrorFrame("boom", category=ErrorCategory.UNKNOWN)))

    def test_str_includes_known_category(self):
        frame = ErrorFrame("boom", category=ErrorCategory.AUTHENTICATION)
        self.assertIn("category: authentication", str(frame))


class TestProcessorUsable(unittest.IsolatedAsyncioTestCase):
    async def test_processors_start_usable(self):
        self.assertTrue(ReportingService().is_usable)

    async def test_service_specific_classification_can_keep_a_service_usable(self):
        class RefreshingCredentialsService(ReportingService):
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
        self.assertTrue(service.is_usable)

    async def test_rejected_credentials_make_the_service_unusable(self):
        service = ReportingService(exception=websocket_rejection(401))

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.AUTHENTICATION)
        self.assertFalse(service.is_usable)

    async def test_server_errors_leave_the_service_usable(self):
        service = ReportingService(exception=websocket_rejection(503))

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.SERVER)
        self.assertTrue(service.is_usable)

    async def test_unclassifiable_exceptions_leave_the_service_usable(self):
        service = ReportingService(exception=ValueError("nope"))

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.UNKNOWN)
        self.assertTrue(service.is_usable)

    async def test_errors_without_an_exception_are_not_classified(self):
        service = ReportingService()

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.UNKNOWN)
        self.assertTrue(service.is_usable)

    async def test_service_specific_classification_takes_precedence(self):
        class ProviderService(ReportingService):
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
        self.assertFalse(service.is_usable)

    async def test_explicit_category_needs_no_opt_in(self):
        class ExplicitService(ReportingService):
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
        self.assertFalse(service.is_usable)

    async def test_an_error_reported_as_terminal_needs_no_category(self):
        class SpentService(ReportingService):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await AIService.process_frame(self, frame, direction)

                if isinstance(frame, TextFrame):
                    await self.push_error("gave up retrying", force_treat_as_permanent=True)
                else:
                    await self.push_frame(frame, direction)

        service = SpentService()

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        # Nothing is misconfigured; the service just ran out of attempts.
        self.assertEqual(up[0].category, ErrorCategory.UNKNOWN)
        self.assertFalse(service.is_usable)

    async def test_the_verdict_is_in_before_the_error_travels(self):
        service = ReportingService(exception=websocket_rejection(401))
        seen = []

        @service.event_handler("on_error")
        async def on_error(service, frame):
            seen.append(frame.processor.is_usable)

        await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(seen, [False])

    async def test_becoming_unusable_notifies_listeners(self):
        service = ReportingService(exception=websocket_rejection(401))
        changed = asyncio.Event()
        transitions = []

        @service.event_handler("on_usable_changed")
        async def on_usable_changed(service, is_usable):
            transitions.append(is_usable)
            changed.set()

        await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        async with asyncio.timeout(5):
            await changed.wait()

        self.assertEqual(transitions, [False])

    async def test_repeated_errors_notify_once(self):
        service = ReportingService(exception=websocket_rejection(401))
        transitions = []

        @service.event_handler("on_usable_changed")
        async def on_usable_changed(service, is_usable):
            transitions.append(is_usable)

        await run_test(
            service,
            frames_to_send=[TextFrame("one"), TextFrame("two"), TextFrame("three")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame, ErrorFrame, ErrorFrame],
        )

        self.assertEqual(len(transitions), 1)

    async def test_updated_settings_give_the_service_another_chance(self):
        service = ReportingService(exception=websocket_rejection(401))
        await service.set_usable(False)

        await service._update_settings(ServiceSettings(model="another-model"))

        self.assertTrue(service.is_usable)

    async def test_unchanged_settings_leave_the_service_alone(self):
        service = ReportingService(exception=websocket_rejection(401))
        await service.set_usable(False)

        await service._update_settings(ServiceSettings())

        self.assertFalse(service.is_usable)


class TestPlainProcessorUsable(unittest.IsolatedAsyncioTestCase):
    """Usability belongs to every processor, not just to services."""

    async def test_a_plain_processor_becomes_unusable(self):
        class FailingProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)

                if isinstance(frame, TextFrame):
                    await self.push_error("out of attempts", force_treat_as_permanent=True)
                else:
                    await self.push_frame(frame, direction)

        processor = FailingProcessor()

        await run_test(
            processor,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertFalse(processor.is_usable)

    async def test_reporting_an_ordinary_error_passes_only_the_frame(self):
        """Test that a processor overriding `push_error_frame` needs no new argument.

        The verdict is only worth passing on when there is one, so reporting an
        error that leaves the processor working stays a one-argument call.
        """
        received = []

        class NarrowProcessor(FrameProcessor):
            async def push_error_frame(self, error):
                received.append(error)

        processor = NarrowProcessor()

        await processor.push_error("something went wrong")

        self.assertEqual(len(received), 1)

    async def test_a_processor_can_be_brought_back(self):
        processor = FrameProcessor()
        await processor.set_usable(False)

        await processor.set_usable(True)

        self.assertTrue(processor.is_usable)


class ExplodingService(AIService):
    """Service whose frame processing raises before it can report anything."""

    def __init__(self, exception: Exception, **kwargs):
        super().__init__(**kwargs)
        self._exception = exception

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            raise self._exception

        await self.push_frame(frame, direction)


class TestUnattributableErrors(unittest.IsolatedAsyncioTestCase):
    """Errors caught by a broad `except` may not have come from the provider."""

    async def test_an_uncaught_exception_is_not_classified(self):
        # The exception carries a 401, but the frame-processing catch-all has
        # no idea where it came from — a downstream processor, an observer, or
        # the service itself.
        service = ExplodingService(exception=websocket_rejection(401))

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.UNKNOWN)
        self.assertTrue(service.is_usable)

    async def test_errors_always_reach_handlers_with_a_category(self):
        # Nothing reports a category here, so the push settles it.
        service = ExplodingService(exception=ValueError("boom"))

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )

        self.assertIsNotNone(up[0].category)
