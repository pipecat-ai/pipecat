#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock, patch

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor


class TestRTVIClientReadyVersionHandling(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.processor = RTVIProcessor()

    async def asyncTearDown(self):
        await self.processor.cleanup()

    async def _call_handle_client_ready(self, data):
        """Helper to call _handle_client_ready with a mocked _send_error_response."""
        self.processor._send_error_response = AsyncMock()
        self.processor._input_transport = None
        self.processor.set_client_ready = AsyncMock()
        await self.processor._handle_client_ready("req-1", data)

    async def test_valid_version_1_0_0_sends_no_error(self):
        data = RTVI.ClientReadyData(
            version="1.0.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_not_called()
        self.assertEqual(self.processor._client_version, [1, 0, 0])

    async def test_valid_version_1_2_0_sends_no_error(self):
        data = RTVI.ClientReadyData(
            version="1.2.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_not_called()
        self.assertEqual(self.processor._client_version, [1, 2, 0])

    async def test_version_below_1_0_0_sends_error(self):
        data = RTVI.ClientReadyData(
            version="0.3.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_called_once()
        error_msg = self.processor._send_error_response.call_args[0][1]
        self.assertIn("0.3.0", error_msg)
        self.assertIn("not compatible", error_msg)

    async def test_version_above_protocol_major_sends_error(self):
        data = RTVI.ClientReadyData(
            version="2.3.1",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_called_once()
        error_msg = self.processor._send_error_response.call_args[0][1]
        self.assertIn("2.3.1", error_msg)
        self.assertIn("not compatible", error_msg)

    async def test_no_version_sends_error(self):
        """Client sends no data (data=None)."""
        await self._call_handle_client_ready(None)
        self.processor._send_error_response.assert_called_once()
        error_msg = self.processor._send_error_response.call_args[0][1]
        self.assertIn("unknown", error_msg)

    async def test_invalid_version_format_sends_error(self):
        bad_versions = ["not-a-version", "123", "1.2.3.0", "junk", "1.2"]
        for version in bad_versions:
            with self.subTest(version=version):
                data = RTVI.ClientReadyData(
                    version=version,
                    about=RTVI.AboutClientData(library="test-client"),
                )
                await self._call_handle_client_ready(data)
                self.processor._send_error_response.assert_called_once()
                error_msg = self.processor._send_error_response.call_args[0][1]
                self.assertIn("Invalid client version format", error_msg)
                self.assertIn(version, error_msg)

    async def test_error_message_includes_compatibility_warning(self):
        """All version errors should append the compatibility warning."""
        for version in ["0.9.9", "2.0.0"]:
            with self.subTest(version=version):
                data = RTVI.ClientReadyData(
                    version=version,
                    about=RTVI.AboutClientData(library="test-client"),
                )
                await self._call_handle_client_ready(data)
                error_msg = self.processor._send_error_response.call_args[0][1]
                self.assertIn("Compatibility issues may occur", error_msg)

    async def test_client_ready_is_set_even_on_version_error(self):
        """Client-ready state should be set regardless of version errors."""
        data = RTVI.ClientReadyData(
            version="0.3.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor.set_client_ready.assert_called_once()

    async def test_client_ready_is_set_when_no_data(self):
        await self._call_handle_client_ready(None)
        self.processor.set_client_ready.assert_called_once()


if __name__ == "__main__":
    unittest.main()
