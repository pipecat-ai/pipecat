#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
import warnings
from unittest.mock import AsyncMock, Mock

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InputTransportStartAudioStreamingFrame,
)
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor


class TestRTVIClientReadyVersionHandling(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.processor = RTVIProcessor()

    async def asyncTearDown(self):
        await self.processor.cleanup()

    async def _call_handle_client_ready(self, data):
        """Helper to call _handle_client_ready with a mocked _send_error_response."""
        self.processor._send_error_response = AsyncMock()
        self.processor.set_client_ready = AsyncMock()
        await self.processor._handle_client_ready("req-1", data)

    # -- Fully compatible versions (protocol major 2) -------------------------

    async def test_valid_version_2_0_0_sends_no_error(self):
        data = RTVI.ClientReadyData(
            version="2.0.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_not_called()
        self.assertEqual(self.processor._client_version, [2, 0, 0])

    async def test_valid_version_2_3_1_sends_no_error(self):
        data = RTVI.ClientReadyData(
            version="2.3.1",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_not_called()
        self.assertEqual(self.processor._client_version, [2, 3, 1])

    # -- Deprecated legacy version (1.4.x) ------------------------------------

    # TODO: enable this once RTVI 2.0.0 is supported by all our client SDKs, and we start to emit the warnings again.
    # async def test_legacy_version_1_4_0_sends_deprecation_warning(self):
    #     """1.4.x clients receive a deprecation warning but the connection is allowed."""
    #     data = RTVI.ClientReadyData(
    #         version="1.4.0",
    #         about=RTVI.AboutClientData(library="test-client"),
    #     )
    #     await self._call_handle_client_ready(data)
    #     self.processor._send_error_response.assert_called_once()
    #     warning_msg = self.processor._send_error_response.call_args[0][1]
    #     self.assertIn("deprecated", warning_msg)
    #     self.assertIn("1.4.0", warning_msg)
    #     self.assertEqual(self.processor._client_version, [1, 4, 0])

    async def test_legacy_version_sets_client_ready(self):
        """1.4.x clients still become client-ready despite the warning."""
        data = RTVI.ClientReadyData(
            version="1.4.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor.set_client_ready.assert_called_once()

    async def test_legacy_version_1_0_0_sets_client_ready(self):
        """Any 1.x client is accepted as legacy."""
        data = RTVI.ClientReadyData(
            version="1.0.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_not_called()
        self.processor.set_client_ready.assert_called_once()

    async def test_legacy_version_1_2_0_sets_client_ready(self):
        """Any 1.x client is accepted as legacy."""
        data = RTVI.ClientReadyData(
            version="1.2.0",
            about=RTVI.AboutClientData(library="test-client"),
        )
        await self._call_handle_client_ready(data)
        self.processor._send_error_response.assert_not_called()
        self.processor.set_client_ready.assert_called_once()

    # -- Incompatible versions ------------------------------------------------

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
        """Incompatible version errors should append the compatibility warning."""
        for version in ["0.9.9", "3.0.0"]:
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

    async def test_client_ready_pushes_start_audio_streaming_frame(self):
        self.processor.push_frame = AsyncMock()
        await self._call_handle_client_ready(None)
        pushed = [c.args[0] for c in self.processor.push_frame.call_args_list]
        self.assertTrue(any(isinstance(f, InputTransportStartAudioStreamingFrame) for f in pushed))


class TestRTVIFrameBasedAudio(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        await self.processor.cleanup()

    async def test_transport_param_is_deprecated(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.processor = RTVIProcessor(transport=Mock())
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    async def test_audio_buffer_pushes_input_audio_frame_downstream(self):
        self.processor = RTVIProcessor()
        self.processor.push_frame = AsyncMock()
        await self.processor._handle_audio_buffer(
            {"base64Audio": "AAAA", "sampleRate": 16000, "numChannels": 1}
        )
        pushed = [c.args[0] for c in self.processor.push_frame.call_args_list]
        self.assertEqual(len(pushed), 1)
        self.assertIsInstance(pushed[0], InputAudioRawFrame)
        self.assertEqual(pushed[0].sample_rate, 16000)


if __name__ == "__main__":
    unittest.main()
