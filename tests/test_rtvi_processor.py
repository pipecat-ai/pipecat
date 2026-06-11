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
