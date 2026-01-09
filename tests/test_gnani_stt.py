#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.services.gnani.stt import GnaniSTTService
from pipecat.transcriptions.language import Language


class TestGnaniSTTService(unittest.IsolatedAsyncioTestCase):
    """Test cases for GnaniSTTService."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key"
        self.organization_id = "test-org-id"
        self.service = GnaniSTTService(
            api_key=self.api_key,
            organization_id=self.organization_id,
            params=GnaniSTTService.InputParams(
                language=Language.HI_IN,
                api_user_id="test-user",
            ),
        )

    def test_initialization(self):
        """Test that the service initializes correctly."""
        self.assertEqual(self.service._api_key, self.api_key)
        self.assertEqual(self.service._organization_id, self.organization_id)
        self.assertEqual(self.service._settings["language"], "hi-IN")
        self.assertEqual(self.service._settings["api_user_id"], "test-user")

    def test_can_generate_metrics(self):
        """Test that metrics generation is supported."""
        self.assertTrue(self.service.can_generate_metrics())

    def test_language_conversion(self):
        """Test language enum to service language code conversion."""
        # Test Hindi
        self.assertEqual(
            self.service.language_to_service_language(Language.HI_IN),
            "hi-IN"
        )
        # Test Tamil
        self.assertEqual(
            self.service.language_to_service_language(Language.TA_IN),
            "ta-IN"
        )
        # Test English (India)
        self.assertEqual(
            self.service.language_to_service_language(Language.EN_IN),
            "en-IN"
        )

    async def test_set_language(self):
        """Test setting language dynamically."""
        await self.service.set_language(Language.TA_IN)
        self.assertEqual(self.service._settings["language"], "ta-IN")

        await self.service.set_language(Language.EN_IN)
        self.assertEqual(self.service._settings["language"], "en-IN")

    @patch("aiohttp.ClientSession")
    async def test_run_stt_success(self, mock_session_class):
        """Test successful transcription."""
        # Mock the API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "success": True,
            "transcript": "नमस्ते",  # Hindi for "Hello"
            "request_id": "test-123",
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_post = MagicMock(return_value=mock_response)
        
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        # Mock audio data
        audio_data = b"fake audio data"

        # Run the STT
        frames = []
        async for frame in self.service.run_stt(audio_data):
            frames.append(frame)

        # Verify we got a TranscriptionFrame
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], TranscriptionFrame)
        self.assertEqual(frames[0].text, "नमस्ते")

    @patch("aiohttp.ClientSession")
    async def test_run_stt_empty_transcript(self, mock_session_class):
        """Test handling of empty transcript response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "success": True,
            "transcript": "",  # Empty transcript
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_post = MagicMock(return_value=mock_response)
        
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        audio_data = b"fake audio data"

        frames = []
        async for frame in self.service.run_stt(audio_data):
            frames.append(frame)

        # Should not yield any frames for empty transcript
        self.assertEqual(len(frames), 0)

    @patch("aiohttp.ClientSession")
    async def test_run_stt_api_error(self, mock_session_class):
        """Test handling of API error response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "success": False,
            "error": "Invalid audio format",
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_post = MagicMock(return_value=mock_response)
        
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        audio_data = b"fake audio data"

        frames = []
        async for frame in self.service.run_stt(audio_data):
            frames.append(frame)

        # Should yield an ErrorFrame
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], ErrorFrame)

    @patch("aiohttp.ClientSession")
    async def test_run_stt_http_error(self, mock_session_class):
        """Test handling of HTTP error status codes."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_post = MagicMock(return_value=mock_response)
        
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        audio_data = b"fake audio data"

        frames = []
        async for frame in self.service.run_stt(audio_data):
            frames.append(frame)

        # Should yield an ErrorFrame
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], ErrorFrame)
        self.assertIn("status 500", frames[0].error)

    @patch("aiohttp.ClientSession")
    async def test_run_stt_network_error(self, mock_session_class):
        """Test handling of network errors."""
        mock_post = MagicMock(side_effect=aiohttp.ClientError("Connection failed"))
        
        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_session_class.return_value = mock_session

        audio_data = b"fake audio data"

        frames = []
        async for frame in self.service.run_stt(audio_data):
            frames.append(frame)

        # Should yield an ErrorFrame
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], ErrorFrame)
        self.assertIn("Network error", frames[0].error)

    def test_multiple_languages(self):
        """Test multiple language support."""
        languages_to_test = [
            (Language.EN_IN, "en-IN"),
            (Language.HI_IN, "hi-IN"),
            (Language.TA_IN, "ta-IN"),
            (Language.TE_IN, "te-IN"),
            (Language.KN_IN, "kn-IN"),
            (Language.GU_IN, "gu-IN"),
            (Language.MR_IN, "mr-IN"),
            (Language.BN_IN, "bn-IN"),
            (Language.ML_IN, "ml-IN"),
            (Language.PA_IN, "pa-IN"),
        ]

        for lang_enum, expected_code in languages_to_test:
            with self.subTest(language=lang_enum):
                result = self.service.language_to_service_language(lang_enum)
                self.assertEqual(result, expected_code)


if __name__ == "__main__":
    unittest.main()

