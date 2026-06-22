#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.services.gnani.stt import GnaniHttpSTTService, GnaniHttpSTTSettings
from pipecat.transcriptions.language import Language


class TestGnaniHttpSTTService(unittest.IsolatedAsyncioTestCase):
    """Test cases for GnaniHttpSTTService (REST, VAD-segmented)."""

    def setUp(self):
        self.api_key = "test-api-key"
        self.mock_session = AsyncMock(spec=aiohttp.ClientSession)
        self.service = GnaniHttpSTTService(
            api_key=self.api_key,
            aiohttp_session=self.mock_session,
            settings=GnaniHttpSTTSettings(language=Language.HI_IN),
        )

    def test_initialization(self):
        self.assertEqual(self.service._api_key, self.api_key)

    def test_can_generate_metrics(self):
        self.assertTrue(self.service.can_generate_metrics())

    def test_language_conversion(self):
        languages = [
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
        for lang_enum, expected in languages:
            with self.subTest(language=lang_enum):
                self.assertEqual(self.service.language_to_service_language(lang_enum), expected)

    def test_default_language(self):
        service = GnaniHttpSTTService(
            api_key="key",
            aiohttp_session=self.mock_session,
        )
        self.assertEqual(service._settings.language, Language.EN_IN)

    async def test_run_stt_success(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"transcript": "नमस्ते"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        self.mock_session.post = MagicMock(return_value=mock_response)

        frames = []
        async for frame in self.service.run_stt(b"fake audio data"):
            if frame is not None:
                frames.append(frame)

        transcriptions = [f for f in frames if isinstance(f, TranscriptionFrame)]
        self.assertEqual(len(transcriptions), 1)
        self.assertEqual(transcriptions[0].text, "नमस्ते")

    async def test_run_stt_empty_transcript(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"transcript": ""})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        self.mock_session.post = MagicMock(return_value=mock_response)

        frames = []
        async for frame in self.service.run_stt(b"fake audio data"):
            if frame is not None:
                frames.append(frame)

        transcriptions = [f for f in frames if isinstance(f, TranscriptionFrame)]
        self.assertEqual(len(transcriptions), 0)

    async def test_run_stt_http_error(self):
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        self.mock_session.post = MagicMock(return_value=mock_response)

        frames = []
        async for frame in self.service.run_stt(b"fake audio data"):
            if frame is not None:
                frames.append(frame)

        errors = [f for f in frames if isinstance(f, ErrorFrame)]
        self.assertEqual(len(errors), 1)
        self.assertIn("Gnani STT API error", errors[0].error)

    async def test_run_stt_network_error(self):
        self.mock_session.post = MagicMock(
            side_effect=aiohttp.ClientError("Connection failed"),
        )

        frames = []
        async for frame in self.service.run_stt(b"fake audio data"):
            if frame is not None:
                frames.append(frame)

        errors = [f for f in frames if isinstance(f, ErrorFrame)]
        self.assertEqual(len(errors), 1)
        self.assertIn("Error transcribing audio", errors[0].error)


class TestGnaniHttpSTTSettings(unittest.TestCase):
    """Test GnaniHttpSTTSettings fields."""

    def test_format_field(self):
        settings = GnaniHttpSTTSettings(language=Language.EN_IN, format="verbatim")
        self.assertEqual(settings.format, "verbatim")

    def test_itn_native_numerals_field(self):
        settings = GnaniHttpSTTSettings(
            language=Language.HI_IN,
            format="transcribe",
            itn_native_numerals=True,
        )
        self.assertTrue(settings.itn_native_numerals)


if __name__ == "__main__":
    unittest.main()
