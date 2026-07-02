#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for custom http_client support in OpenAI TTS and Whisper-based STT services."""

import unittest
from unittest.mock import patch

import httpx

from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService


class TestOpenAIHttpClient(unittest.IsolatedAsyncioTestCase):
    async def test_openai_tts_uses_custom_http_client(self):
        async with httpx.AsyncClient() as http_client:
            with patch("pipecat.services.openai.tts.AsyncOpenAI") as mock_openai:
                OpenAITTSService(api_key="test-key", http_client=http_client)
            self.assertIs(mock_openai.call_args.kwargs["http_client"], http_client)

    async def test_openai_stt_uses_custom_http_client(self):
        async with httpx.AsyncClient() as http_client:
            with patch("pipecat.services.whisper.base_stt.AsyncOpenAI") as mock_openai:
                OpenAISTTService(api_key="test-key", http_client=http_client)
            self.assertIs(mock_openai.call_args.kwargs["http_client"], http_client)

    async def test_openai_tts_http_client_defaults_to_none(self):
        with patch("pipecat.services.openai.tts.AsyncOpenAI") as mock_openai:
            OpenAITTSService(api_key="test-key")
        self.assertIsNone(mock_openai.call_args.kwargs["http_client"])
