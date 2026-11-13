#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for custom http_client support in OpenAI TTS and Whisper-based STT services."""

import unittest

from openai import DefaultAsyncHttpxClient

from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from tests.openai_http_helpers import ASYNC_CLIENT, http

# The SDK adopts a caller-supplied client as-is and derives its request timeout from it.
CUSTOM_TIMEOUT = http.Timeout(60.0, connect=10.0)


def make_http_client() -> DefaultAsyncHttpxClient:
    return DefaultAsyncHttpxClient(timeout=CUSTOM_TIMEOUT)


class TestOpenAIHttpClient(unittest.IsolatedAsyncioTestCase):
    async def test_openai_tts_uses_custom_http_client(self):
        async with make_http_client() as http_client:
            service = OpenAITTSService(api_key="test-key", http_client=http_client)
            self.assertIs(service._client._client, http_client)
            self.assertEqual(service._client.timeout, CUSTOM_TIMEOUT)

    async def test_openai_stt_uses_custom_http_client(self):
        async with make_http_client() as http_client:
            service = OpenAISTTService(api_key="test-key", http_client=http_client)
            self.assertIs(service._client._client, http_client)
            self.assertEqual(service._client.timeout, CUSTOM_TIMEOUT)

    async def test_groq_stt_uses_custom_http_client(self):
        async with make_http_client() as http_client:
            service = GroqSTTService(api_key="test-key", http_client=http_client)
            self.assertIs(service._client._client, http_client)

    async def test_openai_tts_defaults_to_sdk_client(self):
        service = OpenAITTSService(api_key="test-key")
        self.assertIsInstance(service._client._client, ASYNC_CLIENT)
        self.assertNotEqual(service._client.timeout, CUSTOM_TIMEOUT)

    async def test_openai_stt_defaults_to_sdk_client(self):
        service = OpenAISTTService(api_key="test-key")
        self.assertIsInstance(service._client._client, ASYNC_CLIENT)
        self.assertNotEqual(service._client.timeout, CUSTOM_TIMEOUT)
