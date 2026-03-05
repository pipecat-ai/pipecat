#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI WebSocket LLM service initialization and basic properties."""

import unittest
from unittest.mock import patch

from pipecat.services.openai.websocket_llm import (
    InputParams,
    OpenAIWebSocketLLMService,
    OpenAIWebSocketLLMSettings,
)


class TestOpenAIWebSocketLLMServiceInit(unittest.TestCase):
    """Test OpenAIWebSocketLLMService constructor and basic configuration."""

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_default_init(self, _mock_ws):
        """Test service initializes with required model parameter."""
        service = OpenAIWebSocketLLMService(model="gpt-4o")
        self.assertEqual(service._settings.model, "gpt-4o")
        self.assertEqual(service._api_key, "")
        self.assertEqual(service._base_url, "wss://api.openai.com/v1/responses")
        self.assertFalse(service._settings.store)
        self.assertTrue(service._settings.use_previous_response_id)
        self.assertIsNone(service._previous_response_id)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_custom_api_key_and_url(self, _mock_ws):
        """Test service initializes with custom API key and base URL."""
        service = OpenAIWebSocketLLMService(
            model="gpt-4o",
            api_key="test-key-123",
            base_url="wss://custom.endpoint/v1/responses",
        )
        self.assertEqual(service._api_key, "test-key-123")
        self.assertEqual(service._base_url, "wss://custom.endpoint/v1/responses")

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_store_and_previous_response_id_settings(self, _mock_ws):
        """Test store and use_previous_response_id settings."""
        service = OpenAIWebSocketLLMService(
            model="gpt-4o",
            store=True,
            use_previous_response_id=False,
        )
        self.assertTrue(service._settings.store)
        self.assertFalse(service._settings.use_previous_response_id)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_input_params(self, _mock_ws):
        """Test input parameters are applied to settings."""
        params = InputParams(
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        service = OpenAIWebSocketLLMService(model="gpt-4o", params=params)
        self.assertEqual(service._settings.temperature, 0.7)
        self.assertEqual(service._settings.max_tokens, 1024)
        self.assertEqual(service._settings.top_p, 0.9)
        self.assertEqual(service._settings.frequency_penalty, 0.5)
        self.assertEqual(service._settings.presence_penalty, 0.3)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_default_input_params(self, _mock_ws):
        """Test default input parameters are None."""
        service = OpenAIWebSocketLLMService(model="gpt-4o")
        self.assertIsNone(service._settings.temperature)
        self.assertIsNone(service._settings.max_tokens)
        self.assertIsNone(service._settings.top_p)
        self.assertIsNone(service._settings.frequency_penalty)
        self.assertIsNone(service._settings.presence_penalty)

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_can_generate_metrics(self, _mock_ws):
        """Test that metrics generation is reported as supported."""
        service = OpenAIWebSocketLLMService(model="gpt-4o")
        self.assertTrue(service.can_generate_metrics())

    @patch("pipecat.services.openai.websocket_llm.websocket_connect")
    def test_adapter_class(self, _mock_ws):
        """Test that the correct adapter class is used."""
        from pipecat.adapters.services.open_ai_websocket_adapter import (
            OpenAIWebSocketLLMAdapter,
        )

        service = OpenAIWebSocketLLMService(model="gpt-4o")
        self.assertIsInstance(service.get_llm_adapter(), OpenAIWebSocketLLMAdapter)


class TestOpenAIWebSocketLLMSettings(unittest.TestCase):
    """Test OpenAIWebSocketLLMSettings dataclass."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = OpenAIWebSocketLLMSettings(
            model="gpt-4o",
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
        )
        self.assertFalse(settings.store)
        self.assertTrue(settings.use_previous_response_id)

    def test_custom_settings(self):
        """Test settings with custom values."""
        settings = OpenAIWebSocketLLMSettings(
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=2048,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            store=True,
            use_previous_response_id=False,
        )
        self.assertEqual(settings.model, "gpt-4o-mini")
        self.assertEqual(settings.temperature, 0.5)
        self.assertTrue(settings.store)
        self.assertFalse(settings.use_previous_response_id)


class TestInputParams(unittest.TestCase):
    """Test InputParams Pydantic model."""

    def test_default_params(self):
        """Test default InputParams values are all None."""
        params = InputParams()
        self.assertIsNone(params.temperature)
        self.assertIsNone(params.max_tokens)
        self.assertIsNone(params.top_p)
        self.assertIsNone(params.frequency_penalty)
        self.assertIsNone(params.presence_penalty)

    def test_custom_params(self):
        """Test InputParams with custom values."""
        params = InputParams(temperature=0.8, max_tokens=500)
        self.assertEqual(params.temperature, 0.8)
        self.assertEqual(params.max_tokens, 500)


if __name__ == "__main__":
    unittest.main()
