#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for API key validation."""

import pytest

from pipecat.utils.api_key_validator import APIKeyError, validate_api_key


class TestAPIKeyValidation:
    """Test cases for API key validation functionality."""

    def test_valid_api_key(self):
        """Test that a valid API key passes validation."""
        # Should not raise an exception
        validate_api_key("sk-test-key-123", "TestService")

    def test_none_api_key_not_allowed(self):
        """Test that None API key raises error when not allowed."""
        with pytest.raises(APIKeyError) as exc_info:
            validate_api_key(None, "TestService", allow_none=False)

        assert "API key for TestService is missing or empty" in str(exc_info.value)

    def test_none_api_key_allowed(self):
        """Test that None API key is allowed when explicitly permitted."""
        # Should not raise an exception
        validate_api_key(None, "TestService", allow_none=True)

    def test_empty_string_api_key(self):
        """Test that empty string API key raises error."""
        with pytest.raises(APIKeyError) as exc_info:
            validate_api_key("", "TestService")

        assert "API key for TestService is missing or empty" in str(exc_info.value)

    def test_whitespace_only_api_key(self):
        """Test that whitespace-only API key raises error."""
        with pytest.raises(APIKeyError) as exc_info:
            validate_api_key("   ", "TestService")

        assert "API key for TestService is missing or empty" in str(exc_info.value)

    def test_error_message_with_env_var(self):
        """Test that error message includes environment variable hint."""
        with pytest.raises(APIKeyError) as exc_info:
            validate_api_key(None, "TestService", env_var_name="TEST_API_KEY")

        error_msg = str(exc_info.value)
        assert "TEST_API_KEY" in error_msg
        assert "environment variable" in error_msg

    def test_error_message_without_env_var(self):
        """Test that error message works without environment variable hint."""
        with pytest.raises(APIKeyError) as exc_info:
            validate_api_key("", "TestService")

        error_msg = str(exc_info.value)
        assert "TestService" in error_msg
        assert "Please provide a valid API key" in error_msg


class TestServiceIntegration:
    """Test integration with actual services."""

    def test_openai_with_blank_key(self):
        """Test that OpenAI service rejects blank API key."""
        from pipecat.services.openai import OpenAILLMService

        # Blank string should raise error even with allow_none=True
        with pytest.raises(APIKeyError) as exc_info:
            OpenAILLMService(api_key="", model="gpt-4.1")

        assert "OpenAI" in str(exc_info.value)

    def test_anthropic_with_none_key(self):
        """Test that Anthropic service rejects None API key."""
        pytest.importorskip("anthropic")
        from pipecat.services.anthropic import AnthropicLLMService

        with pytest.raises(APIKeyError) as exc_info:
            AnthropicLLMService(api_key=None, model="claude-sonnet-4-5-20250929")

        assert "Anthropic" in str(exc_info.value)

    def test_anthropic_with_blank_key(self):
        """Test that Anthropic service rejects blank API key."""
        pytest.importorskip("anthropic")
        from pipecat.services.anthropic import AnthropicLLMService

        with pytest.raises(APIKeyError) as exc_info:
            AnthropicLLMService(api_key="   ", model="claude-sonnet-4-5-20250929")

        assert "Anthropic" in str(exc_info.value)

    def test_deepgram_with_none_key(self):
        """Test that Deepgram service rejects None API key."""
        pytest.importorskip("deepgram")
        from pipecat.services.deepgram import DeepgramSTTService

        with pytest.raises(APIKeyError) as exc_info:
            DeepgramSTTService(api_key=None)

        assert "Deepgram" in str(exc_info.value)
