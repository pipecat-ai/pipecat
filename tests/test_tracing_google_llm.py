#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Unit tests for Langfuse tracing with GoogleLLMService.

These tests verify that the tracing decorator correctly captures messages
for Google LLM services when using different context types.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService


@pytest.mark.asyncio
async def test_google_tracing_with_universal_llm_context():
    """Test that tracing correctly captures messages for GoogleLLMService with universal LLMContext."""
    # Create service with mocked client
    service = GoogleLLMService(api_key="test-key", model="gemini-2.0-flash")
    service._client = AsyncMock()

    # Enable tracing for this service instance
    service._tracing_enabled = True

    # Create universal LLMContext with system message
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, world!"},
    ]
    context = LLMContext(messages=test_messages)

    # Mock the adapter to return properly formatted messages
    mock_adapter = MagicMock()
    mock_adapter.get_messages_for_logging.return_value = [
        {"role": "user", "parts": [{"text": "You are a helpful assistant"}]},
        {"role": "user", "parts": [{"text": "Hello, world!"}]},
    ]
    mock_adapter.get_llm_invocation_params.return_value = {
        "messages": [],
        "system_instruction": "You are a helpful assistant",
        "tools": [],
    }
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Mock the generate_content_stream to return empty response
    mock_response = AsyncMock()
    mock_response.__aiter__.return_value = []
    service._client.aio.models.generate_content_stream.return_value = mock_response

    # Mock tracing components
    with patch("opentelemetry.trace.get_tracer") as mock_get_tracer:
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        # Execute _process_context
        await service._process_context(context)

        # Verify that adapter's get_messages_for_logging was called (at least once by tracing)
        assert mock_adapter.get_messages_for_logging.call_count >= 1, (
            "Adapter's get_messages_for_logging should be called"
        )
        # Verify it was called with the context
        mock_adapter.get_messages_for_logging.assert_any_call(context)

        # Verify span.set_attribute was called with serialized messages
        # Find the call that sets the "input" attribute
        input_calls = [
            call for call in mock_span.set_attribute.call_args_list if call[0][0] == "input"
        ]

        # Should have exactly one call setting "input"
        assert len(input_calls) == 1, "Expected exactly one call to set_attribute with 'input'"

        # Verify the input is a non-empty JSON string
        input_value = input_calls[0][0][1]
        assert isinstance(input_value, str), "Input should be a JSON string"
        assert input_value != "", "Input should not be empty"

        # Verify it's valid JSON
        parsed_messages = json.loads(input_value)
        assert isinstance(parsed_messages, list), "Parsed input should be a list"
        assert len(parsed_messages) > 0, "Should have at least one message"


@pytest.mark.asyncio
async def test_google_tracing_with_empty_context():
    """Test that tracing correctly handles empty message lists."""
    # Create service with mocked client
    service = GoogleLLMService(api_key="test-key", model="gemini-2.0-flash")
    service._client = AsyncMock()

    # Enable tracing for this service instance
    service._tracing_enabled = True

    # Create universal LLMContext with no messages
    context = LLMContext(messages=[])

    # Mock the adapter to return empty messages list
    mock_adapter = MagicMock()
    mock_adapter.get_messages_for_logging.return_value = []
    mock_adapter.get_llm_invocation_params.return_value = {
        "messages": [],
        "system_instruction": None,
        "tools": [],
    }
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Mock the generate_content_stream to return empty response
    mock_response = AsyncMock()
    mock_response.__aiter__.return_value = []
    service._client.aio.models.generate_content_stream.return_value = mock_response

    # Mock tracing components
    with patch("opentelemetry.trace.get_tracer") as mock_get_tracer:
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        # Execute _process_context
        await service._process_context(context)

        # Verify that adapter's get_messages_for_logging was called (at least once by tracing)
        assert mock_adapter.get_messages_for_logging.call_count >= 1, (
            "Adapter's get_messages_for_logging should be called"
        )
        # Verify it was called with the context
        mock_adapter.get_messages_for_logging.assert_any_call(context)

        # Verify span.set_attribute was called with serialized empty list
        input_calls = [
            call for call in mock_span.set_attribute.call_args_list if call[0][0] == "input"
        ]

        # Should have exactly one call setting "input"
        assert len(input_calls) == 1, "Expected exactly one call to set_attribute with 'input'"

        # Verify the input is "[]" (empty JSON array)
        input_value = input_calls[0][0][1]
        assert input_value == "[]", f"Expected '[]' but got '{input_value}'"


@pytest.mark.asyncio
async def test_google_tracing_uses_adapter_not_context():
    """Test that tracing uses adapter's get_messages_for_logging instead of context's method."""
    # Create service with mocked client
    service = GoogleLLMService(api_key="test-key", model="gemini-2.0-flash")
    service._client = AsyncMock()

    # Enable tracing for this service instance
    service._tracing_enabled = True

    # Create universal LLMContext (doesn't have get_messages_for_logging)
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
    ]
    context = LLMContext(messages=test_messages)

    # Mock the adapter with specific return value
    mock_adapter = MagicMock()
    adapter_messages = [{"role": "user", "parts": [{"text": "Formatted by adapter"}]}]
    mock_adapter.get_messages_for_logging.return_value = adapter_messages
    mock_adapter.get_llm_invocation_params.return_value = {
        "messages": [],
        "system_instruction": "You are a helpful assistant",
        "tools": [],
    }
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Mock the generate_content_stream
    mock_response = AsyncMock()
    mock_response.__aiter__.return_value = []
    service._client.aio.models.generate_content_stream.return_value = mock_response

    # Mock tracing components
    with patch("opentelemetry.trace.get_tracer") as mock_get_tracer:
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        # Execute _process_context
        await service._process_context(context)

        # Verify adapter's method was called (at least once by tracing)
        assert mock_adapter.get_messages_for_logging.call_count >= 1, (
            "Adapter's get_messages_for_logging should be called"
        )
        # Verify it was called with the context
        mock_adapter.get_messages_for_logging.assert_any_call(context)

        # Verify the serialized messages came from the adapter
        input_calls = [
            call for call in mock_span.set_attribute.call_args_list if call[0][0] == "input"
        ]

        assert len(input_calls) == 1
        input_value = input_calls[0][0][1]
        parsed = json.loads(input_value)

        # Should match what the adapter returned
        assert parsed == adapter_messages, "Messages should come from adapter, not context"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
