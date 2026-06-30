#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import NotGiven
from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN
from pydantic import BaseModel

from pipecat.adapters.services.gemini_adapter import GeminiLLMInvocationParams
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesHttpLLMService,
    OpenAIResponsesLLMService,
)


class _Person(BaseModel):
    name: str
    age: int


# --- OpenAI chat completions ---


@pytest.mark.asyncio
async def test_openai_run_structured_inference_with_llm_context():
    """Test run_structured_inference returns a validated model and drops streaming params."""
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(
            settings=OpenAILLMService.Settings(model="gpt-4", max_tokens=100)
        )
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        test_messages = [{"role": "user", "content": "Ada Lovelace, age 36"}]
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=test_messages, tools=OPENAI_NOT_GIVEN, tool_choice=OPENAI_NOT_GIVEN
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)

        expected = _Person(name="Ada Lovelace", age=36)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = expected
        service._client.chat.completions.parse.return_value = mock_response

        result = await service.run_structured_inference(mock_context, _Person)

        assert result == expected
        call_kwargs = service._client.chat.completions.parse.call_args.kwargs
        assert call_kwargs["response_format"] is _Person
        # The structured-output helper rejects streaming params.
        assert "stream" not in call_kwargs
        assert "stream_options" not in call_kwargs
        assert call_kwargs["messages"] == test_messages


@pytest.mark.asyncio
async def test_openai_run_structured_inference_max_tokens_override():
    """Test max_tokens overrides the configured value."""
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(
            settings=OpenAILLMService.Settings(model="gpt-4", max_tokens=100)
        )
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=[{"role": "user", "content": "Hi"}],
            tools=OPENAI_NOT_GIVEN,
            tool_choice=OPENAI_NOT_GIVEN,
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = _Person(name="A", age=1)
        service._client.chat.completions.parse.return_value = mock_response

        await service.run_structured_inference(mock_context, _Person, max_tokens=42)

        call_kwargs = service._client.chat.completions.parse.call_args.kwargs
        # Mirrors run_inference: the override lands on max_completion_tokens when that
        # key is present in the built params (always, for the OpenAI adapter).
        assert call_kwargs["max_completion_tokens"] == 42


@pytest.mark.asyncio
async def test_openai_run_structured_inference_client_exception():
    """Test that exceptions from the client are propagated."""
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="gpt-4"))
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=[], tools=OPENAI_NOT_GIVEN, tool_choice=OPENAI_NOT_GIVEN
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)
        service._client.chat.completions.parse.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await service.run_structured_inference(mock_context, _Person)


# --- OpenAI Responses (WebSocket + HTTP) ---


@pytest.mark.parametrize(
    "service_cls",
    [OpenAIResponsesLLMService, OpenAIResponsesHttpLLMService],
)
@pytest.mark.asyncio
async def test_openai_responses_run_structured_inference(service_cls):
    """Test Responses run_structured_inference returns output_parsed with text_format set."""
    with patch.object(service_cls, "_create_client"):
        service = service_cls(
            settings=service_cls.Settings(model="gpt-4.1", system_instruction="You extract people"),
        )
        service._client = AsyncMock()

        context = LLMContext(messages=[{"role": "user", "content": "Ada Lovelace, age 36"}])

        expected = _Person(name="Ada Lovelace", age=36)
        mock_response = MagicMock()
        mock_response.output_parsed = expected
        service._client.responses.parse = AsyncMock(return_value=mock_response)

        result = await service.run_structured_inference(context, _Person)

        assert result == expected
        call_kwargs = service._client.responses.parse.call_args.kwargs
        assert call_kwargs["text_format"] is _Person
        assert call_kwargs["stream"] is False
        assert call_kwargs["input"] == [{"role": "user", "content": "Ada Lovelace, age 36"}]


@pytest.mark.asyncio
async def test_openai_responses_run_structured_inference_max_tokens_override():
    """Test max_tokens overrides max_output_tokens for the Responses variant."""
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService(
            settings=OpenAIResponsesLLMService.Settings(model="gpt-4.1", max_completion_tokens=500),
        )
        service._client = AsyncMock()

        context = LLMContext(messages=[{"role": "user", "content": "Ada, 36"}])
        mock_response = MagicMock()
        mock_response.output_parsed = _Person(name="Ada", age=36)
        service._client.responses.parse = AsyncMock(return_value=mock_response)

        await service.run_structured_inference(context, _Person, max_tokens=200)

        call_kwargs = service._client.responses.parse.call_args.kwargs
        assert call_kwargs["max_output_tokens"] == 200


@pytest.mark.asyncio
async def test_openai_responses_run_structured_inference_client_exception():
    """Test that exceptions from the Responses client are propagated."""
    with patch.object(OpenAIResponsesLLMService, "_create_client"):
        service = OpenAIResponsesLLMService()
        service._client = AsyncMock()

        context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
        service._client.responses.parse = AsyncMock(side_effect=Exception("API Error"))

        with pytest.raises(Exception, match="API Error"):
            await service.run_structured_inference(context, _Person)


# --- Google Gemini ---


@pytest.mark.asyncio
async def test_google_run_structured_inference_with_llm_context():
    """Test Google run_structured_inference returns response.parsed with schema in config."""
    service = GoogleLLMService(
        api_key="test-key", settings=GoogleLLMService.Settings(model="gemini-2.0-flash")
    )
    service._client = AsyncMock()

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": "Ada Lovelace, age 36"}]
    mock_adapter.get_llm_invocation_params.return_value = GeminiLLMInvocationParams(
        messages=test_messages, system_instruction="You extract people", tools=NotGiven()
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    expected = _Person(name="Ada Lovelace", age=36)
    mock_response = MagicMock()
    mock_response.parsed = expected
    service._client.aio = AsyncMock()
    service._client.aio.models = AsyncMock()
    service._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    result = await service.run_structured_inference(mock_context, _Person)

    assert result == expected
    call_kwargs = service._client.aio.models.generate_content.call_args.kwargs
    config = call_kwargs["config"]
    assert config.response_schema is _Person
    assert config.response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_google_run_structured_inference_client_exception():
    """Test that exceptions from the Google client are propagated."""
    service = GoogleLLMService(
        api_key="test-key", settings=GoogleLLMService.Settings(model="gemini-2.0-flash")
    )
    service._client = AsyncMock()

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = GeminiLLMInvocationParams(
        messages=[], system_instruction="Test system", tools=NotGiven()
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)
    service._client.aio = AsyncMock()
    service._client.aio.models = AsyncMock()
    service._client.aio.models.generate_content = AsyncMock(
        side_effect=Exception("Google API Error")
    )

    with pytest.raises(Exception, match="Google API Error"):
        await service.run_structured_inference(mock_context, _Person)


# --- Services without native structured-output support ---


@pytest.mark.asyncio
async def test_anthropic_run_structured_inference_not_implemented():
    """Anthropic has no native structured output and inherits the base NotImplementedError."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-3-sonnet-20240229")
    )
    mock_context = MagicMock(spec=LLMContext)

    with pytest.raises(NotImplementedError, match="run_structured_inference"):
        await service.run_structured_inference(mock_context, _Person)


@pytest.mark.asyncio
async def test_aws_bedrock_run_structured_inference_not_implemented():
    """AWS Bedrock has no native structured output and inherits the base NotImplementedError."""
    service = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(model="anthropic.claude-3-sonnet-20240229-v1:0")
    )
    mock_context = MagicMock(spec=LLMContext)

    with pytest.raises(NotImplementedError, match="run_structured_inference"):
        await service.run_structured_inference(mock_context, _Person)
