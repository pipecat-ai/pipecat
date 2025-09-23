from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import NOT_GIVEN
from openai import NotGiven
from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN

from pipecat.adapters.services.anthropic_adapter import AnthropicLLMInvocationParams
from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMInvocationParams
from pipecat.adapters.services.gemini_adapter import GeminiLLMInvocationParams
from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService


@pytest.mark.asyncio
async def test_openai_run_inference_with_llm_context():
    """Test run_inference with LLMContext returns expected response."""
    # Create service with mocked client
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(model="gpt-4")
        service._client = AsyncMock()

        # Setup mocks
        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, world!"},
        ]
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=test_messages, tools=OPENAI_NOT_GIVEN, tool_choice=OPENAI_NOT_GIVEN
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)

        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! How can I help you today?"
        service._client.chat.completions.create.return_value = mock_response

        # Execute
        result = await service.run_inference(mock_context)

        # Verify
        assert result == "Hello! How can I help you today?"
        service.get_llm_adapter.assert_called_once()
        mock_adapter.get_llm_invocation_params.assert_called_once_with(mock_context)
        service._client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=test_messages,
            stream=False,
        )


@pytest.mark.asyncio
async def test_openai_run_inference_client_exception():
    """Test that exceptions from the client are propagated."""
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(model="gpt-4")
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=[], tools=OPENAI_NOT_GIVEN, tool_choice=OPENAI_NOT_GIVEN
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)
        service._client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await service.run_inference(mock_context)


@pytest.mark.asyncio
async def test_anthropic_run_inference_with_llm_context():
    """Test run_inference with LLMContext returns expected response for Anthropic."""
    # Create service with mocked client
    service = AnthropicLLMService(api_key="test-key", model="claude-3-sonnet-20240229")
    service._client = AsyncMock()

    # Setup mocks
    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": "Hello, world!"}]
    test_system = "You are a helpful assistant"
    mock_adapter.get_llm_invocation_params.return_value = AnthropicLLMInvocationParams(
        messages=test_messages, system=test_system, tools=[]
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Mock response
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Hello! How can I help you today?"
    service._client.messages.create.return_value = mock_response

    # Execute
    result = await service.run_inference(mock_context)

    # Verify
    assert result == "Hello! How can I help you today?"
    service.get_llm_adapter.assert_called_once()
    mock_adapter.get_llm_invocation_params.assert_called_once_with(
        mock_context, enable_prompt_caching=False
    )
    service._client.messages.create.assert_called_once_with(
        model="claude-3-sonnet-20240229",
        messages=test_messages,
        system=test_system,
        max_tokens=8192,
        stream=False,
    )


@pytest.mark.asyncio
async def test_anthropic_run_inference_client_exception():
    """Test that exceptions from the Anthropic client are propagated."""
    service = AnthropicLLMService(api_key="test-key", model="claude-3-sonnet-20240229")
    service._client = AsyncMock()

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = AnthropicLLMInvocationParams(
        messages=[], system="Test system", tools=[]
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)
    service._client.messages.create.side_effect = Exception("Anthropic API Error")

    with pytest.raises(Exception, match="Anthropic API Error"):
        await service.run_inference(mock_context)


@pytest.mark.asyncio
async def test_google_run_inference_with_llm_context():
    """Test run_inference with LLMContext returns expected response for Google."""
    # Create service with mocked client
    service = GoogleLLMService(api_key="test-key", model="gemini-2.0-flash")
    service._client = AsyncMock()

    # Setup mocks
    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": "Hello, world!"}]
    test_system = "You are a helpful assistant"
    mock_adapter.get_llm_invocation_params.return_value = GeminiLLMInvocationParams(
        messages=test_messages, system_instruction=test_system, tools=NotGiven()
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Mock response
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content = MagicMock()
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Hello! How can I help you today?"
    service._client.aio = AsyncMock()
    service._client.aio.models = AsyncMock()
    service._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    # Execute
    result = await service.run_inference(mock_context)

    # Verify
    assert result == "Hello! How can I help you today?"
    service.get_llm_adapter.assert_called_once()
    mock_adapter.get_llm_invocation_params.assert_called_once_with(mock_context)
    service._client.aio.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_google_run_inference_client_exception():
    """Test that exceptions from the Google client are propagated."""
    service = GoogleLLMService(api_key="test-key", model="gemini-2.0-flash")
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
        await service.run_inference(mock_context)


@pytest.mark.asyncio
async def test_aws_bedrock_run_inference_with_llm_context():
    """Test run_inference with LLMContext returns expected response for AWS Bedrock."""
    # Create service and patch the session client method
    service = AWSBedrockLLMService(model="anthropic.claude-3-sonnet-20240229-v1:0")

    # Setup mocks
    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    test_messages = [{"role": "user", "content": [{"text": "Hello, world!"}]}]
    test_system = [{"text": "You are a helpful assistant"}]
    mock_adapter.get_llm_invocation_params.return_value = AWSBedrockLLMInvocationParams(
        messages=test_messages, system=test_system, tools=[], tool_choice=None
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Mock the client and response
    mock_client = AsyncMock()
    mock_response = {
        "output": {"message": {"content": [{"text": "Hello! How can I help you today?"}]}}
    }
    mock_client.converse.return_value = mock_response

    # Patch the _aws_session.client method to be an async context manager
    async def mock_client_cm(*args, **kwargs):
        return mock_client

    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch.object(service._aws_session, "client", return_value=mock_context_manager):
        # Execute
        result = await service.run_inference(mock_context)

        # Verify
        assert result == "Hello! How can I help you today?"
        service.get_llm_adapter.assert_called_once()
        mock_adapter.get_llm_invocation_params.assert_called_once_with(mock_context)
        mock_client.converse.assert_called_once()


@pytest.mark.asyncio
async def test_aws_bedrock_run_inference_client_exception():
    """Test that exceptions from the AWS Bedrock client are propagated."""
    service = AWSBedrockLLMService(model="anthropic.claude-3-sonnet-20240229-v1:0")

    mock_context = MagicMock(spec=LLMContext)
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = AWSBedrockLLMInvocationParams(
        messages=[], system=[{"text": "Test system"}], tools=[], tool_choice=None
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    # Mock AWS client to raise exception
    mock_client = AsyncMock()
    mock_client.converse.side_effect = Exception("Bedrock API Error")

    # Patch the _aws_session.client method to be an async context manager
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)

    with patch.object(service._aws_session, "client", return_value=mock_context_manager):
        with pytest.raises(Exception, match="Bedrock API Error"):
            await service.run_inference(mock_context)
