#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.sarvam.llm import SarvamLLMService
from pipecat.utils.types import NotGiven


class _FakeSarvamError(Exception):
    def __init__(self, body):
        super().__init__("Request failed")
        self.body = body


class _FakeSarvamResponseError(Exception):
    pass


def _contains_pipecat_not_given(value) -> bool:
    if isinstance(value, NotGiven):
        return True
    if isinstance(value, dict):
        return any(_contains_pipecat_not_given(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_pipecat_not_given(v) for v in value)
    return False


def test_sarvam_llm_constructor_rejects_unsupported_model():
    with pytest.raises(ValueError, match="Unsupported Sarvam LLM model"):
        SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-100b"),
        )


@pytest.mark.parametrize(
    "model",
    [
        "gemma4",
        "glm5.2",
        "sarvam-105b",
        "sarvam-105b-conversations",
    ],
)
def test_sarvam_llm_constructor_accepts_supported_models(model):
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model=model),
        )
    assert service._settings.model == model


def test_sarvam_llm_default_model_is_sarvam_105b():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(api_key="test-key")
    assert service._settings.model == "sarvam-105b"


@pytest.mark.parametrize(
    "model, expected_base_url",
    [
        ("sarvam-105b", "https://api.sarvam.ai/v2"),
        ("gemma4", "https://api.sarvam.ai/v2"),
        ("glm5.2", "https://api.sarvam.ai/v2"),
        ("sarvam-105b-conversations", "https://api.sarvam.ai/v1"),
    ],
)
def test_sarvam_llm_default_base_url_resolved_per_model(model, expected_base_url):
    with patch.object(
        OpenAILLMService,
        "create_client",
        return_value=AsyncMock(),
    ) as create_mock:
        SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model=model),
        )
    assert create_mock.call_args.kwargs["base_url"] == expected_base_url


def test_sarvam_llm_explicit_base_url_overrides_model_default():
    with patch.object(
        OpenAILLMService,
        "create_client",
        return_value=AsyncMock(),
    ) as create_mock:
        SarvamLLMService(
            api_key="test-key",
            base_url="https://custom.example.com/v3",
            settings=SarvamLLMService.Settings(model="sarvam-105b-conversations"),
        )
    assert create_mock.call_args.kwargs["base_url"] == "https://custom.example.com/v3"


def test_sarvam_llm_create_client_injects_required_headers():
    with patch.object(
        OpenAILLMService,
        "create_client",
        return_value=AsyncMock(),
    ) as create_mock:
        SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )

    kwargs = create_mock.call_args.kwargs
    headers = kwargs["default_headers"]
    assert headers["api-subscription-key"] == "test-key"
    assert headers["User-Agent"] == sdk_headers()["User-Agent"]


@pytest.mark.parametrize("reasoning_effort", ["low", "medium", "high"])
def test_sarvam_llm_reasoning_effort_passed_to_request(reasoning_effort):
    with patch.object(SarvamLLMService, "create_client"):
        settings = SarvamLLMService.Settings(
            model="sarvam-105b",
            reasoning_effort=reasoning_effort,
        )
        service = SarvamLLMService(
            api_key="test-key",
            settings=settings,
        )

    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    built_params = service.build_chat_completion_params(invocation)
    assert built_params["reasoning_effort"] == reasoning_effort


def test_sarvam_llm_create_client_merges_default_headers():
    with patch.object(
        OpenAILLMService,
        "create_client",
        return_value=AsyncMock(),
    ) as create_mock:
        SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
            default_headers={
                "X-Test-Header": "enabled",
                "User-Agent": "custom-agent",
            },
        )

    kwargs = create_mock.call_args.kwargs
    headers = kwargs["default_headers"]
    assert headers["X-Test-Header"] == "enabled"
    assert headers["api-subscription-key"] == "test-key"
    assert headers["User-Agent"] == sdk_headers()["User-Agent"]


def test_sarvam_llm_build_params_excludes_pipecat_not_given_sentinel():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )

    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    built_params = service.build_chat_completion_params(invocation)

    assert not _contains_pipecat_not_given(built_params)


def test_sarvam_llm_omits_optional_sarvam_fields_when_unset():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )

    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    built_params = service.build_chat_completion_params(invocation)

    assert "wiki_grounding" not in built_params
    assert "reasoning_effort" not in built_params


def test_sarvam_llm_wiki_grounding_passed_via_extra_body():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(
                model="sarvam-105b",
                wiki_grounding=True,
            ),
        )

    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    built_params = service.build_chat_completion_params(invocation)

    assert "wiki_grounding" not in built_params
    assert built_params["extra_body"]["wiki_grounding"] is True


def test_sarvam_llm_build_chat_completion_params_filters_unsupported_fields():
    with patch.object(SarvamLLMService, "create_client"):
        settings = SarvamLLMService.Settings(
            model="sarvam-105b",
            temperature=0.7,
            max_tokens=128,
            wiki_grounding=False,
            reasoning_effort="medium",
        )
        service = SarvamLLMService(
            api_key="test-key",
            settings=settings,
        )

        invocation = OpenAILLMInvocationParams(
            messages=[{"role": "user", "content": "Hello"}],
            tools=OPENAI_NOT_GIVEN,
            tool_choice=OPENAI_NOT_GIVEN,
        )
        built_params = service.build_chat_completion_params(invocation)

    assert "stream_options" not in built_params
    assert "max_completion_tokens" not in built_params
    assert "service_tier" not in built_params
    assert "wiki_grounding" not in built_params
    assert built_params["extra_body"]["wiki_grounding"] is False
    assert built_params["reasoning_effort"] == "medium"


def test_sarvam_llm_build_params_forward_core_and_extra_fields():
    with patch.object(SarvamLLMService, "create_client"):
        settings = SarvamLLMService.Settings(
            model="sarvam-105b",
            temperature=0.9,
            top_p=0.8,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            seed=11,
            max_tokens=222,
            extra={"n": 2},
        )
        service = SarvamLLMService(
            api_key="test-key",
            settings=settings,
        )

    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    built_params = service.build_chat_completion_params(invocation)

    assert built_params["temperature"] == 0.9
    assert built_params["top_p"] == 0.8
    assert built_params["frequency_penalty"] == 0.1
    assert built_params["presence_penalty"] == 0.2
    assert built_params["seed"] == 11
    assert built_params["max_tokens"] == 222
    assert built_params["n"] == 2


def test_sarvam_llm_extra_body_merges_with_user_extra():
    with patch.object(SarvamLLMService, "create_client"):
        settings = SarvamLLMService.Settings(
            model="sarvam-105b",
            wiki_grounding=True,
            extra={"extra_body": {"user_field": 1}},
        )
        service = SarvamLLMService(
            api_key="test-key",
            settings=settings,
        )

    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    built_params = service.build_chat_completion_params(invocation)

    assert built_params["extra_body"]["user_field"] == 1
    assert built_params["extra_body"]["wiki_grounding"] is True


@pytest.mark.asyncio
async def test_sarvam_llm_tool_choice_requires_non_empty_tools():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )
        service._client = AsyncMock()

    pushed_errors = []

    async def mock_push_error(error_msg, **kw):
        pushed_errors.append(error_msg)

    service.push_error = mock_push_error

    context = LLMContext(
        messages=[{"role": "user", "content": "Hello"}],
        tool_choice="required",
    )
    await service._process_context(context)

    assert len(pushed_errors) == 1
    service._client.chat.completions.create.assert_not_called()
    assert "requires non-empty `tools`" in pushed_errors[0]


def test_sarvam_llm_tool_choice_with_tools_is_allowed():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )

    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Lookup weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="required",
    )

    built_params = service.build_chat_completion_params(invocation)
    assert built_params["tool_choice"] == "required"
    assert built_params["tools"][0]["function"]["name"] == "lookup_weather"


@pytest.mark.asyncio
async def test_sarvam_llm_rejects_image_input_on_non_vision_model():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )
        service._client = AsyncMock()

    pushed_errors = []

    async def mock_push_error(error_msg, **kw):
        pushed_errors.append(error_msg)

    service.push_error = mock_push_error

    context = LLMContext(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo...",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ],
    )
    await service._process_context(context)

    assert len(pushed_errors) == 1
    service._client.chat.completions.create.assert_not_called()
    assert "does not support image input" in pushed_errors[0]


def test_sarvam_llm_accepts_image_input_on_gemma4():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="gemma4"),
        )

    invocation = OpenAILLMInvocationParams(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo...",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    built_params = service.build_chat_completion_params(invocation)
    assert built_params["model"] == "gemma4"


@pytest.mark.asyncio
async def test_sarvam_llm_conversations_rejects_reasoning_effort():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(
                model="sarvam-105b-conversations",
                reasoning_effort="high",
            ),
        )
        service._client = AsyncMock()

    pushed_errors = []

    async def mock_push_error(error_msg, **kw):
        pushed_errors.append(error_msg)

    service.push_error = mock_push_error

    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
    await service._process_context(context)

    assert len(pushed_errors) == 1
    service._client.chat.completions.create.assert_not_called()
    assert "does not support reasoning_effort" in pushed_errors[0]


@pytest.mark.asyncio
async def test_sarvam_llm_conversations_rejects_wiki_grounding():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(
                model="sarvam-105b-conversations",
                wiki_grounding=True,
            ),
        )
        service._client = AsyncMock()

    pushed_errors = []

    async def mock_push_error(error_msg, **kw):
        pushed_errors.append(error_msg)

    service.push_error = mock_push_error

    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
    await service._process_context(context)

    assert len(pushed_errors) == 1
    service._client.chat.completions.create.assert_not_called()
    assert "does not support wiki_grounding" in pushed_errors[0]


@pytest.mark.asyncio
async def test_sarvam_llm_conversations_rejects_image_input():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b-conversations"),
        )
        service._client = AsyncMock()

    pushed_errors = []

    async def mock_push_error(error_msg, **kw):
        pushed_errors.append(error_msg)

    service.push_error = mock_push_error

    context = LLMContext(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo...",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ],
    )
    await service._process_context(context)

    assert len(pushed_errors) == 1
    service._client.chat.completions.create.assert_not_called()
    assert "does not support image input" in pushed_errors[0]


def test_sarvam_llm_conversations_supports_tool_calling():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b-conversations"),
        )

    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Weather in Delhi?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="auto",
    )
    built_params = service.build_chat_completion_params(invocation)
    assert built_params["tool_choice"] == "auto"
    assert built_params["tools"][0]["function"]["name"] == "get_weather"


def test_sarvam_llm_glm52_supports_reasoning_effort():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(
                model="glm5.2",
                reasoning_effort="high",
            ),
        )

    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    built_params = service.build_chat_completion_params(invocation)
    assert built_params["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_sarvam_llm_glm52_rejects_wiki_grounding():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(
                model="glm5.2",
                wiki_grounding=True,
            ),
        )
        service._client = AsyncMock()

    pushed_errors = []

    async def mock_push_error(error_msg, **kw):
        pushed_errors.append(error_msg)

    service.push_error = mock_push_error

    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
    await service._process_context(context)

    assert len(pushed_errors) == 1
    service._client.chat.completions.create.assert_not_called()
    assert "does not support wiki_grounding" in pushed_errors[0]


@pytest.mark.asyncio
async def test_sarvam_llm_glm52_rejects_image_input():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="glm5.2"),
        )
        service._client = AsyncMock()

    pushed_errors = []

    async def mock_push_error(error_msg, **kw):
        pushed_errors.append(error_msg)

    service.push_error = mock_push_error

    context = LLMContext(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo...",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ],
    )
    await service._process_context(context)

    assert len(pushed_errors) == 1
    service._client.chat.completions.create.assert_not_called()
    assert "does not support image input" in pushed_errors[0]


@pytest.mark.asyncio
async def test_sarvam_llm_update_settings_applies_runtime_sarvam_fields():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )

    changed = await service._update_settings(
        SarvamLLMService.Settings(wiki_grounding=True, reasoning_effort="low")
    )

    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    built_params = service.build_chat_completion_params(invocation)

    assert "wiki_grounding" in changed
    assert "reasoning_effort" in changed
    assert built_params["extra_body"]["wiki_grounding"] is True
    assert built_params["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_sarvam_llm_vision_validation_skips_non_dict_messages():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )
        service._client = AsyncMock()

    pushed_errors = []

    async def mock_push_error(error_msg, **kw):
        pushed_errors.append(error_msg)

    service.push_error = mock_push_error

    # Non-dict entries should be skipped by vision validation, not cause an error
    mock_adapter = MagicMock()
    mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
        messages=[
            "not a dict",
            None,
            {"role": "user", "content": "Hello"},
        ],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    service.get_llm_adapter = MagicMock(return_value=mock_adapter)

    with patch.object(OpenAILLMService, "_process_context", new=AsyncMock()) as base_process:
        await service._process_context(LLMContext())

    assert len(pushed_errors) == 0
    base_process.assert_awaited_once()


@pytest.mark.asyncio
async def test_sarvam_llm_run_inference_with_llm_context():
    with patch.object(SarvamLLMService, "create_client"):
        settings = SarvamLLMService.Settings(
            model="sarvam-105b",
            temperature=0.7,
            max_tokens=100,
            frequency_penalty=0.5,
            seed=42,
            wiki_grounding=True,
            reasoning_effort="high",
        )
        service = SarvamLLMService(
            api_key="test-key",
            settings=settings,
        )
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, world!"},
        ]
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=test_messages,
            tools=OPENAI_NOT_GIVEN,
            tool_choice=OPENAI_NOT_GIVEN,
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! How can I help you today?"
        service._client.chat.completions.create.return_value = mock_response

        result = await service.run_inference(mock_context)

    assert result == "Hello! How can I help you today?"
    call_kwargs = service._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is False
    assert "stream_options" not in call_kwargs
    assert "max_completion_tokens" not in call_kwargs
    assert "service_tier" not in call_kwargs
    assert "wiki_grounding" not in call_kwargs
    assert call_kwargs["extra_body"]["wiki_grounding"] is True
    assert call_kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_sarvam_llm_run_inference_with_llm_context_object():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b", wiki_grounding=True),
        )
        service._client = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from LLM context"
        service._client.chat.completions.create.return_value = mock_response

        result = await service.run_inference(context)

    assert result == "Hello from LLM context"
    call_kwargs = service._client.chat.completions.create.call_args.kwargs
    assert "wiki_grounding" not in call_kwargs
    assert call_kwargs["extra_body"]["wiki_grounding"] is True


@pytest.mark.asyncio
async def test_sarvam_llm_run_inference_max_tokens_override():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b", max_tokens=100),
        )
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=[{"role": "user", "content": "Hello"}],
            tools=OPENAI_NOT_GIVEN,
            tool_choice=OPENAI_NOT_GIVEN,
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "token override"
        service._client.chat.completions.create.return_value = mock_response

        await service.run_inference(mock_context, max_tokens=23)

    call_kwargs = service._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 23
    assert "max_completion_tokens" not in call_kwargs


@pytest.mark.asyncio
async def test_sarvam_llm_run_inference_forwards_system_instruction():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=[{"role": "user", "content": "Hello"}],
            tools=OPENAI_NOT_GIVEN,
            tool_choice=OPENAI_NOT_GIVEN,
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "instruction override"
        service._client.chat.completions.create.return_value = mock_response

        await service.run_inference(
            mock_context,
            system_instruction="You are a concise assistant.",
        )

    adapter_kwargs = mock_adapter.get_llm_invocation_params.call_args.kwargs
    assert adapter_kwargs["system_instruction"] == "You are a concise assistant."


@pytest.mark.asyncio
async def test_sarvam_llm_timeout_errors_are_not_wrapped():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=[{"role": "user", "content": "Hello"}],
            tools=OPENAI_NOT_GIVEN,
            tool_choice=OPENAI_NOT_GIVEN,
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)
        service._client.chat.completions.create.side_effect = TimeoutError()

        with pytest.raises(asyncio.TimeoutError):
            await service.run_inference(mock_context)


@pytest.mark.asyncio
async def test_sarvam_llm_run_inference_surfaces_raw_server_error():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )
        service._client = AsyncMock()

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=[{"role": "user", "content": "Hello"}],
            tools=OPENAI_NOT_GIVEN,
            tool_choice=OPENAI_NOT_GIVEN,
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)
        service._client.chat.completions.create.side_effect = _FakeSarvamError(
            {"error": {"message": "model is not available for this account"}}
        )

        with pytest.raises(_FakeSarvamError) as exc_info:
            await service.run_inference(mock_context)

    assert exc_info.value.body["error"]["message"] == "model is not available for this account"


@pytest.mark.asyncio
async def test_sarvam_llm_get_chat_completions_propagates_response_error():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )
        service._client = AsyncMock()
        service._client.chat.completions.create.side_effect = _FakeSarvamResponseError(
            "invalid request format"
        )

        mock_context = MagicMock(spec=LLMContext)
        mock_adapter = MagicMock()
        mock_adapter.get_llm_invocation_params.return_value = OpenAILLMInvocationParams(
            messages=[{"role": "user", "content": "Hello"}],
            tools=OPENAI_NOT_GIVEN,
            tool_choice=OPENAI_NOT_GIVEN,
        )
        service.get_llm_adapter = MagicMock(return_value=mock_adapter)

        with pytest.raises(_FakeSarvamResponseError, match="invalid request format"):
            await service.get_chat_completions(mock_context)


@pytest.mark.asyncio
async def test_sarvam_llm_process_frame_surfaces_raw_server_error():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )
        service._client = AsyncMock()
        service._client.chat.completions.create.side_effect = _FakeSarvamError(
            {"error": {"message": "tool schema is invalid"}}
        )

        pushed_errors = []

        async def mock_push_error(error_msg, exception=None):
            pushed_errors.append({"error_msg": error_msg, "exception": exception})

        service.push_error = mock_push_error
        service.push_frame = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()

        context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
        await service.process_frame(
            LLMContextFrame(context),
            FrameDirection.DOWNSTREAM,
        )

    assert len(pushed_errors) == 1
    assert "Error during completion: Request failed" in pushed_errors[0]["error_msg"]
    assert isinstance(pushed_errors[0]["exception"], _FakeSarvamError)
    assert pushed_errors[0]["exception"].body["error"]["message"] == "tool schema is invalid"


@pytest.mark.asyncio
async def test_sarvam_llm_stream_closed_on_cancellation():
    with patch.object(SarvamLLMService, "create_client"):
        service = SarvamLLMService(
            api_key="test-key",
            settings=SarvamLLMService.Settings(model="sarvam-105b"),
        )
        service._client = AsyncMock()

        stream_closed = False

        class MockAsyncStream:
            def __init__(self):
                self.iteration_count = 0

            async def close(self):
                nonlocal stream_closed
                stream_closed = True

            def __aiter__(self):
                return self

            async def __anext__(self):
                self.iteration_count += 1
                if self.iteration_count > 1:
                    raise asyncio.CancelledError()
                mock_chunk = AsyncMock()
                mock_chunk.usage = None
                mock_chunk.model = None
                mock_chunk.choices = []
                return mock_chunk

        mock_stream = MockAsyncStream()
        service._client.chat.completions.create.return_value = mock_stream
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
        with pytest.raises(asyncio.CancelledError):
            await service._process_context(context)

    assert stream_closed
