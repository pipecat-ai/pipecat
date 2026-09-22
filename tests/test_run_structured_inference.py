#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Structured inference contracts exercised through provider SDKs with mocked HTTP."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from google import genai
from google.auth.credentials import AnonymousCredentials
from google.genai import errors, types
from openai import APIStatusError, AsyncOpenAI
from pydantic import BaseModel

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.vertex.llm import GoogleVertexLLMService
from pipecat.services.llm_service import StructuredInferenceResult
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesHttpLLMService,
    OpenAIResponsesLLMService,
)


class _Person(BaseModel):
    name: str
    age: int


def _response(provider):
    text = '{"name":"Ada","age":36}'
    if provider == "chat":
        return {
            "id": "chat-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 30},
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        }
    if provider.startswith("responses"):
        return {
            "id": "resp-1",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "parallel_tool_calls": True,
            "temperature": 1,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1,
            "output": [
                {
                    "id": "msg-1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {"cached_tokens": 30},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        }
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 15,
            "totalTokenCount": 120,
            "cachedContentTokenCount": 30,
            "thoughtsTokenCount": 5,
        },
        "modelVersion": "test-model",
    }


@pytest_asyncio.fixture(params=["chat", "responses", "responses_http", "google", "google_vertex"])
async def inference(request):
    provider = request.param
    state = SimpleNamespace(
        provider=provider, payload=_response(provider), requests=[], status_code=200
    )

    def handle(http_request):
        state.requests.append(json.loads(http_request.content))
        return httpx.Response(
            state.status_code, json=state.payload, headers={"x-request-id": "request-1"}
        )

    transport = httpx.MockTransport(handle)
    if provider.startswith("google"):
        credentials = AnonymousCredentials()
        credentials.token = "test-token"
        client_kwargs = (
            {
                "vertexai": True,
                "project": "test-project",
                "location": "us-east4",
                "credentials": credentials,
            }
            if provider == "google_vertex"
            else {"api_key": "test-key"}
        )
        client = genai.Client(
            **client_kwargs,
            http_options=types.HttpOptions(async_client_args={"transport": transport}),
        )
        service_cls = GoogleVertexLLMService if provider == "google_vertex" else GoogleLLMService
        kwargs = (
            {"project_id": "test-project"}
            if provider == "google_vertex"
            else {"api_key": "test-key"}
        )
        with (
            patch("pipecat.services.google.llm.genai.Client", return_value=client),
            patch("pipecat.services.google.vertex.llm.Client", return_value=client),
            patch.object(GoogleVertexLLMService, "_get_credentials", return_value=credentials),
        ):
            state.service = service_cls(
                **kwargs,
                settings=service_cls.Settings(
                    model="test-model", system_instruction="Extract people"
                ),
            )
    else:
        client = AsyncOpenAI(
            api_key="test-key", max_retries=0, http_client=httpx.AsyncClient(transport=transport)
        )
        service_cls = {
            "chat": OpenAILLMService,
            "responses": OpenAIResponsesLLMService,
            "responses_http": OpenAIResponsesHttpLLMService,
        }[provider]
        factory = "create_client" if provider == "chat" else "_create_client"
        with patch.object(service_cls, factory, return_value=client):
            state.service = service_cls(
                settings=service_cls.Settings(
                    model="test-model", system_instruction="Extract people"
                )
            )
    assert state.service._client is client
    state.context = LLMContext(messages=[{"role": "user", "content": "Ada, age 36"}])
    state.service.push_frame = AsyncMock()
    state.service.start_llm_usage_metrics = AsyncMock()
    yield state
    if provider.startswith("google"):
        await client.aio.aclose()
        client.close()
    else:
        await client.close()


def _set_text(inference, text):
    if inference.provider == "chat":
        inference.payload["choices"][0]["message"]["content"] = text
    elif inference.provider.startswith("responses"):
        inference.payload["output"][0]["content"][0]["text"] = text
    else:
        inference.payload["candidates"][0]["content"]["parts"][0]["text"] = text


def _assert_usage(result):
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 120
    assert result.usage.cache_read_input_tokens == 30
    assert result.usage.reasoning_tokens == 5


@pytest.mark.asyncio
async def test_success(inference):
    result = await inference.service.run_structured_inference(inference.context, _Person)

    assert isinstance(result, StructuredInferenceResult)
    assert isinstance(result.parsed, _Person)
    assert result.parsed == _Person(name="Ada", age=36)
    assert result.status == "success"
    assert result.reason is None
    assert result.refusal is None
    _assert_usage(result)
    assert result.raw_response is not None
    if not inference.provider.startswith("google"):
        assert result.raw_response._request_id == "request-1"
    assert len(inference.requests) == 1
    inference.service.push_frame.assert_not_called()
    inference.service.start_llm_usage_metrics.assert_not_called()

    body = inference.requests[0]
    if inference.provider == "chat":
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        schema = body["response_format"]["json_schema"]["schema"]
        assert body.get("stream", False) is False
        assert "stream_options" not in body
    elif inference.provider.startswith("responses"):
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["text"]["format"]["strict"] is True
        schema = body["text"]["format"]["schema"]
        assert body["stream"] is False
    else:
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        schema = body["generationConfig"]["responseSchema"]
    assert set(schema["required"]) == {"name", "age"}


@pytest.mark.asyncio
async def test_request_overrides(inference):
    await inference.service.run_structured_inference(
        inference.context, _Person, max_tokens=42, system_instruction="Return a person"
    )
    body = inference.requests[0]
    if inference.provider == "chat":
        assert body["max_completion_tokens"] == 42
        assert body["messages"][0]["content"] == "Return a person"
    elif inference.provider.startswith("responses"):
        assert body["max_output_tokens"] == 42
        assert body["instructions"] == "Return a person"
    else:
        assert body["generationConfig"]["maxOutputTokens"] == 42
        assert body["systemInstruction"]["parts"][0]["text"] == "Return a person"


@pytest.mark.asyncio
async def test_refusal(inference):
    refusal = "I cannot answer this request."
    if inference.provider == "chat":
        inference.payload["choices"][0]["message"].update(content=None, refusal=refusal)
    elif inference.provider.startswith("responses"):
        inference.payload["output"][0]["content"] = [{"type": "refusal", "refusal": refusal}]
    else:
        inference.payload["candidates"][0].update(
            content=None, finishReason="SAFETY", finishMessage=refusal
        )

    result = await inference.service.run_structured_inference(inference.context, _Person)

    assert result.status == "refused"
    assert result.parsed is None
    # The Gemini Developer API SDK does not expose candidate finish messages.
    assert result.refusal == (None if inference.provider == "google" else refusal)
    _assert_usage(result)


@pytest.mark.parametrize("text", ['{"name":', '{"name":"Ada","age":36}'])
@pytest.mark.asyncio
async def test_token_limit_preserves_usage_without_parsing_partial_output(inference, text):
    _set_text(inference, text)
    if inference.provider == "chat":
        reason = "length"
        inference.payload["choices"][0]["finish_reason"] = reason
    elif inference.provider.startswith("responses"):
        reason = "max_output_tokens"
        inference.payload.update(status="incomplete", incomplete_details={"reason": reason})
        inference.payload["output"][0]["status"] = "incomplete"
    else:
        reason = "MAX_TOKENS"
        inference.payload["candidates"][0]["finishReason"] = reason

    result = await inference.service.run_structured_inference(inference.context, _Person)

    assert result.status == "incomplete"
    assert result.parsed is None
    assert result.reason == reason
    assert result.refusal is None
    assert result.raw_response is not None
    _assert_usage(result)


@pytest.mark.parametrize("text", ['{"name":', '{"name":"Ada","age":"unknown"}'])
@pytest.mark.asyncio
async def test_invalid_output_preserves_metadata(inference, text):
    _set_text(inference, text)
    result = await inference.service.run_structured_inference(inference.context, _Person)

    assert result.status == "failed"
    assert result.parsed is None
    assert result.reason == "invalid_response"
    assert result.raw_response is not None
    _assert_usage(result)


@pytest.mark.asyncio
async def test_missing_usage(inference):
    key = "usageMetadata" if inference.provider.startswith("google") else "usage"
    inference.payload.pop(key)
    result = await inference.service.run_structured_inference(inference.context, _Person)
    assert result.status == "success"
    assert result.usage is None


@pytest.mark.asyncio
async def test_empty_output(inference):
    key = (
        "candidates"
        if inference.provider.startswith("google")
        else ("choices" if inference.provider == "chat" else "output")
    )
    inference.payload[key] = []
    result = await inference.service.run_structured_inference(inference.context, _Person)
    assert result.status == "failed"
    assert result.parsed is None
    assert result.reason == "no_structured_output"
    _assert_usage(result)


@pytest.mark.asyncio
async def test_request_errors_propagate(inference):
    inference.status_code = 400
    inference.payload = {
        "error": {
            "message": "Invalid request",
            "code": 400,
            "status": "INVALID_ARGUMENT",
            "type": "invalid_request_error",
        }
    }
    error_type = errors.APIError if inference.provider.startswith("google") else APIStatusError
    with pytest.raises(error_type, match="Invalid request"):
        await inference.service.run_structured_inference(inference.context, _Person)


@pytest.mark.asyncio
async def test_provider_specific_failures(inference):
    if inference.provider == "chat":
        inference.payload["choices"][0].update(finish_reason="content_filter")
        expected_status, reason = "refused", "content_filter"
    elif inference.provider.startswith("responses"):
        inference.payload.update(
            status="failed", error={"code": "server_error", "message": "Generation failed"}
        )
        expected_status, reason = "failed", "server_error"
    else:
        inference.payload["candidates"] = []
        inference.payload["promptFeedback"] = {
            "blockReason": "PROHIBITED_CONTENT",
            "blockReasonMessage": "Blocked prompt",
        }
        expected_status, reason = "refused", "PROHIBITED_CONTENT"

    result = await inference.service.run_structured_inference(inference.context, _Person)

    assert result.status == expected_status
    assert result.reason == reason
    assert result.parsed is None
    if inference.provider.startswith("google"):
        assert result.refusal == "Blocked prompt"
    _assert_usage(result)


@pytest.mark.asyncio
async def test_unexpected_finish_reason(inference):
    if inference.provider == "chat":
        inference.payload["choices"][0]["finish_reason"] = "tool_calls"
        reason = "tool_calls"
    elif inference.provider.startswith("responses"):
        inference.payload["status"] = "cancelled"
        reason = "cancelled"
    else:
        inference.payload["candidates"][0]["finishReason"] = "MALFORMED_FUNCTION_CALL"
        reason = "MALFORMED_FUNCTION_CALL"

    result = await inference.service.run_structured_inference(inference.context, _Person)
    assert result.status == "failed"
    assert result.parsed is None
    assert result.reason == reason
    _assert_usage(result)


@pytest.mark.parametrize("service_cls", [AnthropicLLMService, AWSBedrockLLMService])
@pytest.mark.asyncio
async def test_unsupported_service(service_cls):
    kwargs = {"api_key": "test-key"} if service_cls is AnthropicLLMService else {}
    service = service_cls(**kwargs)
    with pytest.raises(NotImplementedError, match="run_structured_inference"):
        await service.run_structured_inference(LLMContext(), _Person)
