#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for how AWSBedrockLLMService maps tool_choice to the Converse API."""

from unittest.mock import AsyncMock, patch

import pytest

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.aws.llm import AWSBedrockLLMService

_ABSENT = object()


async def _tool_choice_sent(tool_choice):
    service = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(model="anthropic.claude-test"),
        aws_region="us-east-1",
        aws_access_key="x",
        aws_secret_key="y",
    )
    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="get_weather",
                description="Get the weather",
                properties={"city": {"type": "string"}},
                required=["city"],
            )
        ]
    )
    context = LLMContext(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        tool_choice=tool_choice,
    )

    async def stream():
        yield {"messageStop": {"stopReason": "end_turn"}}

    converse = AsyncMock(return_value={"stream": stream()})
    with patch.object(service, "_create_converse_stream", converse):
        await service._process_context(context)

    request_params = converse.call_args.args[1]
    return request_params["toolConfig"].get("toolChoice", _ABSENT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice, expected",
    [
        ("auto", {"auto": {}}),
        ("required", {"any": {}}),
        ("none", _ABSENT),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            {"tool": {"name": "get_weather"}},
        ),
    ],
)
async def test_tool_choice_mapping(tool_choice, expected):
    assert await _tool_choice_sent(tool_choice) == expected
