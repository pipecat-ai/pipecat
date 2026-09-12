#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Azure Voice Live LLM adapter.

The adapter converts the universal context and tool schemas into the shapes the
Voice Live API accepts. It has no I/O, so these tests call it directly.
"""

import json

import pytest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.azure_voicelive_adapter import AzureVoiceLiveLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext

WEATHER = FunctionSchema(
    name="get_current_weather",
    description="Get the current weather.",
    properties={"location": {"type": "string"}},
    required=["location"],
)


@pytest.fixture
def adapter() -> AzureVoiceLiveLLMAdapter:
    return AzureVoiceLiveLLMAdapter()


def test_provider_id(adapter):
    assert adapter.id_for_llm_specific_messages == "azure-voicelive"


def test_tools_convert_to_the_voice_live_shape(adapter):
    tools = adapter.to_provider_tools_format(ToolsSchema(standard_tools=[WEATHER]))

    assert tools == [
        {
            "type": "function",
            "name": "get_current_weather",
            "description": "Get the current weather.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]


def test_a_single_user_message_is_sent_as_one_item(adapter):
    context = LLMContext([{"role": "user", "content": "Hello there."}])

    params = adapter.get_llm_invocation_params(context)

    assert len(params["messages"]) == 1
    item = params["messages"][0]
    assert item.role == "user"
    assert item.type == "message"
    assert item.content[0].type == "input_text"
    assert item.content[0].text == "Hello there."


def test_a_leading_system_message_becomes_the_session_instruction(adapter):
    context = LLMContext(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi."}]
    )

    params = adapter.get_llm_invocation_params(context)

    assert params["system_instruction"] == "Be brief."
    assert len(params["messages"]) == 1
    assert params["messages"][0].content[0].text == "Hi."


def test_an_init_system_instruction_wins_over_the_context(adapter):
    context = LLMContext([{"role": "system", "content": "From context."}])

    params = adapter.get_llm_invocation_params(context, system_instruction="From init.")

    assert params["system_instruction"] == "From init."


def test_a_system_only_context_sends_no_messages(adapter):
    context = LLMContext([{"role": "system", "content": "Be brief."}])

    params = adapter.get_llm_invocation_params(context)

    assert params["messages"] == []
    assert params["system_instruction"] == "Be brief."


def test_an_empty_context_sends_nothing(adapter):
    params = adapter.get_llm_invocation_params(LLMContext([]))

    assert params["messages"] == []
    assert params["system_instruction"] is None


def test_a_history_is_packed_into_one_user_message(adapter):
    """The realtime API has no way to load a long history."""
    context = LLMContext(
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "And of Spain?"},
        ]
    )

    params = adapter.get_llm_invocation_params(context)

    assert len(params["messages"]) == 1
    packed = params["messages"][0]
    assert packed.role == "user"
    text = packed.content[0].text
    assert "previously saved conversation" in text
    assert "What is the capital of France?" in text
    assert "Paris." in text
    assert "And of Spain?" in text


def test_list_content_is_flattened_to_text(adapter):
    context = LLMContext(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is"},
                    {"type": "text", "text": "the time?"},
                ],
            }
        ]
    )

    params = adapter.get_llm_invocation_params(context)

    assert params["messages"][0].content[0].text == "What is the time?"


def test_an_assistant_tool_call_becomes_a_function_call_item(adapter):
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": "get_current_weather", "arguments": '{"location":"Pune"}'},
            }
        ],
    }

    item = adapter._from_universal_context_message(message)

    assert item.type == "function_call"
    assert item.call_id == "call_1"
    assert item.name == "get_current_weather"
    assert json.loads(item.arguments) == {"location": "Pune"}


def test_an_unhandled_message_role_raises(adapter):
    with pytest.raises(ValueError):
        adapter._from_universal_context_message({"role": "tool", "content": "result"})


def test_context_tools_reach_the_invocation_params(adapter):
    context = LLMContext([{"role": "user", "content": "Weather?"}], [WEATHER])

    params = adapter.get_llm_invocation_params(context)

    assert [t["name"] for t in params["tools"]] == ["get_current_weather"]


def test_messages_for_logging_truncate_large_values(adapter):
    context = LLMContext([{"role": "user", "content": "hello"}])

    assert adapter.get_messages_for_logging(context) == [{"role": "user", "content": "hello"}]
