#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.utils.tracing.service_attributes import (
    add_gemini_live_span_attributes,
    add_openai_realtime_span_attributes,
)


class FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def _tools_schema() -> ToolsSchema:
    function = FunctionSchema(
        name="get_weather",
        description="Get the weather in a given location",
        properties={
            "location": {
                "type": "string",
                "description": "The city, e.g. San Francisco",
            },
        },
        required=["location"],
    )
    return ToolsSchema(
        standard_tools=[function],
        custom_tools={
            AdapterType.OPENAI: [{"type": "function", "name": "openai_custom"}],
            AdapterType.GEMINI: [
                {"function_declarations": [{"name": "gemini_custom"}]},
            ],
        },
    )


def test_openai_realtime_span_attributes_accept_tools_schema():
    span = FakeSpan()

    add_openai_realtime_span_attributes(
        span=span,
        service_name="OpenAIRealtimeLLMService",
        model="gpt-realtime-2",
        operation_name="llm_setup",
        tools=_tools_schema(),
    )

    assert span.attributes["tools.count"] == 2
    assert span.attributes["tools.available"] is True
    assert span.attributes["tools.names"] == "get_weather,openai_custom"


def test_gemini_live_span_attributes_accept_tools_schema():
    span = FakeSpan()

    add_gemini_live_span_attributes(
        span=span,
        service_name="GeminiLiveLLMService",
        model="gemini-live",
        operation_name="llm_setup",
        tools=_tools_schema(),
    )

    assert span.attributes["tools.count"] == 2
    assert span.attributes["tools.available"] is True
    assert span.attributes["tools.names"] == "get_weather,gemini_custom"
