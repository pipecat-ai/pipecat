#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Integration tests for the mock LLM service."""

import pytest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    FunctionCallsFromLLMInfoFrame,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.tests.utils import run_test

from mock_llm_service import MockLLMService


@pytest.mark.asyncio
async def test_mock_llm_text_streaming():
    """Test that mock LLM service correctly streams text chunks and generates frames."""
    # Create text chunks
    chunks = MockLLMService.create_text_chunks("Hello, world!", chunk_size=7)

    # Create mock service with predefined chunks
    llm = MockLLMService(
        mock_chunks=chunks,
        chunk_delay=0.001,  # Small delay for testing
    )

    # Create context
    messages = [{"role": "user", "content": "Say hello"}]
    context = LLMContext(messages)

    # Create pipeline and run test
    pipeline = Pipeline([llm])

    frames_to_send = [LLMContextFrame(context)]

    received_frames, _ = await run_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=[
            LLMFullResponseStartFrame,
            LLMTextFrame,  # "Hello, "
            LLMTextFrame,  # "world!"
            LLMFullResponseEndFrame,
        ],
    )

    # Verify text content
    text_frames = [f for f in received_frames if isinstance(f, LLMTextFrame)]
    assert len(text_frames) == 2
    assert text_frames[0].text == "Hello, "
    assert text_frames[1].text == "world!"

    # Verify complete text
    full_text = "".join(f.text for f in text_frames)
    assert full_text == "Hello, world!"


@pytest.mark.asyncio
async def test_mock_llm_function_calling():
    """Test that mock LLM service correctly processes function call chunks."""
    # Create function call chunks
    chunks = MockLLMService.create_function_call_chunks(
        function_name="get_current_weather",
        arguments={"location": "San Francisco, CA", "format": "celsius"},
        tool_call_id="call_weather_123",
    )

    # Create mock service
    llm = MockLLMService(mock_chunks=chunks, chunk_delay=0.001)

    # Track function call
    function_called = False
    received_params = None

    async def mock_weather_function(params: FunctionCallParams):
        nonlocal function_called, received_params
        function_called = True
        received_params = params
        return {"temperature": 20, "condition": "sunny"}

    # Register function
    llm.register_function("get_current_weather", mock_weather_function)

    # Create tools schema
    weather_function = FunctionSchema(
        name="get_current_weather",
        description="Get the current weather",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use.",
            },
        },
        required=["location"],
    )
    tools = ToolsSchema(standard_tools=[weather_function])

    # Create context with tools
    messages = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "What's the weather in San Francisco?"},
    ]
    context = LLMContext(messages, tools)

    # Run test
    pipeline = Pipeline([llm])
    frames_to_send = [LLMContextFrame(context)]

    received_frames, _ = await run_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=[
            LLMFullResponseStartFrame,
            FunctionCallsFromLLMInfoFrame,
            FunctionCallsStartedFrame,
            LLMFullResponseEndFrame,
            FunctionCallInProgressFrame,
        ],
    )

    # Verify function was called
    assert function_called, "Function should have been called"
    assert received_params is not None
    assert received_params.function_name == "get_current_weather"
    assert received_params.tool_call_id == "call_weather_123"
    assert received_params.arguments == {"location": "San Francisco, CA", "format": "celsius"}

    # Verify FunctionCallsFromLLMInfoFrame was generated
    info_frames = [f for f in received_frames if isinstance(f, FunctionCallsFromLLMInfoFrame)]
    assert len(info_frames) == 1
    assert len(info_frames[0].function_calls) == 1

    func_call = info_frames[0].function_calls[0]
    assert func_call.function_name == "get_current_weather"
    assert func_call.tool_call_id == "call_weather_123"
    assert func_call.arguments == {"location": "San Francisco, CA", "format": "celsius"}


@pytest.mark.asyncio
async def test_mock_llm_multiple_function_calls():
    """Test multiple function calls in a single response."""
    # Create chunks with multiple function calls
    functions = [
        {
            "name": "get_current_weather",
            "arguments": {"location": "London, UK"},
            "tool_call_id": "call_london",
        },
        {
            "name": "get_current_weather",
            "arguments": {"location": "Tokyo, Japan"},
            "tool_call_id": "call_tokyo",
        },
        {
            "name": "get_current_weather",
            "arguments": {"location": "Sydney, Australia"},
            "tool_call_id": "call_sydney",
        },
    ]

    chunks = MockLLMService.create_multiple_function_call_chunks(functions)

    # Create mock service
    llm = MockLLMService(mock_chunks=chunks, chunk_delay=0.001)

    # Track all function calls
    function_calls = []

    async def mock_weather(params: FunctionCallParams):
        function_calls.append(
            {
                "name": params.function_name,
                "location": params.arguments["location"],
                "tool_id": params.tool_call_id,
            }
        )
        return {"temperature": 20, "condition": "varies"}

    llm.register_function("get_current_weather", mock_weather)

    # Create context
    messages = [{"role": "user", "content": "Weather in London, Tokyo, and Sydney?"}]
    context = LLMContext(messages)

    # Run test
    pipeline = Pipeline([llm])
    frames_to_send = [LLMContextFrame(context)]

    await run_test(pipeline, frames_to_send=frames_to_send, expected_down_frames=None)

    # Verify all functions were called
    assert len(function_calls) == 3

    # Verify each location
    locations = [call["location"] for call in function_calls]
    assert "London, UK" in locations
    assert "Tokyo, Japan" in locations
    assert "Sydney, Australia" in locations

    # Verify tool call IDs
    tool_ids = [call["tool_id"] for call in function_calls]
    assert "call_london" in tool_ids
    assert "call_tokyo" in tool_ids
    assert "call_sydney" in tool_ids


@pytest.mark.asyncio
async def test_mock_llm_chunked_arguments():
    """Test that function arguments streamed in chunks are properly assembled."""
    # Create function call with complex arguments that will be chunked
    complex_args = {
        "location": "San Francisco, California, United States",
        "format": "celsius",
        "include_forecast": True,
        "days": 7,
        "extra_details": ["humidity", "wind_speed", "uv_index"],
    }

    chunks = MockLLMService.create_function_call_chunks(
        function_name="get_extended_weather",
        arguments=complex_args,
        tool_call_id="call_extended",
        chunk_arguments=True,  # This will stream arguments in chunks
    )

    # Create mock service
    llm = MockLLMService(mock_chunks=chunks, chunk_delay=0.001)

    # Track received arguments
    received_args = None

    async def mock_extended_weather(params: FunctionCallParams):
        nonlocal received_args
        received_args = params.arguments
        return {"status": "success"}

    llm.register_function("get_extended_weather", mock_extended_weather)

    # Create context
    messages = [{"role": "user", "content": "Get extended weather"}]
    context = LLMContext(messages)

    # Run test
    pipeline = Pipeline([llm])
    frames_to_send = [LLMContextFrame(context)]

    await run_test(pipeline, frames_to_send=frames_to_send, expected_down_frames=None)

    # Verify arguments were properly assembled from chunks
    assert received_args == complex_args
    assert received_args["location"] == "San Francisco, California, United States"
    assert received_args["format"] == "celsius"
    assert received_args["include_forecast"] is True
    assert received_args["days"] == 7
    assert received_args["extra_details"] == ["humidity", "wind_speed", "uv_index"]


@pytest.mark.asyncio
async def test_mock_llm_empty_response():
    """Test handling of empty response (no text, no function calls)."""
    # Create only a finish chunk
    chunks = [
        MockLLMService.create_text_chunks("", chunk_size=10)[0]  # Just the finish chunk
    ]

    llm = MockLLMService(mock_chunks=chunks, chunk_delay=0.001)

    messages = [{"role": "user", "content": "Test"}]
    context = LLMContext(messages)

    pipeline = Pipeline([llm])
    frames_to_send = [LLMContextFrame(context)]

    received_frames, _ = await run_test(
        pipeline,
        frames_to_send=frames_to_send,
        expected_down_frames=[LLMFullResponseStartFrame, LLMFullResponseEndFrame],
    )

    # Verify no text frames were generated
    text_frames = [f for f in received_frames if isinstance(f, LLMTextFrame)]
    assert len(text_frames) == 0


