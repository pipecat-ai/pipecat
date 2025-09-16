#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI Agent service."""

import asyncio
import os
import sys
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection


class MockAgent:
    """Mock Agent for testing."""

    def __init__(self, name="Test Agent", instructions="Test instructions"):
        self.name = name
        self.instructions = instructions
        self.tools = []
        self.handoffs = []


class MockRunResult:
    """Mock RunResult for testing."""

    def __init__(self, final_output="Test response"):
        self.final_output = final_output


class MockStreamEvent:
    """Mock StreamEvent for testing."""

    def __init__(self, event_type, data=None, item=None):
        self.type = event_type
        self.data = data
        self.item = item


class MockMessageItem:
    """Mock message item for testing."""

    def __init__(self, content="Test content"):
        self.type = "message_output_item"
        self.content = content


class MockRunner:
    """Mock Runner for testing."""

    @staticmethod
    async def run(agent, input_text, context=None):
        return MockRunResult("Mocked response")

    @staticmethod
    def run_streamed(agent, input_text, context=None):
        class MockStreamResult:
            async def stream_events(self):
                yield MockStreamEvent("raw_response_event", data=MagicMock(delta="Test "))
                yield MockStreamEvent("raw_response_event", data=MagicMock(delta="response"))
                yield MockStreamEvent(
                    "run_item_stream_event", item=MockMessageItem("Test response")
                )

        return MockStreamResult()


@pytest.fixture
def mock_openai_agents():
    """Mock the OpenAI Agents SDK imports."""
    with patch.dict(
        "sys.modules",
        {
            "agents": MagicMock(),
            "agents.stream_events": MagicMock(),
            "agents.result": MagicMock(),
        },
    ):
        # Mock the classes and functions we need
        mock_agent = MagicMock()
        mock_agent.return_value = MockAgent()

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=MockRunResult())
        mock_runner.run_streamed = MagicMock(return_value=MockRunner.run_streamed(None, None))

        with (
            patch("pipecat.services.openai_agent.agent_service.Agent", mock_agent),
            patch("pipecat.services.openai_agent.agent_service.Runner", mock_runner),
        ):
            yield {
                "Agent": mock_agent,
                "Runner": mock_runner,
            }


@pytest.mark.asyncio
async def test_openai_agent_service_init(mock_openai_agents):
    """Test OpenAI Agent service initialization."""
    from pipecat.services.openai_agent.agent_service import OpenAIAgentService

    service = OpenAIAgentService(
        name="Test Agent", instructions="Test instructions", api_key="test-key", streaming=True
    )

    assert service.agent.name == "Test Agent"
    assert service._streaming is True


@pytest.mark.asyncio
async def test_openai_agent_service_process_text_frame_streaming(mock_openai_agents):
    """Test processing text frame with streaming enabled."""
    from pipecat.services.openai_agent.agent_service import OpenAIAgentService

    service = OpenAIAgentService(
        name="Test Agent", instructions="Test instructions", api_key="test-key", streaming=True
    )

    # Mock the push_frame method to capture output
    output_frames = []

    async def mock_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        output_frames.append(frame)

    service.push_frame = mock_push_frame

    # Process a text frame
    text_frame = TextFrame("Hello, agent!")
    await service.process_frame(text_frame, FrameDirection.DOWNSTREAM)

    # Wait a bit for async processing
    await asyncio.sleep(0.1)

    # Check that appropriate frames were generated
    assert len(output_frames) > 0
    assert any(isinstance(frame, LLMFullResponseStartFrame) for frame in output_frames)


@pytest.mark.asyncio
async def test_openai_agent_service_process_text_frame_non_streaming(mock_openai_agents):
    """Test processing text frame with streaming disabled."""
    from pipecat.services.openai_agent.agent_service import OpenAIAgentService

    service = OpenAIAgentService(
        name="Test Agent", instructions="Test instructions", api_key="test-key", streaming=False
    )

    # Mock the push_frame method to capture output
    output_frames = []

    async def mock_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
        output_frames.append(frame)

    service.push_frame = mock_push_frame

    # Process a text frame
    text_frame = TextFrame("Hello, agent!")
    await service.process_frame(text_frame, FrameDirection.DOWNSTREAM)

    # Wait a bit for async processing
    await asyncio.sleep(0.1)

    # Check that appropriate frames were generated
    assert len(output_frames) > 0


@pytest.mark.asyncio
async def test_openai_agent_service_update_config(mock_openai_agents):
    """Test updating agent configuration."""
    from pipecat.services.openai_agent.agent_service import OpenAIAgentService

    service = OpenAIAgentService(
        name="Test Agent", instructions="Test instructions", api_key="test-key"
    )

    # Update configuration
    service.update_agent_config(
        instructions="Updated instructions", model_config={"model": "gpt-4o", "temperature": 0.7}
    )

    assert service.agent.instructions == "Updated instructions"
    assert service.agent.model_config["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_openai_agent_service_session_context(mock_openai_agents):
    """Test session context management."""
    from pipecat.services.openai_agent.agent_service import OpenAIAgentService

    service = OpenAIAgentService(
        name="Test Agent",
        instructions="Test instructions",
        api_key="test-key",
        session_config={"user_id": "test-user"},
    )

    # Get initial context
    context = service.get_session_context()
    assert context["user_id"] == "test-user"

    # Update context
    service.update_session_context({"session_id": "test-session"})

    updated_context = service.get_session_context()
    assert updated_context["user_id"] == "test-user"
    assert updated_context["session_id"] == "test-session"


@pytest.mark.asyncio
async def test_openai_agent_service_add_tools(mock_openai_agents):
    """Test adding tools to the agent."""
    from pipecat.services.openai_agent.agent_service import OpenAIAgentService

    service = OpenAIAgentService(
        name="Test Agent", instructions="Test instructions", api_key="test-key"
    )

    # Define a test tool
    def test_tool():
        return "test result"

    # Add the tool
    await service.add_tool(test_tool)

    # Check if tool was added (this depends on the mock implementation)
    assert hasattr(service.agent, "tools")


@pytest.mark.asyncio
async def test_openai_agent_service_lifecycle(mock_openai_agents):
    """Test service lifecycle methods."""
    from pipecat.frames.frames import CancelFrame, EndFrame, StartFrame
    from pipecat.services.openai_agent.agent_service import OpenAIAgentService

    service = OpenAIAgentService(
        name="Test Agent", instructions="Test instructions", api_key="test-key"
    )

    # Test start
    start_frame = StartFrame()
    await service.start(start_frame)

    # Test cancel
    cancel_frame = CancelFrame()
    await service.cancel(cancel_frame)

    # Test stop
    end_frame = EndFrame()
    await service.stop(end_frame)


def test_openai_agent_service_import_error():
    """Test that import error is handled gracefully."""
    # Mock the import to fail
    with patch.dict("sys.modules", {"agents": None}):
        with pytest.raises(Exception) as exc_info:
            # This should trigger the import error
            import importlib

            import pipecat.services.openai_agent.agent_service

            importlib.reload(pipecat.services.openai_agent.agent_service)

        assert "Missing module" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
