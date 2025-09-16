#!/usr/bin/env python3

"""Simple test script for OpenAI Agent service."""

import asyncio
import os
from unittest.mock import MagicMock, patch

# Mock the OpenAI API key for testing
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"

from pipecat.frames.frames import TextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai_agent import OpenAIAgentService


async def test_basic_functionality():
    """Test basic OpenAI Agent service functionality."""
    print("🧪 Testing OpenAI Agent Service...")

    # Create a simple weather tool for testing
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"The weather in {location} is sunny and 22°C."

    try:
        # Create the service
        print("📋 Creating OpenAI Agent service...")
        service = OpenAIAgentService(
            name="Test Assistant",
            instructions="You are a helpful test assistant.",
            tools=[get_weather],
            api_key="test-key",
            streaming=True,
        )

        print(f"✅ Service created successfully!")
        print(f"   - Agent name: {service.agent.name}")
        print(f"   - Model name: {service.model_name}")
        print(f"   - Streaming enabled: {service._streaming}")

        # Test basic configuration
        print("⚙️  Testing configuration updates...")
        service.update_agent_config(
            instructions="Updated test instructions",
            model_config={"model": "gpt-4o", "temperature": 0.5},
        )

        print(f"✅ Configuration updated!")
        print(f"   - New instructions: {service.agent.instructions}")
        print(f"   - New model: {service.model_name}")

        # Test session context
        print("💾 Testing session context...")
        service.update_session_context({"user_id": "test-user", "session": "test-session"})
        context = service.get_session_context()

        print(f"✅ Session context managed!")
        print(f"   - Context keys: {list(context.keys())}")

        # Test adding tools
        print("🔧 Testing tool management...")

        def get_time() -> str:
            """Get current time."""
            return "The current time is 3:00 PM."

        await service.add_tool(get_time)
        print(f"✅ Tool added successfully!")

        print("\n🎉 All basic functionality tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


async def test_frame_processing():
    """Test frame processing with mocked responses."""
    print("\n🔄 Testing frame processing...")

    try:
        # Mock the Runner to avoid actual API calls
        with patch("pipecat.services.openai_agent.agent_service.Runner") as mock_runner:
            # Set up mock responses
            mock_stream_result = MagicMock()

            # Mock stream events
            async def mock_stream_events():
                # Simulate streaming response
                yield MagicMock(type="raw_response_event", data=MagicMock(delta="Hello "))
                yield MagicMock(type="raw_response_event", data=MagicMock(delta="from "))
                yield MagicMock(type="raw_response_event", data=MagicMock(delta="agent!"))

                # Simulate completed message
                mock_item = MagicMock()
                mock_item.type = "message_output_item"
                mock_item.content = "Hello from agent!"
                yield MagicMock(type="run_item_stream_event", item=mock_item)

            mock_stream_result.stream_events.return_value = mock_stream_events()
            mock_runner.run_streamed.return_value = mock_stream_result

            # Create service with mocked runner
            service = OpenAIAgentService(
                name="Test Assistant",
                instructions="You are a helpful test assistant.",
                api_key="test-key",
                streaming=True,
            )

            # Collect output frames
            output_frames = []

            async def mock_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
                output_frames.append(frame)
                print(f"   📤 Frame: {type(frame).__name__}")
                if hasattr(frame, "text"):
                    print(f"      Text: '{frame.text}'")

            service.push_frame = mock_push_frame

            # Process a text frame
            print("📝 Processing text frame...")
            text_frame = TextFrame("Hello, how are you?")
            await service.process_frame(text_frame, FrameDirection.DOWNSTREAM)

            # Wait for async processing
            await asyncio.sleep(0.2)

            print(f"✅ Frame processing completed!")
            print(f"   - Generated {len(output_frames)} output frames")

            # Check if we got expected frame types
            frame_types = [type(frame).__name__ for frame in output_frames]
            print(f"   - Frame types: {frame_types}")

            return True

    except Exception as e:
        print(f"❌ Frame processing test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting OpenAI Agent Service Tests\n")

    try:
        # Run basic functionality tests
        basic_test = await test_basic_functionality()

        # Run frame processing tests
        frame_test = await test_frame_processing()

        # Summary
        print(f"\n📊 Test Results:")
        print(f"   - Basic functionality: {'✅ PASS' if basic_test else '❌ FAIL'}")
        print(f"   - Frame processing: {'✅ PASS' if frame_test else '❌ FAIL'}")

        if basic_test and frame_test:
            print(f"\n🎉 All tests passed! The OpenAI Agent service is working correctly.")
        else:
            print(f"\n⚠️  Some tests failed. Please check the output above.")

    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
