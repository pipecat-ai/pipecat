#!/usr/bin/env python3

import asyncio
import os

from loguru import logger

# Test the actual agents package API
try:
    from agents import Agent, run

    # Create a simple agent
    agent = Agent(
        name="test-agent",
        instructions="You are a helpful assistant.",
    )

    print("✅ Agent created successfully!")
    print(f"Agent name: {agent.name}")

    # Test a simple conversation
    async def test_agent():
        result = await run(agent, "Hello, how are you?")
        print(f"Agent response: {result}")

    # Run the test
    asyncio.run(test_agent())

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
