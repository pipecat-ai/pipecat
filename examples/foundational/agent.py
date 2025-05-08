import asyncio
import logging
import os
from datetime import datetime

from agents import (
    Agent,
    FunctionTool,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    trace,
)
from httpx import get


@function_tool
async def get_weather(location: str) -> str:
    """Fetch the weather for today.

    Args:
        location: The location to fetch the weather for.
    """
    return f"{location} is sunny"


system_prompt = """
    you are a helpful assistant for a real estate brokerage AI assistant.
"""


bot = Agent(
    name="Assistant agent",
    instructions=system_prompt,
    # tools=[get_weather],
)


async def main():
    # res = await Runner.run(
    #     starting_agent=bot,
    #     input="What is the weather today?",
    # )
    # print(res)

    result = Runner.run_streamed(
        starting_agent=bot,
        # ---
        # with func tool
        input="Tell a joke about pirates.",
        # ---
        # no func tool
        # input="give me a 2 sentences about life",
    )

    final = []
    async for event in result.stream_events():
        # We'll ignore the raw responses event deltas
        name = getattr(event, "name", None)
        # print(f"Event: {event.type} - name {name}")
        # print(event)
        # continue
        if event.type == "raw_response_event":
            if event.data.type == "response.output_text.delta":
                final += event.data.delta

            print(f"raw resp: {event}")
        # When the agent updates, print that
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        # When items are generated, print them
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
            else:
                print(f"-- Unknown item type: {event.item.type}")
                pass  # Ignore other event types
        else:
            print(f"-- Unknown out item type: {event.item.type}")

        print(f"----------------------")

    print(f"FinalFinalFinal: {''.join(final)}")


if __name__ == "__main__":
    asyncio.run(main())


# no func tool:
#
# Event: agent_updated_stream_event - name None
# Event: raw_response_event - name None
# ...
# Event: raw_response_event - name None
# Event: run_item_stream_event - name message_output_created

# with func tool:
#
# Event: agent_updated_stream_event - name None
# Event: raw_response_event - name None
# ...
# Event: raw_response_event - name None
# Event: run_item_stream_event - name tool_called
# Event: run_item_stream_event - name tool_output
# Event: raw_response_event - name None
# ...
# Event: raw_response_event - name None
# Event: run_item_stream_event - name message_output_created
