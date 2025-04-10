#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import json

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport


from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService


from mcp_run import Client

load_dotenv(override=True)

logger.remove()
logger.add(sys.stderr, level="DEBUG")

def convert_mcp_schema_to_pipecat(tool_name: str, tool_schema: dict[str, any]) -> FunctionSchema:
    """Convert an mcp.run tool schema to Pipecat's FunctionSchema format.
    
    Args:
        tool_name: The name of the tool
        tool_schema: The mcp.run tool schema
        
    Returns:
        A FunctionSchema instance
    """
    logger.debug(f"Converting schema for tool '{tool_name}'")
    logger.debug(f"Original schema: {json.dumps(tool_schema, indent=2)}")
    
    # Extract properties and required fields from the mcp.run schema
    properties = tool_schema["input_schema"].get("properties", {})
    required = tool_schema["input_schema"].get("required", [])
    
    schema = FunctionSchema(
        name=tool_name,
        description=tool_schema["description"],
        properties=properties,
        required=required
    )
    
    logger.debug(f"Converted schema: {json.dumps(schema.to_default_dict(), indent=2)}")

    return schema


async def mcp_tool_wrapper(function_name: str, tool_call_id: str, arguments: dict[str, any], 
                         llm: any, context: any, result_callback: any) -> None:
    """Wrapper function for mcp.run tool calls that matches Pipecat's function call interface.
    
    Args:
        function_name: Name of the tool to call
        tool_call_id: Unique ID for this tool call
        arguments: Tool parameters
        llm: LLM service instance
        context: Context object
        result_callback: Callback function to return results
    """
    logger.debug(f"Executing tool '{function_name}' with call ID: {tool_call_id}")
    logger.debug(f"Tool arguments: {json.dumps(arguments, indent=2)}")
    
    try:
        # Call the mcp.run tool
        logger.debug(f"Calling mcp.run tool '{function_name}'")
        results = llm.mcp_client.call_tool(function_name, params=arguments)
        
        # Combine all content into a single response
        response = ""
        for i, content in enumerate(results.content):
            logger.debug(f"Tool response chunk {i}: {content.text}")
            response += content.text
            
        logger.info(f"Tool '{function_name}' completed successfully")
        logger.info(f"Final response: {response}")
            
        # Send result back through callback
        await result_callback(response)
        
    except Exception as e:
        error_msg = f"Error calling mcp.run tool {function_name}: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full exception details:")
        await result_callback(error_msg)

def register_mcp_tools(llm) -> ToolsSchema:
    """Register all available mcp.run tools with the LLM service.
    
    Args:
        llm: The Pipecat LLM service to register tools with
        
    Returns:
        A ToolsSchema containing all registered tools
    """
    logger.debug("Starting registration of mcp.run tools")
    tool_schemas: List[FunctionSchema] = []
    
    # Get all available tools from mcp.run
    available_tools = llm.mcp_client.tools
    logger.debug(f"Found {len(available_tools)} available tools")
    
    for tool_name, tool in available_tools.items():
        logger.debug(f"Processing tool: {tool_name}")
        logger.debug(f"Tool description: {tool.description}")
        
        try:
            # Convert the schema
            function_schema = convert_mcp_schema_to_pipecat(tool_name, {
                "description": tool.description,
                "input_schema": tool.input_schema
            })
            
            # Register the wrapped function
            logger.debug(f"Registering function handler for '{tool_name}'")
            llm.register_function(tool_name, mcp_tool_wrapper)
            
            # Add to our list of schemas
            tool_schemas.append(function_schema)
            logger.debug(f"Successfully registered tool '{tool_name}'")

        except Exception as e:
            logger.error(f"Failed to register tool '{tool_name}': {str(e)}")
            logger.exception("Full exception details:")
            continue
    
    logger.info(f"Completed registration of {len(tool_schemas)} tools")
    tools_schema = ToolsSchema(standard_tools=tool_schemas)
    
    return tools_schema

async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "bot with fetch mcp",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        llm = AnthropicLLMService(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-7-sonnet-latest")
        # llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-001")
        # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # Initialize the mcp.run client
        logger.info("Initializing mcp.run client")
        mcp_client = Client()

        llm.mcp_client = mcp_client
        tools = register_mcp_tools(llm)

        system = """
                Always start with 'Aloha'. You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities 
                in a succinct way. You have access to various tools provided by mcp.run that you can use to help users.
                Your output will be converted to audio so don't include special characters in your answers. 
                Respond to what the user said in a creative and helpful way. Don't overexplain what you are doing as it takes up too much time.
                Just respond with short sentences when you are carrying out tool calls.

                Note: When the user asks you to to put something in Notion, use the parent page "Integration Page"
                which has the id: "15e9debf-7128-80f2-88d6-c5cabcba3ef8"
                """

        messages = [{"role": "system",
                     "content": system
                     }
                    ]                  

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                context_aggregator.user(),  # User spoken responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses and tool context
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info("First participant joined: {}", participant["id"])
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
