#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import json
from typing import Any, Dict, List, Mapping, Optional, Union

from loguru import logger

from pipecat.services.llm_service import LLMService

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from mcp_run import Client

try:
    from anthropic import NOT_GIVEN, AsyncAnthropic, NotGiven
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use mcp.run, you need to `pip install pipecat-ai[mcp_run]`. "
        + "Also, set `MCP_RUN_SESSION_ID` environment variable."
    )
    raise Exception(f"Missing module: {e}")

class MCPRun(Client):
    def __init__(
            self,
            llm: LLMService,
            mcp_run_session_id: Optional[str] = None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self._client = Client()
            self._mcp_run_session_id = mcp_run_session_id or os.getenv("MCP_RUN_SESSION_ID")

    def convert_mcp_schema_to_pipecat(self, tool_name: str, tool_schema: dict[str, any]) -> FunctionSchema:
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

    def register_mcp_tools(self, llm) -> ToolsSchema:
        """Register all available mcp.run tools with the LLM service.
        Args:
            llm: The Pipecat LLM service to register tools with
        Returns:
            A ToolsSchema containing all registered tools
        """

        async def mcp_tool_wrapper(function_name: str, tool_call_id: str, arguments: dict[str, any],
                                 llm: any, context: any, result_callback: any) -> None:
            """Wrapper for mcp.run tool calls to match Pipecat's function call interface.
            """
            logger.debug(f"Executing tool '{function_name}' with call ID: {tool_call_id}")
            logger.debug(f"Tool arguments: {json.dumps(arguments, indent=2)}")

            try:
                # Call the mcp.run tool
                logger.debug(f"Calling mcp.run tool '{function_name}'")
                results = self._client.call_tool(function_name, params=arguments)

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

        logger.debug("Starting registration of mcp.run tools")
        tool_schemas: List[FunctionSchema] = []

        # Get all available tools from mcp.run
        available_tools = self._client.tools
        logger.debug(f"Found {len(available_tools)} available tools")

        for tool_name, tool in available_tools.items():
            logger.debug(f"Processing tool: {tool_name}")
            logger.debug(f"Tool description: {tool.description}")


            try:
                # Convert the schema
                function_schema = self.convert_mcp_schema_to_pipecat(tool_name, {
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
