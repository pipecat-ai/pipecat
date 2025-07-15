#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MCP (Model Context Protocol) client for integrating external tools with LLMs."""

import json
from typing import Any, Dict, List, Tuple

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.utils.base_object import BaseObject

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.session_group import SseServerParameters, StreamableHttpParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use an MCP client, you need to `pip install pipecat-ai[mcp]`.")
    raise Exception(f"Missing module: {e}")


class MCPClient(BaseObject):
    """Client for Model Context Protocol (MCP) servers.

    Enables integration with MCP servers to provide external tools and resources
    to LLMs. Supports both stdio and SSE server connections with automatic tool
    registration and schema conversion.

    Raises:
        TypeError: If server_params is not a supported parameter type.
    """

    def __init__(
        self,
        server_params: Tuple[StdioServerParameters, SseServerParameters, StreamableHttpParameters],
        **kwargs,
    ):
        """Initialize the MCP client with server parameters.

        Args:
            server_params: Server connection parameters (stdio or SSE).
            **kwargs: Additional arguments passed to the parent BaseObject.
        """
        super().__init__(**kwargs)
        self._server_params = server_params
        self._session = ClientSession
        self._needs_alternate_schema = False

        if isinstance(server_params, StdioServerParameters):
            self._client = stdio_client
            self._register_tools = self._stdio_register_tools
        elif isinstance(server_params, SseServerParameters):
            self._client = sse_client
            self._register_tools = self._sse_register_tools
        elif isinstance(server_params, StreamableHttpParameters):
            self._client = streamablehttp_client
            self._register_tools = self._streamable_http_register_tools
        else:
            raise TypeError(
                f"{self} invalid argument type: `server_params` must be either StdioServerParameters, SseServerParameters, or StreamableHttpParameters."
            )

    async def register_tools(self, llm) -> ToolsSchema:
        """Register all available MCP tools with an LLM service.

        Connects to the MCP server, discovers available tools, converts their
        schemas to Pipecat format, and registers them with the LLM service.

        Args:
            llm: The Pipecat LLM service to register tools with.

        Returns:
            A ToolsSchema containing all successfully registered tools.
        """
        # Check once if the LLM needs alternate strict schema
        self._needs_alternate_schema = llm and llm.needs_mcp_alternate_schema()
        tools_schema = await self._register_tools(llm)
        return tools_schema

    def _get_alternate_schema_for_strict_validation(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get an alternate JSON schema to be compatible with LLMs that have strict validation.

        Some LLMs have stricter validation and don't allow certain schema properties
        that are valid in standard JSON Schema.

        Args:
            schema: The JSON schema to get an alternate schema for

        Returns:
            An alternate schema compatible with strict validation
        """
        if not isinstance(schema, dict):
            return schema

        alternate_schema = {}

        for key, value in schema.items():
            # Skip additionalProperties as some LLMs don't like additionalProperties: false
            if key == "additionalProperties":
                continue

            # Recursively get alternate schema for nested objects
            if isinstance(value, dict):
                alternate_schema[key] = self._get_alternate_schema_for_strict_validation(value)
            elif isinstance(value, list):
                alternate_schema[key] = [
                    self._get_alternate_schema_for_strict_validation(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                alternate_schema[key] = value

        return alternate_schema

    def _convert_mcp_schema_to_pipecat(
        self, tool_name: str, tool_schema: Dict[str, Any]
    ) -> FunctionSchema:
        """Convert an mcp tool schema to Pipecat's FunctionSchema format.

        Args:
            tool_name: The name of the tool
            tool_schema: The mcp tool schema
        Returns:
            A FunctionSchema instance
        """
        logger.debug(f"Converting schema for tool '{tool_name}'")
        logger.trace(f"Original schema: {json.dumps(tool_schema, indent=2)}")

        properties = tool_schema["input_schema"].get("properties", {})
        required = tool_schema["input_schema"].get("required", [])

        # Only get alternate schema for LLMs that need strict schema validation
        if self._needs_alternate_schema:
            logger.debug("Getting alternate schema for strict validation")
            properties = self._get_alternate_schema_for_strict_validation(properties)

        schema = FunctionSchema(
            name=tool_name,
            description=tool_schema["description"],
            properties=properties,
            required=required,
        )

        logger.trace(f"Converted schema: {json.dumps(schema.to_default_dict(), indent=2)}")

        return schema

    async def _sse_register_tools(self, llm) -> ToolsSchema:
        """Register all available mcp tools with the LLM service.

        Args:
            llm: The Pipecat LLM service to register tools with
        Returns:
            A ToolsSchema containing all registered tools
        """

        async def mcp_tool_wrapper(params: FunctionCallParams) -> None:
            """Wrapper for mcp tool calls to match Pipecat's function call interface."""
            logger.debug(
                f"Executing tool '{params.function_name}' with call ID: {params.tool_call_id}"
            )
            logger.trace(f"Tool arguments: {json.dumps(params.arguments, indent=2)}")
            try:
                async with self._client(**self._server_params.model_dump()) as (read, write):
                    async with self._session(read, write) as session:
                        await session.initialize()
                        await self._call_tool(
                            session, params.function_name, params.arguments, params.result_callback
                        )
            except Exception as e:
                error_msg = f"Error calling mcp tool {params.function_name}: {str(e)}"
                logger.error(error_msg)
                logger.exception("Full exception details:")
                await params.result_callback(error_msg)

        logger.debug(f"SSE server parameters: {self._server_params}")
        logger.debug("Starting registration of mcp tools")

        async with self._client(**self._server_params.model_dump()) as (read, write):
            async with self._session(read, write) as session:
                await session.initialize()
                tools_schema = await self._list_tools(session, mcp_tool_wrapper, llm)
                return tools_schema

    async def _stdio_register_tools(self, llm) -> ToolsSchema:
        """Register all available mcp tools with the LLM service.

        Args:
            llm: The Pipecat LLM service to register tools with
        Returns:
            A ToolsSchema containing all registered tools
        """

        async def mcp_tool_wrapper(params: FunctionCallParams) -> None:
            """Wrapper for mcp tool calls to match Pipecat's function call interface."""
            logger.debug(
                f"Executing tool '{params.function_name}' with call ID: {params.tool_call_id}"
            )
            logger.trace(f"Tool arguments: {json.dumps(params.arguments, indent=2)}")
            try:
                async with self._client(self._server_params) as streams:
                    async with self._session(streams[0], streams[1]) as session:
                        await session.initialize()
                        await self._call_tool(
                            session, params.function_name, params.arguments, params.result_callback
                        )
            except Exception as e:
                error_msg = f"Error calling mcp tool {params.function_name}: {str(e)}"
                logger.error(error_msg)
                logger.exception("Full exception details:")
                await params.result_callback(error_msg)

        logger.debug("Starting registration of mcp tools")

        async with self._client(self._server_params) as streams:
            async with self._session(streams[0], streams[1]) as session:
                await session.initialize()
                tools_schema = await self._list_tools(session, mcp_tool_wrapper, llm)
                return tools_schema

    async def _streamable_http_register_tools(self, llm) -> ToolsSchema:
        """Register all available mcp tools with the LLM service using streamable HTTP.

        Args:
            llm: The Pipecat LLM service to register tools with
        Returns:
            A ToolsSchema containing all registered tools
        """

        async def mcp_tool_wrapper(params: FunctionCallParams) -> None:
            """Wrapper for mcp tool calls to match Pipecat's function call interface."""
            logger.debug(
                f"Executing tool '{params.function_name}' with call ID: {params.tool_call_id}"
            )
            logger.trace(f"Tool arguments: {json.dumps(params.arguments, indent=2)}")
            try:
                async with self._client(**self._server_params.model_dump()) as (
                    read_stream,
                    write_stream,
                    _,
                ):
                    async with self._session(read_stream, write_stream) as session:
                        await session.initialize()
                        await self._call_tool(
                            session, params.function_name, params.arguments, params.result_callback
                        )
            except Exception as e:
                error_msg = f"Error calling mcp tool {params.function_name}: {str(e)}"
                logger.error(error_msg)
                logger.exception("Full exception details:")
                await params.result_callback(error_msg)

        logger.debug("Starting registration of mcp tools using streamable HTTP")

        async with self._client(**self._server_params.model_dump()) as (
            read_stream,
            write_stream,
            _,
        ):
            async with self._session(read_stream, write_stream) as session:
                await session.initialize()
                tools_schema = await self._list_tools(session, mcp_tool_wrapper, llm)
                return tools_schema

    async def _call_tool(self, session, function_name, arguments, result_callback):
        logger.debug(f"Calling mcp tool '{function_name}'")
        try:
            results = await session.call_tool(function_name, arguments=arguments)
        except Exception as e:
            error_msg = f"Error calling mcp tool {function_name}: {str(e)}"
            logger.error(error_msg)

        response = ""
        if results:
            if hasattr(results, "content") and results.content:
                for i, content in enumerate(results.content):
                    if hasattr(content, "text") and content.text:
                        logger.debug(f"Tool response chunk {i}: {content.text}")
                        response += content.text
                    else:
                        # logger.debug(f"Non-text result content: '{content}'")
                        pass
                logger.info(f"Tool '{function_name}' completed successfully")
                logger.debug(f"Final response: {response}")
            else:
                logger.error(f"Error getting content from {function_name} results.")

        final_response = response if len(response) else "Sorry, could not call the mcp tool"
        await result_callback(final_response)

    async def _list_tools(self, session, mcp_tool_wrapper, llm):
        available_tools = await session.list_tools()
        tool_schemas: List[FunctionSchema] = []

        try:
            logger.debug(f"Found {len(available_tools)} available tools")
        except:
            pass

        for tool in available_tools.tools:
            tool_name = tool.name
            logger.debug(f"Processing tool: {tool_name}")
            logger.debug(f"Tool description: {tool.description}")

            try:
                # Convert the schema
                function_schema = self._convert_mcp_schema_to_pipecat(
                    tool_name,
                    {"description": tool.description, "input_schema": tool.inputSchema},
                )

                # Register the wrapped function
                logger.debug(f"Registering function handler for '{tool_name}'")
                llm.register_function(tool_name, mcp_tool_wrapper)

                # Add to list of schemas
                tool_schemas.append(function_schema)
                logger.debug(f"Successfully registered tool '{tool_name}'")

            except Exception as e:
                logger.error(f"Failed to register tool '{tool_name}': {str(e)}")
                logger.exception("Full exception details:")
                continue

        logger.debug(f"Completed registration of {len(tool_schemas)} tools")
        tools_schema = ToolsSchema(standard_tools=tool_schemas)

        return tools_schema
