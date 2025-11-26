#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MCP (Model Context Protocol) client for integrating external tools with LLMs."""

import json
from typing import Any, Dict, List, TypeAlias

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.services.llm_service import FunctionCallParams, LLMService
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

ServerParameters: TypeAlias = StdioServerParameters | SseServerParameters | StreamableHttpParameters


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
        server_params: ServerParameters,
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

        if isinstance(server_params, StdioServerParameters):
            self._client = stdio_client
            self._list_tools = self._stdio_list_tools
            self._tool_wrapper = self._stdio_tool_wrapper
        elif isinstance(server_params, SseServerParameters):
            self._client = sse_client
            self._list_tools = self._sse_list_tools
            self._tool_wrapper = self._sse_tool_wrapper
        elif isinstance(server_params, StreamableHttpParameters):
            self._client = streamablehttp_client
            self._list_tools = self._streamable_http_list_tools
            self._tool_wrapper = self._streamable_http_tool_wrapper
        else:
            raise TypeError(
                f"{self} invalid argument type: `server_params` must be either StdioServerParameters, SseServerParameters, or StreamableHttpParameters."
            )

    async def register_tools(self, llm: LLMService | LLMSwitcher) -> ToolsSchema:
        """Register all available MCP tools with an LLM service.

        Connects to the MCP server, discovers available tools, converts their
        schemas to Pipecat format, and registers them with the LLM service.

        This is the equivalent of calling get_tools_schema() followed by
        register_tools_schema().

        Args:
            llm: The Pipecat LLM service to register tools with.

        Returns:
            A ToolsSchema containing all successfully registered tools.
        """
        tools_schema = await self.get_tools_schema()
        await self.register_tools_schema(tools_schema, llm)
        return tools_schema

    async def get_tools_schema(self) -> ToolsSchema:
        """Get the schema of all available MCP tools without registering them.

        Connects to the MCP server, discovers available tools, and converts their
        schemas to Pipecat format.

        Returns:
            A ToolsSchema containing all available tools. This can be used for
            subsequent registration using register_tools_schema().
        """
        tools_schema = await self._list_tools()
        return tools_schema

    async def register_tools_schema(
        self, tools_schema: ToolsSchema, llm: LLMService | LLMSwitcher
    ) -> None:
        """Register the MCP tools (previously obtained from get_tools_schema()) with the LLM service.

        Args:
            tools_schema: The ToolsSchema to register with the LLM service.
            llm: The Pipecat LLM service to register tools with.
        """
        for function_schema in tools_schema.standard_tools:
            llm.register_function(function_schema.name, self._tool_wrapper)

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

        schema = FunctionSchema(
            name=tool_name,
            description=tool_schema["description"],
            properties=properties,
            required=required,
        )

        logger.trace(f"Converted schema: {json.dumps(schema.to_default_dict(), indent=2)}")

        return schema

    async def _sse_list_tools(self) -> ToolsSchema:
        """List all available mcp tools with the LLM service.

        Returns:
            A ToolsSchema containing all registered tools
        """
        logger.debug(f"SSE server parameters: {self._server_params}")
        logger.debug(f"Starting reading mcp tools")

        async with self._client(**self._server_params.model_dump()) as (read, write):
            async with self._session(read, write) as session:
                await session.initialize()
                tools_schema = await self._list_tools_helper(session)
                return tools_schema

    async def _sse_tool_wrapper(self, params: FunctionCallParams) -> None:
        """Wrapper for mcp tool calls to match Pipecat's function call interface."""
        logger.debug(f"Executing tool '{params.function_name}' with call ID: {params.tool_call_id}")
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
            await params.result_callback(error_msg)

    async def _stdio_list_tools(self) -> ToolsSchema:
        """List all available mcp tools with the LLM service.

        Returns:
            A ToolsSchema containing all available tools.
        """
        logger.debug(f"Starting reading mcp tools")

        async with self._client(self._server_params) as streams:
            async with self._session(streams[0], streams[1]) as session:
                await session.initialize()
                tools_schema = await self._list_tools_helper(session)
                return tools_schema

    async def _stdio_tool_wrapper(self, params: FunctionCallParams) -> None:
        """Wrapper for mcp tool calls to match Pipecat's function call interface."""
        logger.debug(f"Executing tool '{params.function_name}' with call ID: {params.tool_call_id}")
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
            await params.result_callback(error_msg)

    async def _streamable_http_list_tools(self) -> ToolsSchema:
        """List all available mcp tools with the LLM service using streamable HTTP.

        Returns:
            A ToolsSchema containing all available tools.
        """
        logger.debug(f"Starting reading mcp tools using streamable HTTP")

        async with self._client(**self._server_params.model_dump()) as (
            read_stream,
            write_stream,
            _,
        ):
            async with self._session(read_stream, write_stream) as session:
                await session.initialize()
                tools_schema = await self._list_tools_helper(session)
                return tools_schema

    async def _streamable_http_tool_wrapper(self, params: FunctionCallParams) -> None:
        """Wrapper for mcp tool calls to match Pipecat's function call interface."""
        logger.debug(f"Executing tool '{params.function_name}' with call ID: {params.tool_call_id}")
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
            await params.result_callback(error_msg)

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

    async def _list_tools_helper(self, session):
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

                # Add to list of schemas
                tool_schemas.append(function_schema)
                logger.debug(f"Successfully read tool '{tool_name}'")

            except Exception as e:
                logger.error(f"Failed to read tool '{tool_name}': {str(e)}")
                continue

        logger.debug(f"Completed reading {len(tool_schemas)} tools")
        tools_schema = ToolsSchema(standard_tools=tool_schemas)

        return tools_schema
