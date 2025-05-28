import json
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.utils.base_object import BaseObject

try:
    from mcp import ClientSession, StdioServerParameters, types
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use an MCP client, you need to `pip install pipecat-ai[mcp]`.")
    raise Exception(f"Missing module: {e}")


class MCPClient(BaseObject):
    def __init__(
        self,
        server_params: Union[StdioServerParameters, str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._server_params = server_params
        self._session = ClientSession
        if isinstance(server_params, StdioServerParameters):
            self._client = stdio_client
            self._register_tools = self._stdio_register_tools
        elif isinstance(server_params, str):
            self._client = sse_client
            self._register_tools = self._sse_register_tools
        else:
            raise TypeError(
                f"{self} invalid argument type: `server_params` must be either StdioServerParameters or an SSE server url string."
            )

    async def register_tools(self, llm) -> ToolsSchema:
        tools_schema = await self._register_tools(llm)
        return tools_schema

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

    async def _sse_register_tools(self, llm) -> ToolsSchema:
        """Register all available mcp.run tools with the LLM service.
        Args:
            llm: The Pipecat LLM service to register tools with
        Returns:
            A ToolsSchema containing all registered tools
        """

        async def mcp_tool_wrapper(
            function_name: str,
            tool_call_id: str,
            arguments: Dict[str, Any],
            llm: any,
            context: any,
            result_callback: any,
        ) -> None:
            """Wrapper for mcp.run tool calls to match Pipecat's function call interface."""
            logger.debug(f"Executing tool '{function_name}' with call ID: {tool_call_id}")
            logger.trace(f"Tool arguments: {json.dumps(arguments, indent=2)}")
            try:
                async with self._client(self._server_params) as (read, write):
                    async with self._session(read, write) as session:
                        await session.initialize()
                        await self._call_tool(session, function_name, arguments, result_callback)
            except Exception as e:
                error_msg = f"Error calling mcp tool {function_name}: {str(e)}"
                logger.error(error_msg)
                logger.exception("Full exception details:")
                await result_callback(error_msg)

        logger.debug("Starting registration of mcp.run tools")
        tool_schemas: List[FunctionSchema] = []

        async with self._client(self._server_params) as (read, write):
            async with self._session(read, write) as session:
                await session.initialize()
                tools_schema = await self._list_tools(session, mcp_tool_wrapper, llm)
                return tools_schema

    async def _stdio_register_tools(self, llm) -> ToolsSchema:
        """Register all available mcp.run tools with the LLM service.
        Args:
            llm: The Pipecat LLM service to register tools with
        Returns:
            A ToolsSchema containing all registered tools
        """

        async def mcp_tool_wrapper(
            function_name: str,
            tool_call_id: str,
            arguments: Dict[str, Any],
            llm: any,
            context: any,
            result_callback: any,
        ) -> None:
            """Wrapper for mcp.run tool calls to match Pipecat's function call interface."""
            logger.debug(f"Executing tool '{function_name}' with call ID: {tool_call_id}")
            logger.trace(f"Tool arguments: {json.dumps(arguments, indent=2)}")
            try:
                async with self._client(self._server_params) as streams:
                    async with self._session(streams[0], streams[1]) as session:
                        await session.initialize()
                        await self._call_tool(session, function_name, arguments, result_callback)
            except Exception as e:
                error_msg = f"Error calling mcp tool {function_name}: {str(e)}"
                logger.error(error_msg)
                logger.exception("Full exception details:")
                await result_callback(error_msg)

        logger.debug("Starting registration of mcp.run tools")

        async with self._client(self._server_params) as streams:
            async with self._session(streams[0], streams[1]) as session:
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
