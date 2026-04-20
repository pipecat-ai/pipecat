#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MCP (Model Context Protocol) client for integrating external tools with LLMs."""

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeAlias

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


@dataclass
class MCPResource:
    """A resource fetched from an MCP server.

    Represents a readable resource exposed by an MCP server, including
    its metadata and content. Text resources contain their content in
    the ``text`` field; binary resources are base64-encoded in ``blob``.

    Parameters:
        uri: The URI identifying this resource.
        name: Human-readable name for this resource.
        description: Optional description of what this resource represents.
        mime_type: The MIME type of this resource, if known.
        text: The text content, populated for text resources.
        blob: Base64-encoded binary content, populated for binary resources.
    """

    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    text: Optional[str] = None
    blob: Optional[str] = None


@dataclass
class MCPResourceList:
    """A collection of MCP resources with server capability metadata.

    Returned by ``MCPClient.list_resources()`` to provide both the
    resource list and information about what the server supports.

    Parameters:
        resources: The list of resource metadata (without content).
        supports_subscribe: Whether the server supports resource subscriptions.
        supports_list_changed: Whether the server emits list-changed notifications.
    """

    resources: List[MCPResource] = field(default_factory=list)
    supports_subscribe: bool = False
    supports_list_changed: bool = False


class MCPClient(BaseObject):
    """Client for Model Context Protocol (MCP) servers.

    Enables integration with MCP servers to provide external tools and resources
    to LLMs. Supports stdio, SSE, and streamable HTTP server connections with
    automatic tool registration and schema conversion.

    The client maintains a persistent connection to the MCP server. It must
    be used as an async context manager or explicitly started and closed::

        async with MCPClient(server_params=...) as mcp:
            tools = await mcp.register_tools(llm)

    Raises:
        TypeError: If server_params is not a supported parameter type.
    """

    def __init__(
        self,
        server_params: ServerParameters,
        tools_filter: Optional[List[str]] = None,
        tools_output_filters: Optional[Dict[str, Callable[[Any], Any]]] = None,
        **kwargs,
    ):
        """Initialize the MCP client with server parameters.

        Args:
            server_params: Server connection parameters (stdio, SSE, or streamable HTTP).
            tools_filter: Optional list of tool names to register. If None, all tools are registered.
            tools_output_filters: Optional dict mapping tool names to filter functions that process tool outputs.
                                  Each filter function receives the raw tool output (any type) and returns the processed output (any type).
            **kwargs: Additional arguments passed to the parent BaseObject.
        """
        super().__init__(**kwargs)
        self._server_params = server_params
        self._tools_filter = tools_filter
        self._tools_output_filters = tools_output_filters or {}
        self._exit_stack: Optional[AsyncExitStack] = None
        self._active_session: Optional[ClientSession] = None

        if not isinstance(
            server_params,
            (StdioServerParameters, SseServerParameters, StreamableHttpParameters),
        ):
            raise TypeError(
                f"{self} invalid argument type: `server_params` must be either "
                "StdioServerParameters, SseServerParameters, or StreamableHttpParameters."
            )

    async def start(self) -> None:
        """Start a persistent connection to the MCP server.

        Opens the transport and initializes the MCP session. The session
        is reused for all subsequent tool calls and schema requests until
        close() is called.

        Can also be used via async context manager::

            async with MCPClient(server_params=...) as mcp:
                ...
        """
        if self._active_session:
            return

        # We manage the exit stack manually (not via `async with`) so we can
        # clean up partial resources on failure before assigning to self.
        exit_stack = AsyncExitStack()
        await exit_stack.__aenter__()

        try:
            if isinstance(self._server_params, StdioServerParameters):
                streams = await exit_stack.enter_async_context(stdio_client(self._server_params))
                read_stream, write_stream = streams[0], streams[1]
            elif isinstance(self._server_params, SseServerParameters):
                read_stream, write_stream = await exit_stack.enter_async_context(
                    sse_client(**self._server_params.model_dump())
                )
            else:  # StreamableHttpParameters (validated in __init__)
                read_stream, write_stream, _ = await exit_stack.enter_async_context(
                    streamablehttp_client(**self._server_params.model_dump())
                )

            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()

            self._exit_stack = exit_stack
            self._active_session = session

        except Exception:
            await exit_stack.aclose()
            raise

    async def close(self) -> None:
        """Close the persistent MCP connection.

        Safe to call multiple times or without having called start().
        """
        self._active_session = None
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def register_tools(self, llm: LLMService | LLMSwitcher) -> ToolsSchema:
        """Register all available MCP tools with an LLM service.

        Discovers available tools from the active session, converts their
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

    def _ensure_connected(self) -> ClientSession:
        """Return the active session or raise if not connected."""
        if not self._active_session:
            raise RuntimeError(
                "MCPClient is not connected. Use 'async with MCPClient(...) as mcp:' "
                "or call 'await mcp.start()' before using MCPClient."
            )
        return self._active_session

    async def get_tools_schema(self) -> ToolsSchema:
        """Get the schema of all available MCP tools without registering them.

        Requires the client to be started via start() or async with.

        Returns:
            A ToolsSchema containing all available tools. This can be used for
            subsequent registration using register_tools_schema().
        """
        session = self._ensure_connected()
        return await self._list_tools_helper(session)

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

    async def list_resources(self) -> MCPResourceList:
        """List available resources from the MCP server.

        Returns resource metadata (URI, name, description, MIME type)
        without fetching the actual content. Use ``read_resource()`` or
        ``read_all_resources()`` to retrieve content.

        Requires the client to be started via ``start()`` or ``async with``.

        Returns:
            An MCPResourceList containing resource metadata and server
            capability flags.

        Raises:
            RuntimeError: If the client is not connected.
        """
        session = self._ensure_connected()

        caps = session.get_server_capabilities()
        supports_subscribe = bool(caps and caps.resources and caps.resources.subscribe)
        supports_list_changed = bool(caps and caps.resources and caps.resources.listChanged)

        resources: List[MCPResource] = []
        cursor: Optional[str] = None
        while True:
            result = await session.list_resources(cursor=cursor)
            for r in result.resources:
                resources.append(
                    MCPResource(
                        uri=str(r.uri),
                        name=r.name or str(r.uri),
                        description=r.description,
                        mime_type=r.mimeType,
                    )
                )
            cursor = result.nextCursor
            if not cursor:
                break

        logger.debug(f"Found {len(resources)} available resources")

        return MCPResourceList(
            resources=resources,
            supports_subscribe=supports_subscribe,
            supports_list_changed=supports_list_changed,
        )

    async def read_resource(self, uri: str) -> MCPResource:
        """Read the content of a specific resource by URI.

        Fetches the resource content from the MCP server. Text resources
        will have their content in the ``text`` field; binary resources
        in the ``blob`` field.

        Args:
            uri: The URI of the resource to read.

        Returns:
            An MCPResource with content populated.

        Raises:
            RuntimeError: If the client is not connected.
        """
        session = self._ensure_connected()

        result = await session.read_resource(uri)

        text = None
        blob = None
        mime_type = None
        for content in result.contents:
            mime_type = content.mimeType
            if hasattr(content, "text"):
                text = (text or "") + content.text
            elif hasattr(content, "blob"):
                blob = content.blob

        logger.debug(f"Read resource '{uri}' (mime_type={mime_type})")

        return MCPResource(
            uri=str(uri),
            name=str(uri),
            description=None,
            mime_type=mime_type,
            text=text,
            blob=blob,
        )

    async def read_all_resources(self, max_concurrency: int = 10) -> List[MCPResource]:
        """List and read all resources from the MCP server.

        Convenience method that combines ``list_resources()`` and
        ``read_resource()`` to fetch all resources with their content
        in a single call. Reads are issued concurrently, bounded by
        ``max_concurrency`` to avoid overwhelming the server.

        Args:
            max_concurrency: Maximum number of concurrent read requests.

        Returns:
            A list of MCPResource objects with content populated, in the
            same order as returned by ``list_resources()``. Individual
            read failures are logged and skipped.

        Raises:
            RuntimeError: If the client is not connected.
        """
        resource_list = await self.list_resources()
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _read_one(r: MCPResource) -> Optional[MCPResource]:
            async with semaphore:
                try:
                    full = await self.read_resource(r.uri)
                except Exception as e:
                    logger.warning(f"Failed to read resource '{r.uri}': {e}")
                    return None
            # Preserve the metadata from list_resources
            full.name = r.name
            full.description = r.description
            full.mime_type = full.mime_type or r.mime_type
            return full

        results = await asyncio.gather(*(_read_one(r) for r in resource_list.resources))
        resources = [r for r in results if r is not None]

        logger.debug(f"Read {len(resources)} of {len(resource_list.resources)} resources")

        return resources

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

    async def _tool_wrapper(self, params: FunctionCallParams) -> None:
        """Execute an MCP tool call using the persistent session."""
        session = self._ensure_connected()
        logger.debug(f"Executing tool '{params.function_name}' with call ID: {params.tool_call_id}")
        logger.trace(f"Tool arguments: {json.dumps(params.arguments, indent=2)}")
        await self._call_tool(
            session,
            params.function_name,
            params.arguments,
            params.result_callback,
        )

    async def _call_tool(self, session, function_name, arguments, result_callback):
        logger.debug(f"Calling mcp tool '{function_name}'")
        results = None
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
            else:
                logger.error(f"Error getting content from {function_name} results.")

        # Apply output filter if configured for this tool
        if function_name in self._tools_output_filters:
            try:
                response = self._tools_output_filters[function_name](response)
                logger.debug(f"Final response (after filter): {response}")

            except Exception:
                logger.error(f"Error applying output filter for {function_name}")
                response = ""

        if response and len(response) and isinstance(response, str):
            logger.info(f"Tool '{function_name}' completed successfully")
            logger.debug(f"Final response: {response}")
        else:
            response = "Sorry, could not call the mcp tool"

        await result_callback(response)

    async def _list_tools_helper(self, session):
        available_tools = await session.list_tools()
        tool_schemas: List[FunctionSchema] = []

        logger.debug(f"Found {len(available_tools.tools)} available tools")

        for tool in available_tools.tools:
            tool_name = tool.name

            # Apply tools filter if configured
            if self._tools_filter and tool_name not in self._tools_filter:
                logger.debug(f"Skipping tool '{tool_name}' - not in allowed tools list")
                continue

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
