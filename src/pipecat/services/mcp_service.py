#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MCP (Model Context Protocol) client for integrating external tools with LLMs."""

import asyncio
import json
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any, TypeAlias

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.utils.base_object import BaseObject
from pipecat.utils.deprecation import deprecated

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.session_group import SseServerParameters, StreamableHttpParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use an MCP client, you need to `uv add "pipecat-ai[mcp]"`.')
    raise ImportError(f"Missing module: {e}") from e

ServerParameters: TypeAlias = StdioServerParameters | SseServerParameters | StreamableHttpParameters


class MCPClient(BaseObject):
    """Client for Model Context Protocol (MCP) servers.

    Enables integration with MCP servers to provide external tools and resources
    to LLMs. Supports stdio, SSE, and streamable HTTP server connections with
    automatic tool registration and schema conversion.

    The client maintains a persistent connection to the MCP server, opened on
    the first call to :meth:`tools` (or ahead of time via :meth:`start`). The
    tool schemas returned by :meth:`tools` carry their call handlers, so the
    LLM service registers them automatically when they are advertised through
    an ``LLMContext``, and closes the connection when the pipeline is torn
    down::

        mcp = MCPClient(server_params=...)
        context = LLMContext(messages=[...], tools=await mcp.tools())

    Call :meth:`close` only to release the connection earlier than pipeline
    teardown (e.g. from an ``on_client_disconnected`` handler). :meth:`start`
    and :meth:`close` may be called from different tasks — the session is owned
    by a dedicated internal task. Scoping the client with ``async with
    MCPClient(...)`` is also supported.

    Raises:
        TypeError: If server_params is not a supported parameter type.
    """

    # Ask the LLM service to close() this client when the service is cleaned up
    # at pipeline teardown, so the connection is released without the developer
    # wiring close() manually.
    _pipecat_close_on_teardown = True

    def __init__(
        self,
        server_params: ServerParameters,
        tools_filter: list[str] | None = None,
        tools_output_filters: dict[str, Callable[[Any], Any]] | None = None,
        tools_arguments: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ):
        """Initialize the MCP client with server parameters.

        Args:
            server_params: Server connection parameters (stdio, SSE, or streamable HTTP).
            tools_filter: Optional list of tool names to register. If None, all tools are registered.
            tools_output_filters: Optional dict mapping tool names to filter functions that process tool outputs.
                                  Each filter function receives the raw tool output (any type) and returns the processed output (any type).
            tools_arguments: Optional dict mapping tool names to fixed arguments that are
                             merged into every call of that tool (overriding any
                             model-supplied values). The fixed parameter names are removed
                             from the advertised tool schema so the LLM never sees them.
            **kwargs: Additional arguments passed to the parent BaseObject.
        """
        super().__init__(**kwargs)
        self._server_params = server_params
        self._tools_filter = tools_filter
        self._tools_output_filters = tools_output_filters or {}
        self._tools_arguments = tools_arguments or {}
        self._exit_stack: AsyncExitStack | None = None
        self._active_session: ClientSession | None = None
        # The MCP session is anyio-task-bound: it must be opened and closed in the
        # same task. start() and close() can be called from different tasks, so a
        # dedicated owner task (_run_session) holds the session open and tears it
        # down on signal.
        self._session_task: asyncio.Task | None = None
        self._closing: asyncio.Event | None = None
        self._start_lock = asyncio.Lock()

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

        Opens the transport and initializes the MCP session. The session is
        reused for all subsequent tool calls and schema requests until close()
        is called. Idempotent, and called automatically by :meth:`tools`.

        Can also be used via async context manager::

            async with MCPClient(server_params=...) as mcp:
                ...
        """
        if self._active_session:
            return

        async with self._start_lock:
            if self._active_session:
                return
            self._closing = asyncio.Event()
            ready: asyncio.Future = asyncio.get_running_loop().create_future()
            self._session_task = asyncio.create_task(
                self._run_session(ready, self._closing), name=f"{self}::session"
            )
            try:
                await ready  # resolves once connected; re-raises a connect failure
            except Exception:
                self._session_task = None  # let a later call retry cleanly
                raise

    async def _run_session(self, ready: asyncio.Future, closing: asyncio.Event) -> None:
        """Own the MCP session for its whole lifetime, in a single task.

        Opens the connection, signals ``ready``, then holds the session open
        until :meth:`close` sets the closing event — so open and close happen
        in the same task, as the anyio-based MCP transports require.
        """
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
        except Exception as e:
            await exit_stack.aclose()
            ready.set_exception(e)
            return

        self._exit_stack = exit_stack
        self._active_session = session
        ready.set_result(None)
        try:
            await closing.wait()
        finally:
            self._active_session = None
            self._exit_stack = None
            await exit_stack.aclose()

    async def close(self) -> None:
        """Close the persistent MCP connection.

        Called automatically at pipeline teardown once the client's tools have
        been registered with an LLM service; call it directly only to release
        the connection earlier (e.g. on client disconnect). Safe to call
        multiple times, without having called start(), and from a different
        task than the one that called start().
        """
        if self._session_task is None:
            return
        task, self._session_task = self._session_task, None
        if self._closing is not None:
            self._closing.set()
        await task

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def tools(self) -> ToolsSchema:
        """Get the available MCP tools, ready for LLM auto-registration.

        Starts the server connection if needed, then returns a ToolsSchema
        whose function schemas carry their call handlers, so the LLM service
        registers them automatically when the context advertises its tools::

            context = LLMContext(messages=[...], tools=await mcp.tools())

        The connection is closed automatically at pipeline teardown; call
        :meth:`close` only to release it earlier.

        Returns:
            A ToolsSchema containing the available tools with handlers attached.
        """
        await self.start()
        session = self._ensure_connected()
        return await self._list_tools_helper(session, attach_handlers=True)

    @deprecated(
        "`MCPClient.register_tools` is deprecated since 1.5.0 and will be removed in 2.0.0. "
        "Use `MCPClient.tools` instead."
    )
    async def register_tools(self, llm: LLMService | LLMSwitcher) -> ToolsSchema:
        """Register all available MCP tools with an LLM service.

        .. deprecated:: 1.5.0
            Use :meth:`tools` instead — pass its result to ``LLMContext(tools=...)``
            and the handlers register automatically. Will be removed in 2.0.0.

        Discovers available tools from the active session, converts their
        schemas to Pipecat format, and registers them with the LLM service.

        Args:
            llm: The Pipecat LLM service to register tools with.

        Returns:
            A ToolsSchema containing all successfully registered tools.
        """
        session = self._ensure_connected()
        tools_schema = await self._list_tools_helper(session)
        for function_schema in tools_schema.standard_tools:
            llm.register_function(function_schema.name, self._tool_wrapper)
        return tools_schema

    def _ensure_connected(self) -> ClientSession:
        """Return the active session or raise if not connected."""
        if not self._active_session:
            raise RuntimeError(
                "MCPClient is not connected. Use 'async with MCPClient(...) as mcp:' "
                "or call 'await mcp.start()' before using MCPClient."
            )
        return self._active_session

    @deprecated(
        "`MCPClient.get_tools_schema` is deprecated since 1.5.0 and will be removed in 2.0.0. "
        "Use `MCPClient.tools` instead."
    )
    async def get_tools_schema(self) -> ToolsSchema:
        """Get the schema of all available MCP tools without registering them.

        .. deprecated:: 1.5.0
            Use :meth:`tools` instead. Will be removed in 2.0.0.

        Requires the client to be started via start() or async with.

        Returns:
            A ToolsSchema containing all available tools.
        """
        session = self._ensure_connected()
        return await self._list_tools_helper(session)

    @deprecated(
        "`MCPClient.register_tools_schema` is deprecated since 1.5.0 and will be removed in "
        "2.0.0. Use `MCPClient.tools` instead."
    )
    async def register_tools_schema(
        self, tools_schema: ToolsSchema, llm: LLMService | LLMSwitcher
    ) -> None:
        """Register previously obtained MCP tools with the LLM service.

        .. deprecated:: 1.5.0
            Use :meth:`tools` instead — its schemas carry handlers that
            register automatically. Will be removed in 2.0.0.

        Args:
            tools_schema: The ToolsSchema to register with the LLM service.
            llm: The Pipecat LLM service to register tools with.
        """
        for function_schema in tools_schema.standard_tools:
            llm.register_function(function_schema.name, self._tool_wrapper)

    def _convert_mcp_schema_to_pipecat(
        self,
        tool_name: str,
        tool_schema: dict[str, Any],
        handler: Callable | None = None,
    ) -> FunctionSchema:
        """Convert an mcp tool schema to Pipecat's FunctionSchema format.

        Args:
            tool_name: The name of the tool
            tool_schema: The mcp tool schema
            handler: Optional call handler to attach for LLM auto-registration.

        Returns:
            A FunctionSchema instance
        """
        logger.debug(f"Converting schema for tool '{tool_name}'")
        logger.trace(f"Original schema: {json.dumps(tool_schema, indent=2)}")

        properties = tool_schema["input_schema"].get("properties", {})
        required = tool_schema["input_schema"].get("required", [])

        fixed = self._tools_arguments.get(tool_name)
        if fixed:
            # Fixed arguments are injected on every call, so hide them from the model.
            properties = {k: v for k, v in properties.items() if k not in fixed}
            required = [r for r in required if r not in fixed]

        schema = FunctionSchema(
            name=tool_name,
            description=tool_schema["description"],
            properties=properties,
            required=required,
            handler=handler,
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

    async def call_tool(self, function_name: str, arguments: dict[str, Any] | None) -> str:
        """Call an MCP tool by name and return its text result.

        Requires the client to be started (via :meth:`start`, :meth:`tools`, or
        ``async with``). Useful for callers that invoke a known tool directly
        rather than through LLM function-call registration. Configured
        ``tools_arguments`` are applied here too.

        Args:
            function_name: The MCP tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            The tool's text output with any configured output filter applied, or
            a fallback message when the call yields no text.
        """
        session = self._ensure_connected()
        return await self._call_tool_text(session, function_name, arguments)

    async def _call_tool(self, session, function_name, arguments, result_callback):
        response = await self._call_tool_text(session, function_name, arguments)
        await result_callback(response)

    async def _call_tool_text(self, session, function_name, arguments) -> str:
        fixed = self._tools_arguments.get(function_name)
        if fixed:
            # Fixed arguments win over model-supplied values.
            arguments = {**(arguments or {}), **fixed}

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

        return response

    async def _list_tools_helper(self, session, attach_handlers: bool = False):
        available_tools = await session.list_tools()
        tool_schemas: list[FunctionSchema] = []

        logger.debug(f"Found {len(available_tools.tools)} available tools")

        available_names = {tool.name for tool in available_tools.tools}
        unknown = [name for name in self._tools_arguments if name not in available_names]
        if unknown:
            logger.warning(
                f"{self} tools_arguments configured for tool(s) the server does not "
                f"advertise: {', '.join(unknown)}"
            )

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
                    handler=self._tool_wrapper if attach_handlers else None,
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
