#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MCP (Model Context Protocol) client for integrating external tools with LLMs."""

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from typing import Any, TypeAlias, cast

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.utils.base_object import BaseObject
from pipecat.utils.deprecation import deprecated

try:
    import anyio
    from mcp import StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.client.session_group import SseServerParameters, StreamableHttpParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared import exceptions as mcp_exceptions
    from mcp.types import CONNECTION_CLOSED, TextContent
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use an MCP client, you need to `uv add "pipecat-ai[mcp]"`.')
    raise ImportError(f"Missing module: {e}") from e

# The SDK names the class McpError on 1.x and MCPError on 2.x. The string cast
# keeps the 1.x name out of a 2.x import.
_MCPError = cast(
    "type[mcp_exceptions.McpError]",
    getattr(mcp_exceptions, "MCPError", None) or mcp_exceptions.McpError,
)

ServerParameters: TypeAlias = StdioServerParameters | SseServerParameters | StreamableHttpParameters

_MAX_ERROR_DETAIL = 200


class _NotConnectedError(RuntimeError):
    """The client holds no session, and no owner has asked it for one."""


def _is_connection_lost(error: BaseException) -> bool:
    """Tell a dead transport from an error the MCP server itself reported.

    A transport that fails inside its own task group reports through a group,
    which carries the transport's own error among whatever else failed with it.
    """
    if isinstance(error, BaseExceptionGroup):
        return any(_is_connection_lost(sub) for sub in error.exceptions)
    if isinstance(error, _MCPError):
        return error.error.code == CONNECTION_CLOSED
    return isinstance(
        error, (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream)
    )


def _sole_cause(error: BaseException) -> BaseException:
    """Read the one exception a group of one wraps, at whatever depth.

    A group carries neither the type nor the message of what went wrong, so a
    group of one reports as the cause it holds.
    """
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


def _error_detail(error: BaseException) -> str:
    """Name the cause of a failed tool call, in the one line the model reads."""
    cause = _sole_cause(error)
    detail = str(cause)
    if not detail:
        # An anyio stream error carries no message, so name its class instead.
        return type(cause).__name__
    if len(detail) > _MAX_ERROR_DETAIL:
        return detail[:_MAX_ERROR_DETAIL] + "..."
    return detail


@asynccontextmanager
async def _streamable_http_transport(params: StreamableHttpParameters):
    """Open a streamable-HTTP transport, owning the HTTP client it runs on.

    The transport takes its HTTP settings as a prepared client and leaves that
    client's lifetime to its caller.

    Args:
        params: Connection parameters for the MCP server.

    Yields:
        The transport's streams, read stream first.
    """
    # The client class has to come from the httpx family the SDK itself is built
    # on, which follows the SDK's major version, so take it from the transport's
    # own module rather than importing a family directly.
    transport_module = sys.modules[streamable_http_client.__module__]
    http = getattr(transport_module, "httpx2", None) or transport_module.httpx
    async with http.AsyncClient(
        headers=params.headers,
        timeout=http.Timeout(
            _timeout_seconds(params.timeout), read=_timeout_seconds(params.sse_read_timeout)
        ),
        # Matches the client the SDK builds when given none.
        follow_redirects=True,
    ) as client:
        async with streamable_http_client(
            params.url, http_client=client, terminate_on_close=params.terminate_on_close
        ) as streams:
            yield streams


def _timeout_seconds(value: float | timedelta) -> float:
    """Read a timeout as seconds, whichever way the SDK models it."""
    return value.total_seconds() if isinstance(value, timedelta) else value


def _connect_failure_cause(*candidates: BaseException | None) -> Exception | None:
    """Find the exception that best explains a failed connect.

    A failing transport cancels the connecting task from inside its own task
    group, so the exception raised at the connect site is usually just a
    ``CancelledError``; the real cause — an HTTP error, a refused connection —
    surfaces separately as the exit stack unwinds. Groups wrapping a single
    cause are unwrapped so callers see that cause directly.

    Args:
        candidates: Exceptions seen while connecting, most informative first.

    Returns:
        The first candidate that is a regular exception, or ``None`` if every
        candidate is a cancellation.
    """
    for candidate in candidates:
        if candidate is None:
            continue
        cause = _sole_cause(candidate)
        if isinstance(cause, Exception):
            return cause
    return None


class _ToolWrapperWithCleanup:
    """Tool call wrapper that carries the cleanup releasing its backing resource.

    The LLM service registers this wrapper like any other handler and awaits
    ``_pipecat_cleanup`` when the service is cleaned up, so a resource that
    outlives individual calls — an MCP server connection — is released at
    pipeline teardown without the developer wiring teardown manually. A class
    rather than a closure so the cleanup is a declared attribute.
    """

    def __init__(
        self,
        call_tool: Callable[[FunctionCallParams], Awaitable[None]],
        cleanup: Callable[[], Awaitable[None]],
    ) -> None:
        """Initialize the wrapper.

        Args:
            call_tool: Invoked for each tool call.
            cleanup: Awaited once at LLM service teardown. Expected to be
                idempotent, and equal across wrappers sharing a resource so the
                service records it once.
        """
        self._call_tool = call_tool
        self._pipecat_cleanup = cleanup

    async def __call__(self, params: FunctionCallParams) -> None:
        """Handle a tool call."""
        await self._call_tool(params)


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

    A tool call that finds its connection gone drops that session and
    returns the error to the model, and the call after it connects again.

    Call :meth:`close` only to release the connection earlier than pipeline
    teardown (e.g. from an ``on_client_disconnected`` handler). A closed client
    stays closed until :meth:`start` or :meth:`tools` opens it again, and a
    tool call on one raises. :meth:`start` and :meth:`close` may be called from
    different tasks — the session is owned by a dedicated internal task.
    Scoping the client with ``async with MCPClient(...)`` is also supported.

    Raises:
        TypeError: If server_params is not a supported parameter type.
    """

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
        # One wrapper shared by every tool this client advertises, carrying the
        # cleanup that releases the connection.
        self._tool_wrapper_with_cleanup = _ToolWrapperWithCleanup(self._tool_wrapper, self.close)
        # The MCP session is anyio-task-bound: it must be opened and closed in the
        # same task. start() and close() can be called from different tasks, so a
        # dedicated owner task (_run_session) holds the session open and tears it
        # down on signal.
        self._session_task: asyncio.Task | None = None
        self._closing: asyncio.Event | None = None
        # One lock owns a start, a close, a drop and the connect a tool call
        # needs. The owner task clears the session fields as it exits.
        self._lifecycle_lock = asyncio.Lock()
        # True between start() and close(). A tool call reconnects only a
        # client that lost its session, because only that client has an owner.
        self._session_requested = False

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
        await self._open_session(request=True)

    async def _open_session(self, *, request: bool) -> ClientSession:
        """Return the session the caller runs on, opening one if the client holds none.

        Holds the lifecycle lock across the start and the read, so a close
        cannot take the session away in between.

        Args:
            request: True when the caller owns the connection. A tool call
                passes False and uses the session an owner asked for.

        Raises:
            _NotConnectedError: If no owner has asked the client for a session.
        """
        async with self._lifecycle_lock:
            if request:
                self._session_requested = True
            if self._session_requested:
                await self._start_locked()
            return self._ensure_connected()

    async def _start_locked(self) -> None:
        """Open the session, with the lifecycle lock held."""
        if self._active_session:
            return
        self._closing = asyncio.Event()
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        self._session_task = asyncio.create_task(
            self._run_session(ready, self._closing), name=f"{self}::session"
        )
        try:
            await ready  # resolves once connected; re-raises a connect failure
        except BaseException:
            task, self._session_task = self._session_task, None  # retry cleanly later
            # Cancellation reaches us as a cancelled `ready`. A session that
            # connected releases itself. One still waiting on an unresponsive
            # server holds the transport until this task is cancelled, and
            # nothing else can reach this task once the handle is dropped.
            if ready.cancelled() and self._active_session is None:
                task.cancel()
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
                # Indexed rather than unpacked: the transport yields three
                # stream elements on the SDK's 1.x line and two on 2.x.
                streams = await exit_stack.enter_async_context(
                    _streamable_http_transport(self._server_params)
                )
                read_stream, write_stream = streams[0], streams[1]

            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
        except BaseException as raised:
            # A failing transport cancels this task from inside its own task
            # group, so `raised` is typically a CancelledError and the cause
            # worth reporting only surfaces as the exit stack unwinds. Settling
            # `ready` either way is what keeps start() from waiting forever.
            unwound: BaseException | None = None
            try:
                await exit_stack.aclose()
            except BaseException as e:
                unwound = e
            if not ready.cancelled():
                ready.set_exception(
                    _connect_failure_cause(unwound, raised)
                    or ConnectionError(f"{self} could not connect to the MCP server")
                )
            if isinstance(raised, asyncio.CancelledError):
                raise
            return

        self._exit_stack = exit_stack
        self._active_session = session
        try:
            if ready.cancelled():
                # The caller was cancelled mid-connect, so nobody will ever call
                # close() for this session: release it here instead of holding a
                # connection no one owns.
                return
            ready.set_result(None)
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

        The client stays closed until :meth:`start` or :meth:`tools` opens it
        again: a tool call on a closed client raises rather than connect.
        """
        async with self._lifecycle_lock:
            self._session_requested = False
            await self._close_locked()

    async def _close_locked(self) -> None:
        """Release the session, with the lifecycle lock held."""
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
        session = await self._open_session(request=True)
        return await self._list_tools_helper(session, attach_handlers=True)

    @deprecated(
        "`MCPClient.register_tools` is deprecated since 1.8.0 and will be removed in 2.0.0. "
        "Use `MCPClient.tools` instead."
    )
    async def register_tools(self, llm: LLMService | LLMSwitcher) -> ToolsSchema:
        """Register all available MCP tools with an LLM service.

        .. deprecated:: 1.8.0
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
            llm.register_function(function_schema.name, self._tool_wrapper_with_cleanup)
        return tools_schema

    def _ensure_connected(self) -> ClientSession:
        """Return the active session or raise if not connected."""
        if not self._active_session:
            raise _NotConnectedError(
                "MCPClient is not connected. Use 'async with MCPClient(...) as mcp:' "
                "or call 'await mcp.start()' before using MCPClient."
            )
        return self._active_session

    @deprecated(
        "`MCPClient.get_tools_schema` is deprecated since 1.8.0 and will be removed in 2.0.0. "
        "Use `MCPClient.tools` instead."
    )
    async def get_tools_schema(self) -> ToolsSchema:
        """Get the schema of all available MCP tools without registering them.

        .. deprecated:: 1.8.0
            Use :meth:`tools` instead. Will be removed in 2.0.0.

        Requires the client to be started via start() or async with.

        Returns:
            A ToolsSchema containing all available tools.
        """
        session = self._ensure_connected()
        return await self._list_tools_helper(session)

    @deprecated(
        "`MCPClient.register_tools_schema` is deprecated since 1.8.0 and will be removed in "
        "2.0.0. Use `MCPClient.tools` instead."
    )
    async def register_tools_schema(
        self, tools_schema: ToolsSchema, llm: LLMService | LLMSwitcher
    ) -> None:
        """Register previously obtained MCP tools with the LLM service.

        .. deprecated:: 1.8.0
            Use :meth:`tools` instead — its schemas carry handlers that
            register automatically. Will be removed in 2.0.0.

        Args:
            tools_schema: The ToolsSchema to register with the LLM service.
            llm: The Pipecat LLM service to register tools with.
        """
        for function_schema in tools_schema.standard_tools:
            llm.register_function(function_schema.name, self._tool_wrapper_with_cleanup)

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
        logger.debug(f"Executing tool '{params.function_name}' with call ID: {params.tool_call_id}")
        logger.trace(f"Tool arguments: {json.dumps(params.arguments, indent=2)}")
        await self._call_tool(params.function_name, params.arguments, params.result_callback)

    async def _call_tool(self, function_name, arguments, result_callback):
        response = await self._call_tool_text(function_name, arguments)
        await result_callback(response)

    async def _call_tool_text(self, function_name, arguments) -> str:
        fixed = self._tools_arguments.get(function_name)
        if fixed:
            # Fixed arguments win over model-supplied values.
            arguments = {**(arguments or {}), **fixed}

        logger.debug(f"Calling mcp tool '{function_name}'")
        session = None
        results = None
        error_msg = None
        try:
            session = await self._open_session(request=False)
            results = await session.call_tool(function_name, arguments=arguments)
        except _NotConnectedError:
            # The call reached no session. That is a wiring mistake, so it
            # goes to the caller, not to the model.
            raise
        except Exception as e:
            error_msg = f"Error calling mcp tool {function_name}: {_error_detail(e)}"
            logger.error(error_msg)
            if session is not None and _is_connection_lost(e):
                logger.warning(f"{self} dropping the session that tool '{function_name}' ran on")
                try:
                    await self._drop_session(session)
                except Exception as drop_error:
                    # The model gets the cause of its call. A transport that
                    # fails on close goes to the log only.
                    logger.error(f"{self} error dropping the dead session: {drop_error}")

        response = ""
        if results:
            if results.content:
                for i, content in enumerate(results.content):
                    # The session is typed, so the result is too: only TextContent
                    # carries a text field, on either SDK line.
                    if isinstance(content, TextContent) and content.text:
                        logger.debug(f"Tool response chunk {i}: {content.text}")
                        response += content.text
                    else:
                        logger.debug(
                            f"Tool response chunk {i} carries no text: {type(content).__name__}"
                        )
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

        if isinstance(response, str) and response:
            logger.info(f"Tool '{function_name}' completed successfully")
            logger.debug(f"Final response: {response}")
        else:
            response = error_msg or "Sorry, could not call the mcp tool"

        return response

    async def _drop_session(self, session: ClientSession) -> None:
        """Drop the session a failed call ran on, so a later call connects again.

        A session outlives its transport, so each later call runs on a dead
        session until something drops it. Tool calls run concurrently, so a
        caller whose session another call already replaced drops nothing.
        """
        async with self._lifecycle_lock:
            if self._active_session is session:
                await self._close_locked()

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
                # The SDK spells this field inputSchema on 1.x, input_schema on 2.x.
                input_schema = (
                    tool.input_schema if hasattr(tool, "input_schema") else tool.inputSchema
                )
                function_schema = self._convert_mcp_schema_to_pipecat(
                    tool_name,
                    {"description": tool.description, "input_schema": input_schema},
                    handler=self._tool_wrapper_with_cleanup if attach_handlers else None,
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
