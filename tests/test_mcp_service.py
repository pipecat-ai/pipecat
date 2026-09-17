#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the MCP client service."""

import asyncio
import io
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

# MCP is an optional dependency (the `mcp` extra); skip the whole module if it
# isn't installed.
pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp.client.session import ClientSession  # noqa: E402
from mcp.client.session_group import StreamableHttpParameters  # noqa: E402
from mcp.shared import exceptions as mcp_exceptions  # noqa: E402
from mcp.shared.memory import create_client_server_memory_streams  # noqa: E402
from mcp.types import (  # noqa: E402
    CONNECTION_CLOSED,
    CallToolResult,
    ErrorData,
    JSONRPCResponse,
    TextContent,
)

from pipecat.services import mcp_service  # noqa: E402
from pipecat.services.llm_service import LLMService  # noqa: E402
from pipecat.services.mcp_service import MCPClient  # noqa: E402

# The SDK spells the error class McpError on its 1.x line and MCPError on 2.x.
_MCPError = getattr(mcp_exceptions, "MCPError", None) or mcp_exceptions.McpError


def _tool(name, properties=None, required=None, description="A tool.", schema_field="inputSchema"):
    """Build a fake MCP server tool as returned by ``session.list_tools()``.

    ``schema_field`` selects which spelling of the schema attribute the tool
    carries: the SDK spells it inputSchema on 1.x and input_schema on 2.x.
    """
    return SimpleNamespace(
        name=name,
        description=description,
        **{schema_field: {"properties": properties or {}, "required": required or []}},
    )


class _FakeServer:
    """Fake MCP server, minting one transport and one session per connect.

    It outlives every session it mints, so what the test scripts and what the
    client did both survive a dropped session. Tests set its attributes to
    script a failure; ``on_connect``, ``on_call_once`` and ``on_exit_once`` run inside a
    connect, inside a tool call and inside a disconnect, to drive what those
    race with. The two once hooks run for one call and one disconnect, so a
    hook that closes the client does not meet itself on the way out.
    """

    def __init__(self, tools):
        self.tools = tools
        self.fail_initializes = 0
        self.cancel_initialize = False
        self.transport_exit_error = None
        self.connect_delay = 0
        self.call_errors = []
        self.empty_results = False
        self.non_text_results = False
        self.on_connect = None
        self.on_call_once = None
        self.on_exit_once = None
        self.sessions = []
        self.calls = []
        self.enters = 0
        self.exits = 0
        self.initializes = 0
        self.enter_task = None
        self.exit_task = None

    def connect(self, read_stream, write_stream):
        """Mint the session for one connection, as ``ClientSession(...)`` does."""
        session = _FakeSession(self)
        self.sessions.append(session)
        return session


class _FakeSession:
    """Fake mcp ClientSession, serving what its server holds."""

    def __init__(self, server):
        self._server = server

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def initialize(self):
        server = self._server
        if server.cancel_initialize:
            # An anyio transport cancels the connecting task when its own
            # request fails, so initialize() ends in cancellation.
            raise asyncio.CancelledError("Cancelled via cancel scope")
        if server.fail_initializes > 0:
            server.fail_initializes -= 1
            raise RuntimeError("connect failed")
        server.initializes += 1

    async def list_tools(self):
        return SimpleNamespace(tools=self._server.tools)

    async def call_tool(self, name, arguments=None):
        server = self._server
        server.calls.append((name, arguments))
        if server.on_call_once:
            hook, server.on_call_once = server.on_call_once, None
            await hook()
        if server.call_errors:
            raise server.call_errors.pop(0)
        if server.empty_results:
            return CallToolResult(content=[])
        if server.non_text_results:
            # A block of another content type, carrying a text field the tool
            # response must still leave alone. No SDK block carries text outside
            # TextContent, so the result stands in for a future one that does.
            return SimpleNamespace(content=[SimpleNamespace(text=f"{name}-IMAGE")])
        return CallToolResult(content=[TextContent(type="text", text=f"{name}-RESULT")])


class _FakeTransport:
    """Fake streamable-HTTP transport; records its enter and exit on the server."""

    def __init__(self, server):
        self._server = server

    async def __aenter__(self):
        server = self._server
        server.enters += 1
        server.enter_task = asyncio.current_task()
        if server.on_connect:
            await server.on_connect()
        if server.connect_delay:
            # A connect window wide enough to cancel the caller inside it.
            await asyncio.sleep(server.connect_delay)
        return (MagicMock(), MagicMock(), MagicMock())

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        server = self._server
        server.exits += 1
        server.exit_task = asyncio.current_task()
        if server.on_exit_once:
            hook, server.on_exit_once = server.on_exit_once, None
            await hook()
        if server.transport_exit_error is not None:
            # A real transport reports a failed connection as its task group
            # unwinds, rather than at the connect site.
            raise server.transport_exit_error
        return False


def _connection_closed_error():
    """Build the error an SDK raises for a request its transport could not serve."""
    message = "Connection closed"
    try:
        return _MCPError(code=CONNECTION_CLOSED, message=message)
    except TypeError:
        # The 1.x line takes the error data as one argument.
        return _MCPError(ErrorData(code=CONNECTION_CLOSED, message=message))


def _connection_lost_errors():
    """The errors an SDK raises once the transport behind a session is gone.

    A transport that fails inside its own task group reports through a group of
    one, so the same causes arrive wrapped as well as bare.
    """
    return [
        _connection_closed_error(),
        anyio.ClosedResourceError(),
        anyio.BrokenResourceError(),
        anyio.EndOfStream(),
        ExceptionGroup("unhandled errors in a TaskGroup", [anyio.ClosedResourceError()]),
    ]


class MCPClientTestBase(unittest.IsolatedAsyncioTestCase):
    """Builds MCPClients against a fake MCP server."""

    def _make_client(self, tools, **client_kwargs):
        server = _FakeServer(tools)
        ctx = patch.multiple(
            "pipecat.services.mcp_service",
            _streamable_http_transport=lambda params: _FakeTransport(server),
            ClientSession=server.connect,
        )
        ctx.start()
        self.addCleanup(ctx.stop)
        client = MCPClient(
            server_params=StreamableHttpParameters(url="http://test/mcp"),
            **client_kwargs,
        )
        self.addAsyncCleanup(client.close)
        return client, server

    async def _call_via_handler(self, tools_schema, name, arguments=None):
        """Invoke a tool the way the LLM service does, through its handler.

        Returns:
            The result callback, so callers can assert on what was delivered.
        """
        schema = next(s for s in tools_schema.standard_tools if s.name == name)
        params = SimpleNamespace(
            function_name=name,
            tool_call_id="call-1",
            arguments=arguments,
            result_callback=AsyncMock(),
        )
        await schema.handler(params)
        return params.result_callback


class TestTools(MCPClientTestBase):
    """tools(): JIT start, handler attachment, filtering."""

    async def test_tools_starts_connection_and_attaches_handlers(self):
        client, server = self._make_client([_tool("tool_a"), _tool("tool_b")])
        tools_schema = await client.tools()
        self.assertEqual(server.initializes, 1)
        self.assertEqual({s.name for s in tools_schema.standard_tools}, {"tool_a", "tool_b"})
        for schema in tools_schema.standard_tools:
            self.assertIsNotNone(schema.handler)
        await client.close()

    async def test_tools_reads_either_schema_field_spelling(self):
        """The SDK spells the schema field inputSchema in 1.x, input_schema in 2.x."""
        for schema_field in ("inputSchema", "input_schema"):
            with self.subTest(schema_field=schema_field):
                client, _ = self._make_client(
                    [
                        _tool(
                            "tool_a",
                            properties={"x": {"type": "string"}},
                            required=["x"],
                            schema_field=schema_field,
                        )
                    ]
                )
                tools_schema = await client.tools()
                schema = tools_schema.standard_tools[0]
                self.assertEqual(schema.properties, {"x": {"type": "string"}})
                self.assertEqual(schema.required, ["x"])
                await client.close()

    async def test_tools_is_idempotent_on_connection(self):
        client, server = self._make_client([_tool("tool_a")])
        await client.tools()
        await client.tools()
        self.assertEqual(server.enters, 1)
        self.assertEqual(server.initializes, 1)
        await client.close()

    async def test_tools_respects_tools_filter(self):
        client, server = self._make_client(
            [_tool("tool_a"), _tool("tool_b")], tools_filter=["tool_b"]
        )
        tools_schema = await client.tools()
        self.assertEqual({s.name for s in tools_schema.standard_tools}, {"tool_b"})
        await client.close()

    async def test_handler_calls_tool_and_delivers_result(self):
        client, server = self._make_client([_tool("tool_a", properties={"x": {"type": "string"}})])
        tools_schema = await client.tools()
        schema = tools_schema.standard_tools[0]
        params = SimpleNamespace(
            function_name="tool_a",
            tool_call_id="call-1",
            arguments={"x": "hello"},
            result_callback=AsyncMock(),
        )
        await schema.handler(params)
        self.assertEqual(server.calls, [("tool_a", {"x": "hello"})])
        params.result_callback.assert_awaited_once_with("tool_a-RESULT")
        await client.close()

    async def test_handler_applies_output_filter(self):
        client, server = self._make_client(
            [_tool("tool_a")], tools_output_filters={"tool_a": lambda text: text.upper()}
        )
        tools_schema = await client.tools()
        result_callback = await self._call_via_handler(tools_schema, "tool_a", {"x": "y"})
        result_callback.assert_awaited_once_with("TOOL_A-RESULT")
        await client.close()


class TestToolsArguments(MCPClientTestBase):
    """tools_arguments: schema stripping and call-time injection."""

    def _search_tools(self):
        return [
            _tool(
                "search",
                properties={"query": {"type": "string"}, "mode": {"type": "string"}},
                required=["query", "mode"],
            ),
            _tool("other", properties={"x": {"type": "string"}}, required=["x"]),
        ]

    async def test_fixed_arguments_hidden_from_advertised_schema(self):
        client, server = self._make_client(
            self._search_tools(), tools_arguments={"search": {"mode": "realtime"}}
        )
        tools_schema = await client.tools()
        by_name = {s.name: s for s in tools_schema.standard_tools}
        self.assertEqual(set(by_name["search"].properties), {"query"})
        self.assertEqual(by_name["search"].required, ["query"])
        # Other tools are untouched.
        self.assertEqual(set(by_name["other"].properties), {"x"})
        self.assertEqual(by_name["other"].required, ["x"])
        await client.close()

    async def test_fixed_arguments_injected_and_win_over_model_arguments(self):
        client, server = self._make_client(
            self._search_tools(), tools_arguments={"search": {"mode": "realtime"}}
        )
        tools_schema = await client.tools()
        await self._call_via_handler(
            tools_schema, "search", {"query": "news", "mode": "model-supplied"}
        )
        self.assertEqual(server.calls, [("search", {"query": "news", "mode": "realtime"})])
        await client.close()

    async def test_fixed_arguments_injected_when_no_model_arguments(self):
        client, server = self._make_client(
            self._search_tools(), tools_arguments={"search": {"mode": "realtime"}}
        )
        tools_schema = await client.tools()
        await self._call_via_handler(tools_schema, "search")
        self.assertEqual(server.calls, [("search", {"mode": "realtime"})])
        await client.close()

    async def test_fixed_argument_absent_from_server_schema_still_injected(self):
        client, server = self._make_client(
            self._search_tools(), tools_arguments={"other": {"hidden": 1}}
        )
        tools_schema = await client.tools()
        by_name = {s.name: s for s in tools_schema.standard_tools}
        # Stripping a name the schema doesn't have is a no-op...
        self.assertEqual(set(by_name["other"].properties), {"x"})
        # ...but the argument is still injected at call time.
        await self._call_via_handler(tools_schema, "other", {"x": "y"})
        self.assertEqual(server.calls, [("other", {"x": "y", "hidden": 1})])
        await client.close()


class TestCallErrors(MCPClientTestBase):
    """Failed tool calls: the error reaches the model, a dead session is dropped."""

    async def test_a_lost_connection_drops_the_session_and_runs_the_tool_once(self):
        for error in _connection_lost_errors():
            with self.subTest(error=type(error).__name__):
                client, server = self._make_client([_tool("tool_a")])
                tools_schema = await client.tools()
                server.call_errors.append(error)
                await self._call_via_handler(tools_schema, "tool_a")
                # The client does not run the call again: the SDK reports a
                # request that never left and a lost answer as one error.
                self.assertEqual(len(server.calls), 1)
                self.assertEqual(server.enters, 1)
                self.assertIsNone(client._active_session)
                await client.close()

    async def test_a_failed_call_returns_the_error_to_the_model(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        server.call_errors.append(RuntimeError("upstream unavailable"))
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with(
            "Error calling mcp tool tool_a: upstream unavailable"
        )
        await client.close()

    async def test_a_lost_connection_returns_the_error_to_the_model(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        server.call_errors.append(_connection_closed_error())
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with("Error calling mcp tool tool_a: Connection closed")
        await client.close()

    async def test_an_error_with_no_message_gives_its_class_name(self):
        # str(ClosedResourceError()) is empty, so the model would otherwise read
        # a line that ends at the colon.
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        server.call_errors.append(anyio.ClosedResourceError())
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with(
            "Error calling mcp tool tool_a: ClosedResourceError"
        )
        await client.close()

    async def test_a_group_of_one_gives_the_cause_it_wraps(self):
        # The group itself says only that a task group failed.
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        server.call_errors.append(
            ExceptionGroup("unhandled errors in a TaskGroup", [anyio.BrokenResourceError()])
        )
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with(
            "Error calling mcp tool tool_a: BrokenResourceError"
        )
        await client.close()

    async def test_a_call_after_a_lost_connection_connects_again(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        server.call_errors.append(_connection_closed_error())
        await self._call_via_handler(tools_schema, "tool_a")

        # The model decides to call again, and that call opens its own transport.
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with("tool_a-RESULT")
        self.assertEqual(server.enters, 2)
        self.assertEqual(len(server.calls), 2)
        await client.close()

    async def test_other_errors_return_the_error_and_keep_the_session(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        server.call_errors.append(ValueError("upstream unavailable"))
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with(
            "Error calling mcp tool tool_a: upstream unavailable"
        )
        self.assertEqual(server.enters, 1)
        self.assertIs(client._active_session, server.sessions[-1])
        await client.close()

    async def test_dropping_a_session_another_call_replaced_closes_nothing(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        replaced = server.sessions[0]
        server.call_errors.append(_connection_closed_error())
        await self._call_via_handler(tools_schema, "tool_a")
        await self._call_via_handler(tools_schema, "tool_a")
        self.assertEqual(server.enters, 2)

        # Tool calls run concurrently, so a second call can still hold the dead
        # session the first one dropped.
        await client._drop_session(replaced)
        self.assertIs(client._active_session, server.sessions[-1])
        self.assertEqual(server.exits, 1)
        await client.close()

    async def test_a_closed_client_reports_the_lost_connection(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()

        async def close_before_the_call_fails():
            await client.close()

        server.on_call_once = close_before_the_call_fails
        server.call_errors.append(_connection_closed_error())
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        # The client stays closed. The call reports the connection it lost,
        # which differs from the setup error of a client that never started.
        result_callback.assert_awaited_once_with("Error calling mcp tool tool_a: Connection closed")
        self.assertEqual(server.enters, 1)

    async def test_a_teardown_meeting_a_drop_leaves_nothing_open(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        teardown = None

        async def close_while_dropping():
            nonlocal teardown
            teardown = asyncio.create_task(client.close())
            await asyncio.sleep(0)  # let the teardown reach the lock it waits on

        server.on_exit_once = close_while_dropping
        server.call_errors.append(_connection_closed_error())
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        await teardown
        # The teardown waits out the drop and finds the session already released.
        result_callback.assert_awaited_once_with("Error calling mcp tool tool_a: Connection closed")
        self.assertEqual(server.exits, server.enters)
        self.assertIsNone(client._active_session)
        self.assertIsNone(client._session_task)

    async def test_a_group_of_two_causes_still_drops_the_session(self):
        # The group names no single cause, so the model reads the group itself.
        # The transport under it is gone either way.
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        server.call_errors.append(
            ExceptionGroup(
                "unhandled errors in a TaskGroup",
                [anyio.ClosedResourceError(), ValueError("and one more")],
            )
        )
        await self._call_via_handler(tools_schema, "tool_a")
        self.assertIsNone(client._active_session)
        self.assertEqual(server.enters, 1)
        await client.close()

    async def test_a_drop_whose_transport_fails_keeps_the_cause_of_the_call(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        server.transport_exit_error = RuntimeError("transport exit failed")
        server.call_errors.append(_connection_closed_error())
        result_callback = await self._call_via_handler(tools_schema, "tool_a")

        # The model reads the cause of its own call. The transport failure on
        # close goes to the log. The client holds no session, so the next call
        # connects.
        result_callback.assert_awaited_once_with("Error calling mcp tool tool_a: Connection closed")
        self.assertIsNone(client._active_session)
        self.assertIsNone(client._session_task)
        server.transport_exit_error = None
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with("tool_a-RESULT")
        await client.close()

    async def test_a_failed_connect_after_a_drop_reports_its_error_and_heals(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        server.call_errors.append(_connection_closed_error())
        await self._call_via_handler(tools_schema, "tool_a")

        # The connect the next call runs fails, and reports why.
        server.fail_initializes = 1
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with("Error calling mcp tool tool_a: connect failed")

        # The call after it connects and runs.
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with("tool_a-RESULT")
        await client.close()

    async def test_an_empty_result_still_reads_as_the_stock_line(self):
        client, server = self._make_client([_tool("tool_a")])
        server.empty_results = True
        tools_schema = await client.tools()
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with("Sorry, could not call the mcp tool")
        await client.close()

    async def test_a_long_error_is_cut_to_size(self):
        # The line goes into the LLM context, so it is cut to a fixed length.
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        server.call_errors.append(ValueError("x" * (mcp_service._MAX_ERROR_DETAIL * 2)))
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with(
            f"Error calling mcp tool tool_a: {'x' * mcp_service._MAX_ERROR_DETAIL}..."
        )
        await client.close()

    async def test_a_result_of_another_content_type_reads_as_the_stock_line(self):
        client, server = self._make_client([_tool("tool_a")])
        server.non_text_results = True
        tools_schema = await client.tools()
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with("Sorry, could not call the mcp tool")
        await client.close()


class TestSDKConnectionErrors(unittest.IsolatedAsyncioTestCase):
    """What a real SDK session raises once the transport under it is gone."""

    async def test_a_call_on_a_closed_transport_reads_as_a_lost_connection(self):
        # The SDK exposes no liveness on its session, on either line, so a lost
        # connection is read from the error a call raises. The class of that
        # error differs by line: anyio on 1.x, the SDK's own on 2.x.
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            read_stream, write_stream = client_streams
            async with ClientSession(read_stream, write_stream) as session:
                for stream in (*server_streams, *client_streams):
                    await stream.aclose()
                await asyncio.sleep(0)
                with self.assertRaises(Exception) as ctx:
                    await session.call_tool("tool_a", arguments={})
        self.assertTrue(mcp_service._is_connection_lost(ctx.exception))


async def _serve(server_streams):
    """Answer one client over memory streams, the way an MCP server does.

    Enough of a server to initialize, advertise one tool and run it. The two SDK
    lines wrap a message differently, so the reply goes back in the envelope the
    request arrived in.
    """
    read_stream, write_stream = server_streams
    async for message in read_stream:
        request = getattr(message.message, "root", message.message)
        if getattr(request, "method", None) is None or not hasattr(request, "id"):
            continue  # a notification, which takes no reply
        if request.method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-server", "version": "1"},
            }
        elif request.method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "tool_a",
                        "description": "A tool.",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            }
        else:
            result = {"content": [{"type": "text", "text": "tool_a-RESULT"}]}
        response = JSONRPCResponse(jsonrpc="2.0", id=request.id, result=result)
        envelope = type(message.message) if hasattr(message.message, "root") else None
        await write_stream.send(type(message)(message=envelope(response) if envelope else response))


class TestRealSession(MCPClientTestBase):
    """The client over a real ClientSession, with a real server behind it."""

    def _make_real_client(self):
        """Build a client whose transport mints real memory streams per connect.

        Returns:
            The client, and the server end of each connection it opens.
        """
        connections = []

        @asynccontextmanager
        async def transport(params):
            async with create_client_server_memory_streams() as (client_streams, server_streams):
                serving = asyncio.create_task(_serve(server_streams))
                connections.append(server_streams)
                try:
                    yield client_streams
                finally:
                    serving.cancel()

        ctx = patch.object(mcp_service, "_streamable_http_transport", transport)
        ctx.start()
        self.addCleanup(ctx.stop)
        client = MCPClient(server_params=StreamableHttpParameters(url="http://test/mcp"))
        self.addAsyncCleanup(client.close)
        return client, connections

    async def test_a_real_transport_that_dies_drops_its_session_and_reconnects(self):
        client, connections = self._make_real_client()
        tools_schema = await client.tools()
        result_callback = await self._call_via_handler(tools_schema, "tool_a", {})
        result_callback.assert_awaited_once_with("tool_a-RESULT")

        # Kill the server end of the live connection, as a dropped connection does.
        for stream in connections[-1]:
            await stream.aclose()

        result_callback = await self._call_via_handler(tools_schema, "tool_a", {})
        delivered = result_callback.await_args.args[0]
        self.assertIn("Error calling mcp tool tool_a:", delivered)
        self.assertIsNone(client._active_session)

        # The call after it connects again, on a transport of its own.
        result_callback = await self._call_via_handler(tools_schema, "tool_a", {})
        result_callback.assert_awaited_once_with("tool_a-RESULT")
        self.assertEqual(len(connections), 2)
        await client.close()


class TestLifecycle(MCPClientTestBase):
    """start/close: task safety, retry, idempotency."""

    async def test_close_from_other_task_exits_transport_in_owner_task(self):
        client, server = self._make_client([_tool("tool_a")])
        await client.tools()
        # Close from a different task than the one that started the client (as
        # an on_client_disconnected handler would).
        await asyncio.create_task(client.close())
        self.assertEqual(server.exits, 1)
        # The anyio-based transport requires enter/exit in the same task.
        self.assertIs(server.enter_task, server.exit_task)
        self.assertIsNot(server.enter_task, asyncio.current_task())

    async def test_failed_start_raises_and_allows_retry(self):
        client, server = self._make_client([_tool("tool_a")])
        server.fail_initializes = 1
        with self.assertRaises(RuntimeError):
            await client.start()
        # The partially opened transport was cleaned up.
        self.assertEqual(server.exits, 1)
        # A later call retries cleanly.
        await client.start()
        self.assertEqual(server.initializes, 1)
        await client.close()

    async def test_cancelling_transport_raises_its_underlying_cause(self):
        # A transport whose request fails cancels the connecting task and reports
        # the cause only as it unwinds; tools() must raise that cause rather than
        # wait on a connection that will never arrive.
        cause = RuntimeError("Client error '401 Unauthorized'")
        client, server = self._make_client([_tool("tool_a")])
        server.cancel_initialize = True
        server.transport_exit_error = ExceptionGroup("unhandled errors in a TaskGroup", [cause])
        with self.assertRaises(RuntimeError) as ctx:
            await asyncio.wait_for(client.tools(), timeout=5)
        self.assertIs(ctx.exception, cause)
        self.assertEqual(server.exits, 1)

    async def test_cancelling_transport_without_a_cause_still_raises(self):
        # Nothing surfaced on unwind, so there's no cause to report — but the
        # caller still gets an error instead of waiting forever.
        client, server = self._make_client([_tool("tool_a")])
        server.cancel_initialize = True
        with self.assertRaises(ConnectionError):
            await asyncio.wait_for(client.tools(), timeout=5)

    async def test_cancelling_the_caller_stops_a_connect_in_flight(self):
        # Cancelling the task awaiting start() cancels the future it waits on. A
        # session still inside the connect can't see that, so start() stops it —
        # an unresponsive server would otherwise hold the transport (a spawned
        # server process) with nothing left able to reach it.
        client, server = self._make_client([_tool("tool_a")])
        server.connect_delay = 3600
        caller = asyncio.create_task(client.tools())
        await asyncio.sleep(0.05)  # inside the connect window
        session_task = client._session_task
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(session_task, timeout=5)
        self.assertIsNone(client._active_session)
        self.assertIsNone(client._session_task)
        # Nothing is left for close() to do, and it says so quietly.
        await client.close()

    async def test_connection_nobody_awaits_is_released(self):
        # The connect can still land in the window between the caller's
        # cancellation and start() reacting to it, leaving a session with no one
        # to hand it to. It has to release itself rather than sit open.
        client, server = self._make_client([_tool("tool_a")])
        ready = asyncio.get_running_loop().create_future()
        ready.cancel()

        await client._run_session(ready, asyncio.Event())

        self.assertEqual(server.exits, 1)
        self.assertIsNone(client._active_session)

    async def test_failed_connect_nobody_awaits_settles_quietly(self):
        # Same window, but the connect fails: with no one waiting, the failure has
        # nowhere to go and must not become an error of its own.
        client, server = self._make_client([_tool("tool_a")])
        server.fail_initializes = 1
        ready = asyncio.get_running_loop().create_future()
        ready.cancel()

        await client._run_session(ready, asyncio.Event())

        self.assertEqual(server.exits, 1)
        self.assertIsNone(client._active_session)

    async def test_close_before_start_and_double_close_are_safe(self):
        client, server = self._make_client([_tool("tool_a")])
        await client.close()
        await client.start()
        await client.close()
        await client.close()
        self.assertEqual(server.exits, 1)

    async def test_calling_a_tool_after_close_raises(self):
        # A closed client stays closed. A transport opened here has no owner
        # to close it.
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        await client.close()
        with self.assertRaises(RuntimeError):
            await self._call_via_handler(tools_schema, "tool_a", {})
        self.assertEqual(server.enters, 1)

    async def test_calling_a_tool_before_start_raises(self):
        # No owner has asked for a session, so the call opens no transport.
        # The owner must start the client.
        client, server = self._make_client([_tool("tool_a")])
        params = SimpleNamespace(
            function_name="tool_a",
            tool_call_id="call-1",
            arguments={},
            result_callback=AsyncMock(),
        )
        with self.assertRaises(RuntimeError):
            await client._tool_wrapper(params)
        self.assertEqual(server.enters, 0)
        params.result_callback.assert_not_awaited()

    async def test_tools_after_close_reconnects(self):
        client, server = self._make_client([_tool("tool_a")])
        await client.tools()
        await client.close()
        tools_schema = await client.tools()
        self.assertEqual(server.enters, 2)
        self.assertEqual({s.name for s in tools_schema.standard_tools}, {"tool_a"})
        await client.close()

    async def test_a_start_behind_a_close_leaves_a_live_session(self):
        # The lock serialises the two, so the one that reaches it last decides.
        # A start that lands behind a close has a session to hand the next call.
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        await asyncio.gather(client.close(), client.start())
        self.assertEqual(server.enters, 2)
        self.assertEqual(server.exits, 1)
        result_callback = await self._call_via_handler(tools_schema, "tool_a")
        result_callback.assert_awaited_once_with("tool_a-RESULT")
        await client.close()

    async def test_a_close_behind_a_start_leaves_nothing_open(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        await asyncio.gather(client.start(), client.close())
        self.assertEqual(server.exits, server.enters)
        self.assertIsNone(client._active_session)
        self.assertIsNone(client._session_task)
        # The client is closed, so the call raises and opens no transport.
        with self.assertRaises(RuntimeError):
            await self._call_via_handler(tools_schema, "tool_a")
        self.assertEqual(server.enters, 1)

    async def test_concurrent_starts_open_one_connection(self):
        client, server = self._make_client([_tool("tool_a")])
        await asyncio.gather(client.start(), client.start(), client.start())
        self.assertEqual(server.enters, 1)
        await client.close()

    async def test_async_with_still_works(self):
        client, server = self._make_client([_tool("tool_a")])
        async with client as mcp:
            tools_schema = await mcp.tools()
            self.assertEqual({s.name for s in tools_schema.standard_tools}, {"tool_a"})
        self.assertEqual(server.exits, 1)


class TestDeprecatedRegistrationApi(MCPClientTestBase):
    """register_tools/register_tools_schema/get_tools_schema are deprecated but work."""

    async def test_get_tools_schema_warns_and_returns_handlerless_schemas(self):
        client, server = self._make_client([_tool("tool_a")])
        await client.start()
        with self.assertWarns(DeprecationWarning):
            tools_schema = await client.get_tools_schema()
        self.assertEqual({s.name for s in tools_schema.standard_tools}, {"tool_a"})
        for schema in tools_schema.standard_tools:
            self.assertIsNone(schema.handler)
        await client.close()

    async def test_register_tools_warns_and_registers(self):
        client, server = self._make_client([_tool("tool_a"), _tool("tool_b")])
        await client.start()
        llm = MagicMock()
        with self.assertWarns(DeprecationWarning):
            tools_schema = await client.register_tools(llm)
        self.assertEqual(llm.register_function.call_count, 2)
        registered = {call.args[0] for call in llm.register_function.call_args_list}
        self.assertEqual(registered, {"tool_a", "tool_b"})
        for schema in tools_schema.standard_tools:
            self.assertIsNone(schema.handler)
        await client.close()

    async def test_register_tools_schema_warns_and_registers(self):
        client, server = self._make_client([_tool("tool_a")])
        await client.start()
        tools_schema = await client.tools()
        llm = MagicMock()
        with self.assertWarns(DeprecationWarning):
            await client.register_tools_schema(tools_schema, llm)
        llm.register_function.assert_called_once()
        await client.close()


class TestAutoCloseOnCleanup(MCPClientTestBase):
    """LLMService.cleanup() closes clients whose handlers were registered."""

    async def test_cleanup_closes_registered_client(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        service = LLMService()
        service._sync_registered_tool_handlers(tools_schema)
        await service.cleanup()
        self.assertEqual(server.exits, 1)

    async def test_cleanup_twice_is_safe(self):
        client, server = self._make_client([_tool("tool_a")])
        service = LLMService()
        service._sync_registered_tool_handlers(await client.tools())
        await service.cleanup()
        await service.cleanup()
        self.assertEqual(server.exits, 1)

    async def test_two_services_sharing_client_close_idempotently(self):
        # e.g. two LLMs behind a switcher advertising the same context tools.
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        service_a, service_b = LLMService(), LLMService()
        service_a._sync_registered_tool_handlers(tools_schema)
        service_b._sync_registered_tool_handlers(tools_schema)
        await service_a.cleanup()
        await service_b.cleanup()
        self.assertEqual(server.exits, 1)

    async def test_no_close_when_handlers_never_registered(self):
        # Known gap: a connected client the LLM service never learned about
        # (no inference ran) is not auto-closed.
        client, server = self._make_client([_tool("tool_a")])
        await client.tools()
        service = LLMService()
        await service.cleanup()
        self.assertEqual(server.exits, 0)
        await client.close()
        self.assertEqual(server.exits, 1)

    async def test_deprecated_register_tools_path_also_auto_closes(self):
        client, server = self._make_client([_tool("tool_a")])
        await client.start()
        service = LLMService()
        with self.assertWarns(DeprecationWarning):
            await client.register_tools(service)
        await service.cleanup()
        self.assertEqual(server.exits, 1)

    async def test_client_survives_tool_pruning_until_cleanup(self):
        # De-advertising a tool prunes its handler but must not close the
        # session mid-conversation; the close happens at teardown.
        client, server = self._make_client([_tool("tool_a")])
        service = LLMService()
        service._sync_registered_tool_handlers(await client.tools())
        service._sync_registered_tool_handlers([])  # tool set replaced
        self.assertEqual(server.exits, 0)
        await service.cleanup()
        self.assertEqual(server.exits, 1)


class TestLLMAutoRegistration(MCPClientTestBase):
    """End-to-end with a real LLMService: tools() auto-registers, old path doesn't warn."""

    async def test_tools_auto_register_with_llm_service(self):
        client, server = self._make_client([_tool("tool_a")])
        tools_schema = await client.tools()
        service = LLMService()
        service._sync_registered_tool_handlers(tools_schema)
        self.assertTrue(service.has_function("tool_a"))
        self.assertTrue(service._functions["tool_a"].auto_registered)
        await client.close()

    async def test_deprecated_register_tools_path_does_not_warn_redundant(self):
        client, server = self._make_client([_tool("tool_a")])
        await client.start()
        service = LLMService()
        with self.assertWarns(DeprecationWarning):
            tools_schema = await client.register_tools(service)
        # Advertising the handler-less schemas alongside the manual registration
        # must not trip the redundant-manual-registration advisory.
        sink = io.StringIO()
        handler_id = logger.add(sink, level="WARNING", format="{message}")
        try:
            service._sync_registered_tool_handlers(tools_schema)
        finally:
            logger.remove(handler_id)
        self.assertEqual(sink.getvalue(), "")
        self.assertTrue(service.has_function("tool_a"))
        await client.close()


if __name__ == "__main__":
    unittest.main()
